import streamlit as st
import cv2
import numpy as np
import os
import time
import json
import tempfile
import shutil
import mimetypes
import subprocess
from datetime import datetime
from ultralytics import YOLO
import queue
import threading
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 扩展MIME类型映射
ADDITIONAL_MIME_TYPES = {
    '.mp4': 'video/mp4',
    '.mov': 'video/quicktime',
    '.avi': 'video/x-msvideo',
    '.mkv': 'video/x-matroska',
    '.flv': 'video/x-flv',
    '.wmv': 'video/x-ms-wmv',
    '.mpeg': 'video/mpeg',
    '.mpg': 'video/mpeg'
}

# 注册额外的MIME类型
for ext, mime in ADDITIONAL_MIME_TYPES.items():
    mimetypes.add_type(mime, ext)


# 检查FFmpeg是否安装
def check_ffmpeg():
    try:
        subprocess.run(
            ['ffmpeg', '-version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        st.success("已检测到FFmpeg，准备就绪")
        return True
    except FileNotFoundError:
        st.error("未找到FFmpeg可执行文件，请检查是否已安装并添加到系统环境变量")
        return False
    except Exception as e:
        st.error(f"FFmpeg检查失败: {str(e)}")
        return False


# 执行两次FFmpeg转换：使用简化命令
def convert_and_compress_avi(avi_input_path, progress_queue):
    """将YOLO生成的AVI通过两次FFmpeg转换为700k比特率的MP4，使用简化命令"""
    if not os.path.exists(avi_input_path):
        error_msg = f"输入文件不存在: {avi_input_path}"
        progress_queue.put(("error", error_msg))
        return False, error_msg, None

    if os.path.getsize(avi_input_path) < 1024:
        error_msg = f"输入文件过小，可能损坏: {avi_input_path}"
        progress_queue.put(("error", error_msg))
        return False, error_msg, None

    # 创建输出文件路径（与AVI同目录，同名不同扩展名）
    avi_dir = os.path.dirname(avi_input_path)
    avi_filename = os.path.basename(avi_input_path)
    base_name = os.path.splitext(avi_filename)[0]
    intermediate_mp4 = os.path.join(avi_dir, f"{base_name}_temp.mp4")
    final_mp4 = os.path.join(avi_dir, f"{base_name}.mp4")

    try:
        # 第一步：复制流到MP4（不重新编码）
        progress_queue.put(("log", f"开始格式转换: {avi_filename} -> MP4"))
        ffmpeg_cmd1 = [
            'ffmpeg',
            '-v', 'error',
            '-y',
            '-i', avi_input_path,
            '-c', 'copy',
            intermediate_mp4
        ]

        result1 = subprocess.run(
            ffmpeg_cmd1,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        if result1.returncode != 0 or not os.path.exists(intermediate_mp4):
            error_msg = f"格式转换失败: {result1.stdout}"
            progress_queue.put(("error", error_msg))
            return False, error_msg, None

        # 第二步：压缩到700k比特率
        progress_queue.put(("log", f"开始压缩视频至700k比特率"))
        ffmpeg_cmd2 = [
            'ffmpeg',
            '-v', 'error',
            '-y',
            '-i', intermediate_mp4,
            '-b:v', '700k',
            final_mp4
        ]

        result2 = subprocess.run(
            ffmpeg_cmd2,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        if result2.returncode != 0 or not os.path.exists(final_mp4):
            error_msg = f"视频压缩失败: {result2.stdout}"
            progress_queue.put(("error", error_msg))
            return False, error_msg, None

        # 验证输出文件
        if os.path.getsize(final_mp4) < 1024:
            error_msg = f"生成的输出文件为空: {final_mp4}"
            progress_queue.put(("error", error_msg))
            return False, error_msg, None

        # 清理中间文件和原始AVI
        if os.path.exists(intermediate_mp4):
            os.remove(intermediate_mp4)
        if os.path.exists(avi_input_path):
            os.remove(avi_input_path)
            progress_queue.put(("log", f"已删除原始AVI文件: {avi_filename}"))

        # 转换成功
        mp4_size_mb = os.path.getsize(final_mp4) / (1024 * 1024)
        success_msg = f"视频处理完成：{os.path.basename(final_mp4)}（{mp4_size_mb:.1f}MB）"
        progress_queue.put(("log", success_msg))
        return True, success_msg, final_mp4

    except Exception as e:
        error_msg = f"处理异常: {str(e)}"
        progress_queue.put(("error", error_msg))
        if os.path.exists(intermediate_mp4):
            os.remove(intermediate_mp4)
        return False, error_msg, None


# 设置页面配置
st.set_page_config(
    page_title="视频文本检测",
    page_icon="🎥",
    layout="wide"
)

# 检查FFmpeg是否安装
if not check_ffmpeg():
    st.error("⚠️ FFmpeg检查失败，无法继续进行视频处理")
    st.markdown("""
    请按照以下步骤解决：
    1. 确认FFmpeg已正确安装
    2. 检查FFmpeg是否已添加到系统环境变量
    3. 重启计算机使环境变量生效
    4. 重新运行程序
    """)
    st.stop()

# 确保必要目录存在
os.makedirs("custom_models", exist_ok=True)
os.makedirs("detection_results", exist_ok=True)
os.makedirs("detection_logs", exist_ok=True)
os.makedirs("detectvideo_results", exist_ok=True)

# 初始化session_state
if "model" not in st.session_state:
    st.session_state.model = None
if "model_path" not in st.session_state:
    st.session_state.model_path = ""
if "video_detection_result" not in st.session_state:
    st.session_state.video_detection_result = None
if "processing_folder" not in st.session_state:
    st.session_state.processing_folder = None  # 用于跟踪当前处理的文件夹


def cleanup_temp_files(*paths):
    """清理临时文件和目录"""
    for path in paths:
        try:
            if path and os.path.exists(path):
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
        except Exception as e:
            st.warning(f"清理临时文件失败: {str(e)}")


def get_video_mime_type(file_path):
    """获取视频文件的MIME类型"""
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith('video/'):
        ext = os.path.splitext(file_path)[1].lower()
        mime_type = ADDITIONAL_MIME_TYPES.get(ext, 'video/unknown')
    return mime_type


def load_model(model_path: str) -> YOLO:
    """加载模型并预热"""
    if st.session_state.model is not None and st.session_state.model_path == model_path:
        return st.session_state.model

    try:
        with st.spinner(f"正在加载模型 {os.path.basename(model_path)}..."):
            model = YOLO(model_path)
            # 模型预热
            warmup_img = np.zeros((640, 640, 3), dtype=np.uint8)
            model.track(
                source=warmup_img,
                imgsz=640,
                save=False,
                show=False,
                tracker='ultralytics/cfg/trackers/botsort.yaml',
                verbose=False
            )

            st.session_state.model = model
            st.session_state.model_path = model_path
            st.success(f"模型加载完成: {os.path.basename(model_path)}")
            return model
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None


def process_video_frames(model: YOLO, video_path: str, conf_threshold: float, frame_interval: int,
                         progress_queue: queue.Queue):
    """处理视频帧（YOLO生成AVI后，自动调用FFmpeg转换为700k MP4）"""
    try:
        # 1. 生成唯一处理文件夹并保存到session_state
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processing_folder = os.path.join("detectvideo_results", f"track_{timestamp}")
        st.session_state.processing_folder = processing_folder  # 保存当前处理文件夹

        # 打开视频获取基础信息
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            progress_queue.put(("error", "无法打开视频文件"))
            return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        # YOLO处理：生成AVI到指定文件夹
        results = model.track(
            source=video_path,
            imgsz=640,
            project=processing_folder,
            name="exp",  # 固定子目录名称
            save=True,
            show=False,
            tracker='ultralytics/cfg/trackers/botsort.yaml',
            conf=conf_threshold,
            stream=True,
            verbose=False
        )

        # 2. 处理结果流，更新进度
        frame_count = 0
        detection_stats = {}
        has_detections = False
        for result in results:
            frame_count += 1

            # 统计检测结果
            if frame_count % frame_interval == 0:
                if hasattr(result, 'boxes'):
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    for cls_id in classes:
                        cls_name = result.names[cls_id]
                        if any(keyword in cls_name.lower() for keyword in ["text", "word", "character"]):
                            has_detections = True
                            detection_stats[cls_name] = detection_stats.get(cls_name, 0) + 1

            # 更新进度
            if total_frames > 0 and (frame_count % 5 == 0 or frame_count == total_frames):
                progress = min(frame_count / total_frames, 1.0)
                progress_queue.put(("progress", progress))

        # 3. 找到YOLO生成的AVI文件 - 使用session_state中保存的文件夹路径
        yolo_avi_path = None
        exp_dir = os.path.join(processing_folder, "exp")
        if os.path.exists(exp_dir):
            avi_files = [f for f in os.listdir(exp_dir) if f.endswith(".avi")]
            if avi_files:
                yolo_avi_path = os.path.join(exp_dir, avi_files[0])
                avi_size_mb = os.path.getsize(yolo_avi_path) / (1024 * 1024)
                progress_queue.put(("log", f"YOLO生成视频：{os.path.basename(yolo_avi_path)}（{avi_size_mb:.1f}MB）"))

                if os.path.getsize(yolo_avi_path) < 1024:
                    progress_queue.put(("error", f"生成的AVI文件无效（过小）：{os.path.basename(yolo_avi_path)}"))
                    return
            else:
                progress_queue.put(("error", "YOLO未生成AVI文件"))
                return
        else:
            progress_queue.put(("error", f"YOLO输出目录不存在: {exp_dir}"))
            return

        # 4. 调用FFmpeg进行两次转换
        success, msg, final_mp4_path = convert_and_compress_avi(yolo_avi_path, progress_queue)

        if not success:
            progress_queue.put(("error", f"视频处理失败：{msg}"))
            return

        # 5. 验证最终MP4文件
        if not os.path.exists(final_mp4_path) or os.path.getsize(final_mp4_path) < 1024:
            progress_queue.put(("error", "压缩后的MP4文件无效"))
            return

        # 6. 通知处理完成
        progress_queue.put(("complete", final_mp4_path, detection_stats,
                            total_frames, fps, has_detections, processing_folder))

    except Exception as e:
        progress_queue.put(("error", f"处理错误: {str(e)}"))


def save_detection_results(video_path: str, stats: dict, total_frames: int,
                           fps: float, video_name: str, has_detections: bool) -> tuple[str, str]:
    """保存压缩后的MP4结果"""
    try:
        # 使用与处理时相同的文件夹名称中的时间戳
        if st.session_state.processing_folder:
            timestamp = os.path.basename(st.session_state.processing_folder).replace("track_", "")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        video_basename = os.path.splitext(video_name)[0]
        safe_basename = "".join([c for c in video_basename if c.isalnum() or c in "._- "])

        # 保存压缩后的MP4
        dest_path = os.path.join("detection_results", f"{safe_basename}_compressed_{timestamp}.mp4")
        shutil.copy2(video_path, dest_path)

        if os.path.getsize(dest_path) != os.path.getsize(video_path):
            raise Exception("压缩视频保存不完整")

        # 保存元数据
        meta_path = os.path.join("detection_logs", f"{safe_basename}_meta_{timestamp}.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": timestamp,
                "video_name": video_name,
                "compressed_video_path": dest_path,
                "compressed_size_mb": os.path.getsize(dest_path) / (1024 * 1024),
                "duration": total_frames / fps if fps > 0 else 0,
                "total_frames": total_frames,
                "fps": fps,
                "detection_stats": stats,
                "total_detections": sum(stats.values()) if stats else 0,
                "has_detections": has_detections
            }, f, indent=2, ensure_ascii=False)

        return dest_path, meta_path

    except Exception as e:
        st.error(f"保存结果失败: {str(e)}")
        if 'dest_path' in locals() and os.path.exists(dest_path):
            os.remove(dest_path)
        if 'meta_path' in locals() and os.path.exists(meta_path):
            os.remove(meta_path)
        return None, None


def main():
    st.title("🎥 视频文本检测")
    st.markdown("自动处理视频并生成压缩后的检测结果")

    # 侧边栏配置
    with st.sidebar:
        st.header("模型配置")
        model_file = st.file_uploader("上传自定义模型 (.pt)", type=["pt"])
        model_path = None
        if model_file:
            model_path = os.path.join("custom_models", model_file.name)
            with open(model_path, "wb") as f:
                f.write(model_file.getbuffer())
            load_model(model_path)
        else:
            st.info("请上传你的YOLO模型文件 (.pt)")

        # 推理参数
        st.header("推理参数")
        conf_threshold = st.slider("置信度阈值", 0.01, 1.0, 0.3, 0.05)
        frame_interval = st.slider("检测统计间隔（每N帧统计一次）", 1, 120, 60)

    # 主内容区
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("输入视频")
        uploaded_file = st.file_uploader(
            "上传视频文件",
            type=[ext[1:] for ext in ADDITIONAL_MIME_TYPES.keys()]
        )

        # 限制视频大小
        if uploaded_file and uploaded_file.size > 100 * 1024 * 1024:
            st.error("视频文件过大，请上传小于100MB的视频")
            uploaded_file = None

        video_info = None
        temp_path = None
        if uploaded_file:
            # 保存上传的视频到临时文件
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name

            # 获取视频信息
            cap = cv2.VideoCapture(temp_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps if fps > 0 else 0
                mime_type = get_video_mime_type(temp_path)

                video_info = {
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "total_frames": total_frames,
                    "duration": duration,
                    "name": uploaded_file.name,
                    "mime_type": mime_type
                }
                cap.release()

                # 显示原始视频信息
                st.video(uploaded_file)
                st.info(f"""
                视频信息:
                - 格式: {file_ext} ({mime_type})
                - 分辨率: {width}x{height}
                - 帧率: {fps:.1f} FPS
                - 时长: {duration:.1f}秒
                """)
            else:
                st.error("无法解析视频文件")
                cleanup_temp_files(temp_path)
                temp_path = None

        detect_btn = st.button("开始检测", use_container_width=True)

    with col2:
        st.subheader("检测结果")
        result_container = st.container()

        # 执行检测
        if detect_btn and st.session_state.model and uploaded_file and temp_path:
            with result_container:
                st.session_state.video_detection_result = None

                # 清除之前的处理文件夹（如果存在）
                if st.session_state.processing_folder and os.path.exists(st.session_state.processing_folder):
                    cleanup_temp_files(st.session_state.processing_folder)

                # 显示进度
                progress_bar = st.progress(0)
                status_text = st.text("准备处理视频...")
                log_area = st.empty()

                # 创建进度队列
                progress_queue = queue.Queue()

                # 启动处理线程
                start_time = time.time()
                processing_thread = threading.Thread(
                    target=process_video_frames,
                    args=(st.session_state.model, temp_path, conf_threshold, frame_interval,
                          progress_queue)
                )
                processing_thread.start()

                # 主线程更新进度和处理结果
                output_video_path = None
                detection_stats = None
                total_frames = 0
                fps = 0
                has_detections = False
                output_dir = None
                logs = []
                error_occurred = False

                while processing_thread.is_alive():
                    try:
                        item = progress_queue.get(timeout=0.5)

                        if item[0] == "progress":
                            _, progress = item
                            progress_bar.progress(progress)
                            status_text.text(f"处理进度: {int(progress * 100)}%")
                        elif item[0] == "log":
                            _, log_msg = item
                            logs.append(log_msg)
                            log_area.text("\n".join(logs))
                        elif item[0] == "complete":
                            _, output_video_path, detection_stats, total_frames, fps, has_detections, output_dir = item
                        elif item[0] == "error":
                            _, error_msg = item
                            st.error(error_msg)
                            error_occurred = True
                            processing_thread.join()
                            break
                    except queue.Empty:
                        continue

                processing_thread.join()
                end_time = time.time()

                # 保存结果到session_state
                if output_video_path and os.path.exists(output_video_path) and os.path.getsize(
                        output_video_path) > 1024:
                    st.session_state.video_detection_result = {
                        "video_path": output_video_path,
                        "stats": detection_stats,
                        "total_frames": total_frames,
                        "fps": fps,
                        "video_name": uploaded_file.name,
                        "has_detections": has_detections,
                        "processing_time": end_time - start_time,
                        "output_dir": output_dir
                    }
                elif not error_occurred:
                    # 只在没有发生其他错误但文件无效时才显示提示
                    st.warning("视频处理已完成")

                # 显示处理结果信息
                if st.session_state.video_detection_result:
                    output_video_path = st.session_state.video_detection_result["video_path"]
                    processing_time = st.session_state.video_detection_result["processing_time"]
                    file_size_mb = os.path.getsize(output_video_path) / (1024 * 1024)

                    # 获取相对路径
                    rel_video_path = os.path.relpath(output_video_path)

                    status_text.text("处理完成")

                    # 显示状态信息
                    if has_detections:
                        st.success(
                            f"视频处理完成 | 耗时: {processing_time:.2f}秒 | "
                            f"总检测次数: {sum(detection_stats.values())} | "
                            f"文件大小: {file_size_mb:.1f}MB"
                        )
                    else:
                        st.info(
                            f"视频处理完成 | 耗时: {processing_time:.2f}秒 | "
                            f"未检测到文本 | 文件大小: {file_size_mb:.1f}MB"
                        )

                    # 显示视频
                    with open(output_video_path, "rb") as f:
                        st.video(f)

                    # 显示视频文件路径和访问说明
                    st.info(f"检测结果视频已保存至：`{rel_video_path}`")

                    # 保存结果
                    if st.button("保存结果", use_container_width=True):
                        with st.spinner("正在保存结果..."):
                            result = st.session_state.video_detection_result
                            media_path, meta_path = save_detection_results(
                                output_video_path,
                                detection_stats,
                                total_frames,
                                fps,
                                result["video_name"],
                                has_detections
                            )

                            if media_path and meta_path:
                                rel_media_path = os.path.relpath(media_path)
                                st.success(f"结果已保存至: `{rel_media_path}`")

                                with open(media_path, "rb") as f:
                                    st.download_button(
                                        "下载检测结果视频",
                                        f,
                                        file_name=os.path.basename(media_path),
                                        mime=get_video_mime_type(media_path),
                                        use_container_width=True
                                    )
                else:
                    # 不显示"未生成有效的检测结果视频"错误，只在有明确错误时显示
                    if error_occurred:
                        cleanup_temp_files(temp_path, output_dir)
        else:
            with result_container:
                if detect_btn:
                    if not uploaded_file:
                        st.warning("请先上传视频")
                    elif not st.session_state.model:
                        st.warning("模型加载失败，请检查模型文件")
                    elif not temp_path:
                        st.warning("无法处理视频文件，请重试")
                else:
                    st.info("上传视频并点击检测按钮查看结果")


if __name__ == "__main__":
    main()
