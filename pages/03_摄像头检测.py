import streamlit as st
import cv2
import numpy as np
import os
import time
import json
import tempfile
from datetime import datetime
import pandas as pd
import queue
import matplotlib.pyplot as plt
from ultralytics import YOLO
import threading

# 设置页面配置
st.set_page_config(
    page_title="摄像头检测 - YOLOv11",
    page_icon="📹",
    layout="wide"
)

# 确保必要目录存在
os.makedirs("custom_models", exist_ok=True)
os.makedirs("detection_results", exist_ok=True)
os.makedirs("detection_logs", exist_ok=True)

# 初始化session_state
if "model" not in st.session_state:
    st.session_state.model = None
if "model_path" not in st.session_state:
    st.session_state.model_path = "yolov11s.pt"
if "detection_active" not in st.session_state:
    st.session_state.detection_active = False
if "recording_active" not in st.session_state:
    st.session_state.recording_active = False
if "camera_recording_result" not in st.session_state:
    st.session_state.camera_recording_result = None


def load_model(model_path: str) -> YOLO:
    """加载模型并预热"""
    if st.session_state.model is not None and st.session_state.model_path == model_path:
        return st.session_state.model

    try:
        with st.spinner(f"正在加载模型 {os.path.basename(model_path)}..."):
            model = YOLO(model_path)
            # 模型预热
            warmup_img = np.zeros((640, 640, 3), dtype=np.uint8)
            model(warmup_img, conf=0.5, verbose=False)

            st.session_state.model = model
            st.session_state.model_path = model_path
            st.success(f"模型加载完成: {os.path.basename(model_path)}")
            return model
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None


def camera_detection_loop(model: YOLO, conf_threshold: float, frame_queue, stop_event, record_event):
    """摄像头检测循环"""
    # 打开摄像头
    cap = cv2.VideoCapture(0)  # 0表示默认摄像头

    # 获取摄像头参数
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # 默认为30FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 视频录制相关
    out = None
    temp_file = None
    detection_stats = {}
    frame_count = 0

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                frame_queue.put(("error", "无法获取摄像头画面"))
                break

            # 推理
            results = model(frame, conf=conf_threshold, verbose=False)

            # 可视化
            visualized_frame = results[0].plot(conf=True, line_width=2)

            # 更新统计信息
            frame_count += 1
            for box in results[0].boxes:
                cls_id = int(box.cls)
                cls_name = results[0].names[cls_id]
                if any(keyword in cls_name.lower() for keyword in ["text", "word", "character"]):
                    if cls_name in detection_stats:
                        detection_stats[cls_name] += 1
                    else:
                        detection_stats[cls_name] = 1

            # 处理录制
            if record_event.is_set():
                if out is None:
                    # 创建临时文件
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(temp_file.name, fourcc, fps, (width, height))

                out.write(visualized_frame)
            elif out is not None:
                # 停止录制时释放资源
                out.release()
                out = None

            # 发送处理后的帧到主线程
            frame_queue.put(("frame", visualized_frame, detection_stats.copy(), frame_count, fps))

            # 控制帧率
            time.sleep(0.01)

    finally:
        # 释放资源
        cap.release()
        if out is not None:
            out.release()

        # 发送结束信号
        if temp_file:
            frame_queue.put(("recording_complete", temp_file.name, detection_stats, frame_count, fps))
        frame_queue.put(("done",))


def save_camera_recording(video_path: str, stats: dict, frame_count: int, fps: float) -> tuple[str, str]:
    """保存摄像头录制结果（与图片/视频检测保持一致的逻辑）"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"camera_recording_{timestamp}"

    # 复制录制的视频
    dest_path = os.path.join("detection_results", f"{base_name}.mp4")
    with open(video_path, 'rb') as f_in, open(dest_path, 'wb') as f_out:
        f_out.write(f_in.read())

    # 保存元数据（格式与图片/视频检测一致）
    meta_path = os.path.join("detection_logs", f"{base_name}_meta.json")
    with open(meta_path, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "original_name": f"camera_recording_{timestamp}",
            "media_path": dest_path,
            "duration": frame_count / fps if fps > 0 else 0,
            "total_frames": frame_count,
            "fps": fps,
            "detection_stats": stats,
            "total_detections": sum(stats.values()) if stats else 0
        }, f, indent=2)

    return dest_path, meta_path


def main():
    st.title("📹 摄像头文本检测")
    st.markdown("实时摄像头文本检测与录制")

    # 侧边栏配置 - 优化后的模型设置
    with st.sidebar:
        st.header("模型配置")

        # 上传自定义模型
        model_file = st.file_uploader("上传自定义模型 (.pt)", type=["pt"])
        model_path = None
        model = None

        if model_file:
            # 保存上传的模型文件
            model_path = os.path.join("custom_models", model_file.name)
            with open(model_path, "wb") as f:
                f.write(model_file.getbuffer())

            # 加载模型
            model = load_model(model_path)
        else:
            # 未上传模型时显示提示信息
            st.info("请上传你的YOLO模型文件 (.pt)")

        # 推理参数
        st.header("推理参数")
        conf_threshold = st.slider("置信度阈值", 0.01, 1.0, 0.5, 0.05)

        # 显示选项
        st.header("显示选项")
        show_stats = st.checkbox("显示实时统计", value=True)

    # 主内容区
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("摄像头控制")

        # 控制按钮
        col_btn1, col_btn2 = st.columns(2)
        start_btn = col_btn1.button("开始检测", use_container_width=True)
        stop_btn = col_btn2.button("停止检测", use_container_width=True)

        # 录制控制
        record_btn = st.button("开始/停止录制", use_container_width=True)

        # 状态显示
        status_placeholder = st.empty()

        if st.session_state.detection_active:
            status_placeholder.info("状态: 正在检测...")
        else:
            status_placeholder.info("状态: 未检测")

        if st.session_state.recording_active:
            status_placeholder.error("状态: 正在检测并录制...")

    with col2:
        st.subheader("实时检测画面")
        frame_placeholder = st.empty()
        stats_placeholder = st.empty()

        # 处理开始检测
        if start_btn and not st.session_state.detection_active and st.session_state.model:
            st.session_state.detection_active = True
            st.session_state.recording_active = False
            st.session_state.camera_recording_result = None

            # 创建通信队列和事件
            frame_queue = queue.Queue()
            stop_event = threading.Event()
            record_event = threading.Event()

            # 启动检测线程
            detection_thread = threading.Thread(
                target=camera_detection_loop,
                args=(st.session_state.model, conf_threshold, frame_queue, stop_event, record_event)
            )
            detection_thread.start()

            # 主线程处理画面更新
            start_time = time.time()

            while st.session_state.detection_active:
                # 检查是否需要停止
                if stop_btn:
                    stop_event.set()
                    st.session_state.detection_active = False
                    st.session_state.recording_active = False
                    record_event.clear()

                # 检查录制状态
                if record_btn:
                    st.session_state.recording_active = not st.session_state.recording_active
                    if st.session_state.recording_active:
                        record_event.set()
                    else:
                        record_event.clear()
                    time.sleep(0.5)  # 防止重复触发

                # 获取处理结果
                try:
                    item = frame_queue.get(timeout=0.1)

                    if item[0] == "frame":
                        _, frame, stats, frame_count, fps = item
                        # 显示画面
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, use_container_width=True)

                        # 显示统计
                        if show_stats and stats:
                            with stats_placeholder.container():
                                st.subheader("实时检测统计")
                                stats_df = pd.DataFrame(list(stats.items()), columns=["文本类别", "检测次数"])
                                st.dataframe(stats_df, use_container_width=True)

                    elif item[0] == "error":
                        frame_placeholder.error(item[1])
                        break

                    elif item[0] == "recording_complete":
                        _, recorded_video_path, stats, frame_count, fps = item
                        st.session_state.camera_recording_result = {
                            "video_path": recorded_video_path,
                            "stats": stats,
                            "frame_count": frame_count,
                            "fps": fps
                        }

                    elif item[0] == "done":
                        break

                except queue.Empty:
                    continue

            # 等待线程结束
            detection_thread.join()
            end_time = time.time()

            # 显示结束信息
            frame_placeholder.info(f"检测已停止 | 持续时间: {end_time - start_time:.2f}秒")

            # 处理录制文件（与其他检测保持一致的保存逻辑）
            if st.session_state.camera_recording_result:
                with stats_placeholder.container():
                    st.success("录制已完成")

                    # 保存选项
                    if st.button("保存录制结果", use_container_width=True):
                        with st.spinner("正在保存结果..."):
                            result = st.session_state.camera_recording_result
                            media_path, meta_path = save_camera_recording(
                                result["video_path"],
                                result["stats"],
                                result["frame_count"],
                                result["fps"]
                            )
                            st.success(f"录制结果已保存至: {media_path}")

                            with open(media_path, "rb") as f:
                                st.download_button(
                                    "下载录制视频",
                                    f,
                                    file_name=os.path.basename(media_path),
                                    use_container_width=True
                                )

        elif start_btn and not st.session_state.model:
            frame_placeholder.warning("模型加载失败，请检查模型文件")
        elif start_btn and st.session_state.detection_active:
            frame_placeholder.info("检测已在进行中")
        else:
            frame_placeholder.info("点击开始检测按钮启动摄像头")


if __name__ == "__main__":
    main()
