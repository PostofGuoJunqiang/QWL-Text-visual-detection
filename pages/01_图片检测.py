import streamlit as st
import cv2
import numpy as np
import os
import time
import json
import zipfile
from datetime import datetime
import pandas as pd
from ultralytics import YOLO

# 设置页面配置
st.set_page_config(
    page_title="图片检测 - YOLOv11",
    page_icon="🖼️",
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
    st.session_state.model_path = ""
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []
if "preprocessed_imgs" not in st.session_state:
    st.session_state.preprocessed_imgs = []
if "detection_completed" not in st.session_state:
    st.session_state.detection_completed = False
if "last_upload_count" not in st.session_state:
    st.session_state.last_upload_count = 0


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


def batch_preprocess_images(uploaded_files):
    """批量预处理图片"""
    preprocessed_list = []
    progress = st.progress(0)
    status_text = st.empty()

    for i, file in enumerate(uploaded_files):
        status_text.text(f"预处理第 {i + 1}/{len(uploaded_files)} 张图片...")

        try:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                raise Exception("无法解析图片")

            # 转换为RGB
            if image.shape[-1] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # 限制最大尺寸
            max_size = 1280
            h, w = image_rgb.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                image_rgb = cv2.resize(image_rgb, (int(w * scale), int(h * scale)))

            preprocessed_list.append({
                "original_name": file.name,
                "image_rgb": image_rgb,
                "valid": True
            })
        except Exception as e:
            preprocessed_list.append({
                "original_name": file.name,
                "error": str(e),
                "valid": False
            })

        progress.progress((i + 1) / len(uploaded_files))
        file.seek(0)

    progress.empty()
    status_text.empty()
    return preprocessed_list


def detect_batch(model, preprocessed_imgs, conf_threshold):
    """批量检测图片"""
    results = []
    total = len(preprocessed_imgs)
    progress = st.progress(0)
    status_text = st.empty()

    for i, img_data in enumerate(preprocessed_imgs):
        if not img_data["valid"]:
            results.append(None)
            progress.progress((i + 1) / total)
            continue

        status_text.text(f"检测第 {i + 1}/{total} 张图片: {img_data['original_name']}")

        try:
            # 执行检测
            start_time = time.time()
            detect_result = model(
                img_data["image_rgb"],
                conf=conf_threshold,
                imgsz=640,
                verbose=False,
                device="auto"
            )
            processing_time = time.time() - start_time

            # 可视化结果
            visualized_img = detect_result[0].plot(conf=True, line_width=2)

            # 解析检测结果
            detected_texts = []
            for box in detect_result[0].boxes:
                cls_id = int(box.cls)
                cls_name = model.names[cls_id]
                if any(keyword in cls_name.lower() for keyword in ["text", "word", "character"]):
                    detected_texts.append({
                        "class": cls_name,
                        "confidence": float(box.conf),
                        "bbox": tuple(map(int, box.xyxy[0])),
                        "width": int(box.xyxy[0][2] - box.xyxy[0][0]),
                        "height": int(box.xyxy[0][3] - box.xyxy[0][1])
                    })

            # 保存结果文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name_base = os.path.splitext(img_data["original_name"])[0]
            media_path = os.path.join("detection_results", f"{name_base}_det_{timestamp}.jpg")
            meta_path = os.path.join("detection_logs", f"{name_base}_meta_{timestamp}.json")

            cv2.imwrite(media_path, visualized_img)
            with open(meta_path, 'w') as f:
                json.dump({
                    "timestamp": timestamp,
                    "original_name": img_data["original_name"],
                    "detections": detected_texts,
                    "count": len(detected_texts),
                    "processing_time": round(processing_time, 2)
                }, f, indent=2)

            results.append({
                "original_name": img_data["original_name"],
                "visualized_img": visualized_img,
                "detected_texts": detected_texts,
                "processing_time": processing_time,
                "media_path": media_path,
                "meta_path": meta_path
            })
        except Exception as e:
            st.error(f"处理 {img_data['original_name']} 时出错: {str(e)}")
            results.append(None)

        progress.progress((i + 1) / total)

    progress.empty()
    status_text.empty()
    return results


def save_batch_zip(batch_results):
    """批量保存为ZIP"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = os.path.join("detection_results", f"batch_{timestamp}.zip")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for res in batch_results:
            if res is not None:
                zipf.write(res["media_path"], os.path.basename(res["media_path"]))
                zipf.write(res["meta_path"], os.path.basename(res["meta_path"]))

    return zip_path


def main():
    st.title("🖼️ 图片文本检测")
    st.markdown("支持批量图片检测与文本识别")

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
        # 加载模型
        model = None
        if model_path:
            model = load_model(model_path)

        # 推理参数
        st.header("推理参数")
        conf_threshold = st.slider("置信度阈值", 0.2, 1.0, 0.5, 0.05)
        max_images = st.slider("最大批量处理数量", 1, 20, 5)

    # 主内容区
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("上传图片")
        uploaded_files = st.file_uploader(
            "支持格式: JPG, PNG, BMP",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True
        )

        # 限制上传数量
        if uploaded_files and len(uploaded_files) > max_images:
            uploaded_files = uploaded_files[:max_images]
            st.warning(f"已自动限制为 {max_images} 张图片")

        # 预处理图片
        if uploaded_files and len(uploaded_files) != st.session_state.last_upload_count:
            st.session_state.preprocessed_imgs = batch_preprocess_images(uploaded_files)
            st.session_state.last_upload_count = len(uploaded_files)
            st.session_state.detection_completed = False
            st.session_state.batch_results = []

        # 显示预处理结果
        valid_count = sum(1 for img in st.session_state.preprocessed_imgs if img.get("valid", False))
        if st.session_state.preprocessed_imgs:
            st.info(f"已预处理 {valid_count}/{len(st.session_state.preprocessed_imgs)} 张有效图片")

            # 显示缩略图
            if valid_count > 0:
                cols = st.columns(3)
                for i, img_data in enumerate(st.session_state.preprocessed_imgs[:9]):
                    if img_data.get("valid", False):
                        with cols[i % 3]:
                            thumb = cv2.resize(img_data["image_rgb"], (100, 100))
                            st.image(thumb, caption=img_data["original_name"], use_container_width=True)

        # 检测按钮
        if uploaded_files and model and valid_count > 0:
            if st.button("开始批量检测", use_container_width=True, type="primary"):
                st.session_state.detection_completed = False
                with st.spinner("正在进行批量检测，请稍候..."):
                    st.session_state.batch_results = detect_batch(
                        model,
                        st.session_state.preprocessed_imgs,
                        conf_threshold
                    )
                st.session_state.detection_completed = True
                st.success("批量检测完成！")

    with col2:
        st.subheader("检测结果")

        if st.session_state.detection_completed and st.session_state.batch_results:
            # 过滤无效结果
            valid_results = [res for res in st.session_state.batch_results if res is not None]

            if not valid_results:
                st.error("没有有效的检测结果")
            else:
                # 选择要查看的图片
                selected_idx = st.selectbox(
                    "选择图片",
                    range(len(valid_results)),
                    format_func=lambda x: valid_results[x]["original_name"]
                )
                current = valid_results[selected_idx]

                # 显示检测结果
                st.image(
                    current["visualized_img"],
                    caption=f"{current['original_name']} (耗时: {current['processing_time']:.2f}s)",
                    use_container_width=True
                )

                # 显示检测详情
                with st.expander("查看检测详情", expanded=False):
                    if current["detected_texts"]:
                        st.dataframe(
                            pd.DataFrame(current["detected_texts"]),
                            use_container_width=True
                        )
                    else:
                        st.info("未检测到文本")

                # 保存和下载功能
                col_save1, col_save2 = st.columns(2)
                with col_save1:
                    if st.button("保存当前结果", use_container_width=True):
                        with open(current["media_path"], "rb") as f:
                            st.download_button(
                                "下载当前图片",
                                f,
                                file_name=os.path.basename(current["media_path"]),
                                use_container_width=True
                            )

                with col_save2:
                    if len(valid_results) > 1 and st.button("批量保存所有结果", use_container_width=True):
                        with st.spinner("正在打包所有结果..."):
                            zip_path = save_batch_zip(valid_results)
                        st.success(f"已打包 {len(valid_results)} 个结果")
                        with open(zip_path, "rb") as f:
                            st.download_button(
                                "下载批量结果",
                                f,
                                file_name=os.path.basename(zip_path),
                                use_container_width=True
                            )
        else:
            if model and uploaded_files and valid_count > 0:
                st.info("点击『开始批量检测』按钮开始处理")
            elif not model:
                st.info("请先在左侧加载模型")
            elif not uploaded_files:
                st.info("请上传图片后开始检测")
            else:
                st.warning("没有有效的图片可检测")


if __name__ == "__main__":
    main()
