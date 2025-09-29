import streamlit as st
import os
from datetime import datetime

# 设置页面配置
st.set_page_config(
    page_title="YOLOv11 文本检测系统",
    page_icon="📄",
    layout="wide"
)

# 确保必要目录存在
os.makedirs("custom_models", exist_ok=True)
os.makedirs("detection_results", exist_ok=True)
os.makedirs("detection_logs", exist_ok=True)

# 全局配置存储
if "config" not in st.session_state:
    st.session_state.config = {
        "default_model": "yolov11s.pt",
        "confidence_threshold": 0.5,
        "show_scores": True,
        "show_analysis": True,
        "save_results": True
    }

# 主页面内容
st.title("📄 YOLOv11 文本检测系统")
st.markdown("一个基于改进YOLOv11的文本检测可视化工具，支持图片、视频和摄像头实时检测")

# 功能介绍
st.subheader("系统功能")
col1, col2, col3 = st.columns(3)

with col1:
    st.info("🖼️ **图片文本检测**")
    st.markdown("""
    - 支持多种图片格式（JPG、PNG、BMP等）
    - 精准识别图像中的文本区域
    - 提供详细的文本位置和置信度信息
    """)

with col2:
    st.info("🎥 **视频文本检测**")
    st.markdown("""
    - 处理各类视频文件（MP4、MOV、AVI等）
    - 逐帧检测并标记文本内容
    - 生成带检测结果的输出视频
    """)

with col3:
    st.info("📹 **摄像头实时检测**")
    st.markdown("""
    - 调用设备摄像头进行实时检测
    - 实时标记画面中的文本区域
    - 支持检测结果录制与保存
    """)

# 使用指南
st.subheader("使用指南")
st.markdown("""
1. 在左侧栏选择需要使用的功能页面（图片检测/视频检测/摄像头检测）
2. 如需使用自定义模型，先在「模型管理」页面上传你的YOLOv11模型
3. 在对应功能页面设置检测参数
4. 上传文件或启动摄像头开始检测
5. 查看检测结果并可选择保存或下载

系统会自动保存所有检测结果到本地目录，便于后续查看和分析。
""")

# 页脚信息
st.markdown("---")
st.markdown(f"© {datetime.now().year} YOLOv11文本检测系统 | 版本 1.0.0")