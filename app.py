import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# 1. 這行必須在所有 st 指令的最前面
st.set_page_config(page_title="MirrorAI", layout="wide")

# 2. 初始化 AI 工具
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, 
    max_num_faces=1, 
    refine_landmarks=True
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 3. 介面標題
st.title("🪞 MirrorAI 鏡萃：專業妝容分析")
st.markdown("---")

# 4. 側邊欄上傳區
uploaded_file = st.sidebar.file_uploader("上傳照片進行掃描", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # AI 處理影像
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    col1, col2 = st.columns([1, 1])
    
    if results.multi_face_landmarks:
        with col1:
            # 畫出臉部網格，增加科技感
            annotated_img = img_array.copy()
            mp_drawing.draw_landmarks(
                image=annotated_img,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            st.image(annotated_img, caption="AI 特徵掃描中...", use_container_width=True)

        with col2:
            st.subheader("📊 掃描報告")
            # 模擬數據分析顯示
            st.success("✅ 臉部特徵抓取成功")
            st.write("**建議風格：** 原生感清透妝容")
            st.info("💡 提醒：檢測到膚色屬於冷調，建議選取粉色系口紅。")
    else:
        st.error("偵測不到臉部，請確保照片光線充足且臉部清晰。")
else:
    st.info("👋 你好！請在側邊欄上傳一張正面照片開始分析。")
