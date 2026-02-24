import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 1. 頁面配置必須在最前面
st.set_page_config(page_title="MirrorAI", layout="wide")

# 2. 強勢導入 MediaPipe 核心組件
try:
    import mediapipe as mp
    # 直接從核心路徑導入，避開 solutions 屬性報錯
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
    MP_AVAILABLE = True
except Exception as e:
    st.error(f"AI 模組初始化失敗: {e}")
    MP_AVAILABLE = False

st.title("🪞 MirrorAI 鏡萃")

if MP_AVAILABLE:
    uploaded_file = st.sidebar.file_uploader("上傳正面照片", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # 初始化臉部網格
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True
        ) as face_mesh_engine:
            results = face_mesh_engine.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            
            if results.multi_face_landmarks:
                st.success("✅ 臉部掃描完成！")
                annotated_img = img_array.copy()
                mp_drawing.draw_landmarks(
                    image=annotated_img,
                    landmark_list=results.multi_face_landmarks[0],
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                st.image(annotated_img, use_container_width=True)
            else:
                st.warning("無法偵測到臉部，請確保臉部無遮擋。")
    else:
        st.info("請在側邊欄上傳照片開始。")
