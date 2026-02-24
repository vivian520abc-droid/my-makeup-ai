import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- 第一段：頁面配置 (必須在最前面) ---
st.set_page_config(page_title="MirrorAI", layout="wide")

# --- 第二段：呼叫法 (核心 AI 引擎) ---
try:
    import mediapipe as mp
    # 這就是所謂的呼叫法：直接指定 mp 裡面的工具
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    MP_AVAILABLE = True
except Exception as e:
    st.error(f"引擎啟動失敗，錯誤訊息: {e}")
    MP_AVAILABLE = False

# --- 第三段：網頁內容 ---
st.title("🪞 MirrorAI 鏡萃：AI 臉部分析")

if MP_AVAILABLE:
    # 這邊放入你原本的功能代碼
    uploaded_file = st.sidebar.file_uploader("上傳照片", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # 啟動臉部網格掃描
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True
        ) as face_mesh_engine:
            
            # 轉換顏色並處理
            results = face_mesh_engine.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            
            if results.multi_face_landmarks:
                st.success("✅ 臉部特徵抓取成功！")
                # 這裡可以繼續寫繪製或分析的邏輯
                st.image(image, caption="原始照片", use_container_width=True)
            else:
                st.warning("偵測不到臉部，請換一張清晰的照片。")
    else:
        st.info("請在左側選單上傳照片開始。")
