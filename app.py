import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 1. 頁面配置必須在最頂端
st.set_page_config(page_title="MirrorAI", layout="wide")

# 2. 修復版導入邏輯
try:
    import mediapipe as mp
    # 直接深入 mediapipe 的內部路徑，不經過 mp.solutions
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
    MP_AVAILABLE = True
except Exception as e:
    st.error(f"AI 引擎啟動失敗，請聯繫開發者。錯誤代碼: {e}")
    MP_AVAILABLE = False

# 3. 介面設計
st.title("🪞 MirrorAI 鏡萃：AI 臉部分析")
st.markdown("---")

if MP_AVAILABLE:
    uploaded_file = st.sidebar.file_uploader("上傳你的正面照", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # 4. 啟動 AI 掃描
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh_engine:
            
            # 轉換顏色給 OpenCV 使用
            results = face_mesh_engine.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            
            if results.multi_face_landmarks:
                st.success("✅ 臉部特徵掃描成功！")
                
                # 繪製掃描網格
                annotated_img = img_array.copy()
                mp_drawing.draw_landmarks(
                    image=annotated_img,
                    landmark_list=results.multi_face_landmarks[0],
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                
                # 顯示結果
                st.image(annotated_img, caption="AI 分析中...", use_container_width=True)
                st.info("💡 提示：你的臉型輪廓精緻，建議加強腮紅暈染提升氣色。")
            else:
                st.warning("⚠️ 沒看到臉喔！請確保照片光線充足，且沒有戴口罩。")
    else:
        st.info("👈 請先從左側邊欄上傳一張照片。")
