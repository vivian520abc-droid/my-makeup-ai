import cv2
import mediapipe as mp

def analyze_face_proportions(image):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return "未偵測到臉部"

        landmarks = results.multi_face_landmarks[0].landmark
        
        # 定義關鍵點座標 (y座標用於垂直比例)
        forehead = landmarks[10].y      # 髮際線頂端
        eyebrow_mid = landmarks[168].y  # 眉心
        nose_tip = landmarks[1].y       # 鼻尖
        chin = landmarks[152].y         # 下巴底端
        
        # 1. 計算三庭 (Three Proportions)
        upper_face = eyebrow_mid - forehead
        mid_face = nose_tip - eyebrow_mid
        lower_face = chin - nose_tip
        total = upper_face + mid_face + lower_face
        
        # 轉化為比例 (以中庭為基準 1)
        ratio = (round(upper_face/mid_face, 2), 1.0, round(lower_face/mid_face, 2))
        
        # 2. 計算眼距 (Horizontal - Five Eyes)
        left_eye_outer = landmarks[33].x
        left_eye_inner = landmarks[133].x
        right_eye_inner = landmarks[362].x
        eye_width = left_eye_inner - left_eye_outer
        eye_distance = right_eye_inner - left_eye_inner
        
        return {
            "three_proportions": ratio,
            "eye_spacing_ratio": round(eye_distance / eye_width, 2),
            "advice": "中庭較長，建議使用橫向腮紅" if ratio[1] > 1.05 else "比例均勻，適合多種妝容"
        }

# 這段邏輯就是 App 的「大腦」
