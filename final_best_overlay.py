import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# ==========================
# 1️⃣ SETUP & MODEL LOADING
# ==========================
# TFLite Model load karein
TFLITE_PATH = "iris_pure_float32.tflite"
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Lens Texture load karein (Make sure path is correct)
LENS_PATH = "images/2.png" 
lens_img = cv2.imread(LENS_PATH, cv2.IMREAD_UNCHANGED)

# Mediapipe Face Mesh Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
)

# ==========================
# 2️⃣ HELPER FUNCTIONS
# ==========================

def predict_mask(crop):
    """UNet model se iris ka mask nikalne ke liye"""
    img = cv2.resize(crop, (384, 384)).astype(np.float32) / 255.0
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img, axis=0))
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0]
    mask = (np.squeeze(pred) > 0.3).astype(np.uint8) * 255
    return cv2.resize(mask, (crop.shape[1], crop.shape[0]))

# def apply_hybrid_lens(frame, landmarks, lens_texture):
#     h, w = frame.shape[:2]
    
#     # 1. Landmarks for Eyelid Shapes (Palkon ki shape nikalne ke liye)
#     # In indices se hum aankh ki asli geometry banayenge
#     LEFT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
#     RIGHT_EYE_POINTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

#     eye_configs = [
#         (468, 159, 145, 469, LEFT_EYE_POINTS), # Left
#         (473, 386, 374, 474, RIGHT_EYE_POINTS) # Right
#     ]

#     for iris_idx, top_idx, bot_idx, edge_idx, eye_pts in eye_configs:
#         try:
#             cx = int(float(landmarks[iris_idx].x) * w)
#             cy = int(float(landmarks[iris_idx].y) * h)
            
#             # Eye Openness Check
#             t_y = float(landmarks[top_idx].y) * h
#             b_y = float(landmarks[bot_idx].y) * h
#             if abs(t_y - b_y) < (h * 0.012): continue 

#             # Radius & Safe Crop
#             ex = int(float(landmarks[edge_idx].x) * w)
#             ey = int(float(landmarks[edge_idx].y) * h)
#             r = int(np.sqrt((cx - ex)**2 + (cy - ey)**2) * 1.2)
#             y1, y2, x1, x2 = max(0, cy-r), min(h, cy+r), max(0, cx-r), min(w, cx+r)
#             crop = frame[y1:y2, x1:x2].copy()
#             if crop.size == 0: continue

#             # 2. 🔥 EYELID MASK (Occlusion Logic)
#             # Aankh ki poori boundary ka ek mask banayein
#             eye_poly = np.array([[(float(landmarks[p].x)*w - x1), (float(landmarks[p].y)*h - y1)] for p in eye_pts], dtype=np.int32)
#             occlusion_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
#             cv2.fillPoly(occlusion_mask, [eye_poly], 255)

#             # 3. Combine Masks
#             model_mask = predict_mask(crop) 
#             ch, cw = crop.shape[:2]
#             geo_mask = np.zeros((ch, cw), dtype=np.uint8)
#             cv2.circle(geo_mask, (cw//2, ch//2), int(r * 0.9), 255, -1)

#             # Final mask = (Model + Geo) AND Occlusion
#             # Isse lens sirf palkon ke andar rahega
#             combined_mask = cv2.bitwise_or(geo_mask, model_mask)
#             final_mask = cv2.bitwise_and(combined_mask, occlusion_mask)
#             final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)

#             # 4. Lens Overlay
#             lens_res = cv2.resize(lens_texture, (x2-x1, y2-y1), interpolation=cv2.INTER_LANCZOS4)
            
#             if lens_res.shape[2] == 4:
#                 alpha_map = (final_mask.astype(float) / 255.0) * (lens_res[:,:,3] / 255.0)
#                 alpha_3d = cv2.merge([alpha_map, alpha_map, alpha_map])
                
#                 result = (lens_res[:, :, :3].astype(float) * alpha_3d) + (crop.astype(float) * (1.0 - alpha_3d))
#                 frame[y1:y2, x1:x2] = np.clip(result, 0, 255).astype(np.uint8)

#         except Exception as e:
#             continue
                
#     return frame
def apply_hybrid_lens(frame, landmarks, lens_texture):
    h, w = frame.shape[:2]
    
    # Natural Eyelid Geometry
    LEFT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_POINTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    eye_configs = [
        (468, 159, 145, 469, LEFT_EYE_POINTS), 
        (473, 386, 374, 474, RIGHT_EYE_POINTS) 
    ]

    for iris_idx, top_idx, bot_idx, edge_idx, eye_pts in eye_configs:
        try:
            cx = int(float(landmarks[iris_idx].x) * w)
            cy = int(float(landmarks[iris_idx].y) * h)
            
            # 1. Blinking & Occlusion Check
            t_y = float(landmarks[top_idx].y) * h
            b_y = float(landmarks[bot_idx].y) * h
            if abs(t_y - b_y) < (h * 0.012): continue 

            # 2. Precise Sizing
            ex = int(float(landmarks[edge_idx].x) * w)
            ey = int(float(landmarks[edge_idx].y) * h)
            r = int(np.sqrt((cx - ex)**2 + (cy - ey)**2) * 1.25)
            
            y1, y2, x1, x2 = max(0, cy-r), min(h, cy+r), max(0, cx-r), min(w, cx+r)
            crop = frame[y1:y2, x1:x2].copy()
            if crop.size == 0: continue
            ch, cw = crop.shape[:2]

            # 3. Eyelid Mask (Lens ko palkon ke peeche le jane ke liye)
            eye_poly = np.array([[(float(landmarks[p].x)*w - x1), (float(landmarks[p].y)*h - y1)] for p in eye_pts], dtype=np.int32)
            occlusion_mask = np.zeros((ch, cw), dtype=np.uint8)
            cv2.fillPoly(occlusion_mask, [eye_poly], 255)

            # 4. Hybrid Segmentation Mask
            model_mask = predict_mask(crop) 
            geo_mask = np.zeros((ch, cw), dtype=np.uint8)
            cv2.circle(geo_mask, (cw//2, ch//2), int(r * 0.95), 255, -1)
            final_mask = cv2.bitwise_and(cv2.bitwise_or(geo_mask, model_mask), occlusion_mask)
            final_mask = cv2.GaussianBlur(final_mask, (7, 7), 0)

            # 5. Texture Pre-Processing (Handling Halftone Dots)
            lens_res = cv2.resize(lens_texture, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
            # Dots ko smooth karne ke liye halka sa blur
            lens_res = cv2.GaussianBlur(lens_res, (3, 3), 0) 

            if lens_res.shape[2] == 4:
                # A. Highlights Preservation (Real-world reflection)
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                _, highlights = cv2.threshold(gray_crop, 210, 255, cv2.THRESH_BINARY)
                highlights_3d = cv2.merge([highlights, highlights, highlights]).astype(float) / 255.0

                # B. Ambient Depth Shadow
                shadow_map = np.ones((ch, cw), dtype=np.float32)
                shadow_map[:int(ch*0.5), :] = 0.82 # Top shadow for realism
                shadow_3d = cv2.merge([shadow_map, shadow_map, shadow_map])

                # C. Advanced Blending
                lens_bgr = lens_res[:, :, :3].astype(float)
                # Saturation and Contrast boost for vibrant eyes
                lens_bgr = cv2.convertScaleAbs(lens_bgr, alpha=1.2, beta=-10)
                
                # Using Texture's own Alpha + our Geometric Mask
                alpha_tex = (lens_res[:, :, 3].astype(float) / 255.0)
                alpha_final = alpha_tex * (final_mask.astype(float) / 255.0)
                alpha_3d = cv2.merge([alpha_final, alpha_final, alpha_final])
                
                # Blend: Result = (Lens * Shadow * Alpha) + (Eye * (1-Alpha))
                fg = (lens_bgr * shadow_3d) * alpha_3d
                bg = crop.astype(float) * (1.0 - alpha_3d)
                blended = cv2.add(fg, bg)
                
                # D. Re-Overlaying Original Highlights (The "TTDEye" Sparkle)
                final_result = cv2.addWeighted(blended, 1.0, crop.astype(float) * highlights_3d, 0.45, 0)
                
                frame[y1:y2, x1:x2] = np.clip(final_result, 0, 255).astype(np.uint8)

        except Exception as e:
            continue
                
    return frame
# ==========================
# 3️⃣ MAIN LOOP
# ==========================
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1) # Mirror effect
    
    # Process Frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        # Hybrid function call karein
        frame = apply_hybrid_lens(frame, landmarks, lens_img)

    cv2.imshow("TTDEye Hybrid Precise Filter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()