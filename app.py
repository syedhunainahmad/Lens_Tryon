# import cv2
# import numpy as np
# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
# import tflite_runtime.interpreter as tflite
# from av import VideoFrame
# import mediapipe as mp
# from mediapipe.python.solutions import face_mesh as mp_face_mesh

# # --- 1. Resources Loading ---
# @st.cache_resource
# def load_assets():
#     # TFLite Interpreter (No TensorFlow for stability)
#     interpreter = tflite.Interpreter(model_path="iris_pure_float32.tflite")
#     interpreter.allocate_tensors()
    
#     # Lens texture loading
#     lens_img = cv2.imread("images/1.png", cv2.IMREAD_UNCHANGED)
#     return interpreter, lens_img

# # Initialize interpreter and assets
# interpreter, lens_img = load_assets()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # --- 2. Mediapipe Face Mesh Setup ---
# # Direct class call karein taaki attribute error na aaye
# face_mesh_tool = mp_face_mesh.FaceMesh(
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # --- 2. Model Prediction Logic ---
# def predict_mask_with_model(crop):
#     # UNet input processing
#     img = cv2.resize(crop, (384, 384)).astype(np.float32) / 255.0
#     interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img, axis=0))
#     interpreter.invoke()
#     pred = interpreter.get_tensor(output_details[0]['index'])[0]
    
#     # Mask creation
#     mask = (np.squeeze(pred) > 0.3).astype(np.uint8) * 255
#     return cv2.resize(mask, (crop.shape[1], crop.shape[0]))

# # --- 3. Lens Application (Hybrid: Model + Landmarks) ---
# def apply_hybrid_lens(frame, landmarks, lens_texture):
#     h, w = frame.shape[:2]
    
#     # Aapka local code wala geometry
#     LEFT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
#     RIGHT_EYE_POINTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

#     eye_configs = [
#         (468, 159, 145, 469, LEFT_EYE_POINTS), 
#         (473, 386, 374, 474, RIGHT_EYE_POINTS) 
#     ]

#     for iris_idx, top_idx, bot_idx, edge_idx, eye_pts in eye_configs:
#         try:
#             cx = int(landmarks[iris_idx].x * w)
#             cy = int(landmarks[iris_idx].y * h)
            
#             # Precise Sizing (Aapki local logic)
#             ex = int(landmarks[edge_idx].x * w)
#             ey = int(landmarks[edge_idx].y * h)
#             r = int(np.sqrt((cx - ex)**2 + (cy - ey)**2) * 1.25)
            
#             y1, y2, x1, x2 = max(0, cy-r), min(h, cy+r), max(0, cx-r), min(w, cx+r)
#             crop = frame[y1:y2, x1:x2].copy()
#             if crop.size == 0: continue
#             ch, cw = crop.shape[:2]

#             # 1. UNet Model Mask (Aapka trained model)
#             model_mask = predict_mask_with_model(crop) 
            
#             # 2. Geometric Mask (Backup ke liye)
#             geo_mask = np.zeros((ch, cw), dtype=np.uint8)
#             cv2.circle(geo_mask, (cw//2, ch//2), int(r * 0.95), 255, -1)

#             # 3. Eyelid Occlusion (Palkon ke peeche)
#             eye_poly = np.array([[(landmarks[p].x*w - x1), (landmarks[p].y*h - y1)] for p in eye_pts], dtype=np.int32)
#             occlusion_mask = np.zeros((ch, cw), dtype=np.uint8)
#             cv2.fillPoly(occlusion_mask, [eye_poly], 255)

#             # 4. Hybrid Mask Combine
#             # Hum bitwise_or use kar rahe hain taaki agar model fail ho toh geo_mask kaam kare
#             final_mask = cv2.bitwise_and(cv2.bitwise_or(geo_mask, model_mask), occlusion_mask)
#             final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)

#             # 5. Advanced Blending (Aapki local advanced logic)
#             lens_res = cv2.resize(lens_texture, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
#             # if lens_res.shape[2] == 4:
#             #     alpha_tex = (lens_res[:, :, 3].astype(float) / 255.0)
#             #     alpha_final = alpha_tex * (final_mask.astype(float) / 255.0)
#             #     alpha_3d = cv2.merge([alpha_final] * 3)
                
#             #     fg = lens_res[:, :, :3].astype(float) * alpha_3d
#             #     bg = crop.astype(float) * (1.0 - alpha_3d)
                
#             #     # Blend aur Frame update
#             #     frame[y1:y2, x1:x2] = cv2.add(fg, bg).astype(np.uint8)
#             if lens_res.shape[2] == 4:
#                 # 3. Alpha calculation ko thora sharp banayein
#                 # Hum 0.3 ki jagah 0.5 threshold use karenge taaki mask solid rahe
#                 alpha_mask = (final_mask.astype(float) / 255.0)
                
#                 # Texture ki details bachane ke liye mask ko thora "punchy" banayein
#                 alpha_mask = np.where(alpha_mask > 0.2, alpha_mask, 0) 

#                 alpha_tex = (lens_res[:, :, 3].astype(float) / 255.0)
#                 alpha_final = alpha_tex * alpha_mask
#                 alpha_3d = cv2.merge([alpha_final] * 3)
                
#                 fg = lens_res[:, :, :3].astype(float) * alpha_3d
#                 bg = crop.astype(float) * (1.0 - alpha_3d)
                
#                 frame[y1:y2, x1:x2] = cv2.add(fg, bg).astype(np.uint8)

#         except Exception as e:
#             continue
#     return frame

# class VideoProcessor(VideoTransformerBase):
#     # 'transform' ki jagah 'recv' likhein
#     def recv(self, frame):
#         # Frame ko ndarray mein badlein
#         img = frame.to_ndarray(format="bgr24")
#         img = cv2.flip(img, 1)
        
#         h_orig, w_orig = img.shape[:2]
#         img_proc = cv2.resize(img, (640, 480))
        
#         rgb = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)
#         results = face_mesh_tool.process(rgb)
        
#         if results.multi_face_landmarks:
#             img_proc = apply_hybrid_lens(img_proc, results.multi_face_landmarks[0].landmark, lens_img)
            
#         img_final = cv2.resize(img_proc, (w_orig, h_orig))

#         # Naye tareeke mein VideoFrame object wapas bhejna hota hai
#         return VideoFrame.from_ndarray(img_final, format="bgr24")

# RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
# # --- Streamer Settings (Mobile Optimized) ---
# webrtc_streamer(
#     key="ttdeye-v3",
#     video_processor_factory=VideoProcessor,
#     rtc_configuration=RTC_CONFIG,
#     # Async processing on karne se UI freeze nahi hoti
#     async_processing=True,
#     media_stream_constraints={
#         "video": {
#             "width": {"ideal": 640},
#             "frameRate": {"ideal": 20}
#         },
#         "audio": False
#     }
# )
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow import lite as tflite
# import tflite_runtime.interpreter as tflite
from av import VideoFrame
import mediapipe as mp
from mediapipe.python.solutions import face_mesh as mp_face_mesh

# --- 1. Resources Loading ---
@st.cache_resource
def load_assets():
    interpreter = tflite.Interpreter(model_path="iris_pure_float32.tflite")
    interpreter.allocate_tensors()
    lens_img = cv2.imread("images/1.png", cv2.IMREAD_UNCHANGED)
    return interpreter, lens_img

interpreter, lens_img = load_assets()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

face_mesh_tool = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def predict_mask_with_model(crop):
    img = cv2.resize(crop, (384, 384)).astype(np.float32) / 255.0
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img, axis=0))
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0]
    mask = (np.squeeze(pred) > 0.3).astype(np.uint8) * 255
    return cv2.resize(mask, (crop.shape[1], crop.shape[0]))

# --- 3. Fixed Hybrid Logic ---
def apply_hybrid_lens(frame, landmarks, lens_texture, cached_masks):
    h, w = frame.shape[:2]
    
    # Eyelid boundary points (Important for clipping)
    LEFT_EYE_PTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_PTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    eye_configs = [
        (468, 471, LEFT_EYE_PTS, 0), 
        (473, 476, RIGHT_EYE_PTS, 1) 
    ]

    new_masks = []
    for iris_idx, edge_idx, eye_pts, eye_id in eye_configs:
        try:
            cx, cy = int(landmarks[iris_idx].x * w), int(landmarks[iris_idx].y * h)
            ex, ey = int(landmarks[edge_idx].x * w), int(landmarks[edge_idx].y * h)
            r = int(np.sqrt((cx - ex)**2 + (cy - ey)**2) * 1.3)
            
            y1, y2, x1, x2 = max(0, cy-r), min(h, cy+r), max(0, cx-r), min(w, cx+r)
            crop = frame[y1:y2, x1:x2].copy()
            if crop.size == 0: continue
            ch, cw = crop.shape[:2]

            # 1. UNet Mask (Current or Cached)
            if cached_masks is None:
                unet_m = predict_mask_with_model(crop)
            else:
                unet_m = cv2.resize(cached_masks[eye_id], (cw, ch))
            new_masks.append(unet_m)

            # 2. Precise Eyelid Clipping (Isi se lens eyelid ke peeche jayega)
            eyelid_mask = np.zeros((ch, cw), dtype=np.uint8)
            # Coordinates ko crop ke relative convert karna zaroori hai
            eye_poly = np.array([[(int(landmarks[p].x * w) - x1), (int(landmarks[p].y * h) - y1)] for p in eye_pts], dtype=np.int32)
            cv2.fillPoly(eyelid_mask, [eye_poly], 255)

            # 3. Combine Masks
            final_mask = cv2.bitwise_and(unet_m, eyelid_mask)
            final_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)

            # 4. Blending
            lens_res = cv2.resize(lens_texture, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
            if lens_res.shape[2] == 4:
                alpha = (lens_res[:, :, 3].astype(float) / 255.0) * (final_mask.astype(float) / 255.0)
                alpha_3d = cv2.merge([alpha] * 3)
                
                fg = lens_res[:, :, :3].astype(float) * alpha_3d
                bg = crop.astype(float) * (1.0 - alpha_3d)
                frame[y1:y2, x1:x2] = cv2.add(fg, bg).astype(np.uint8)

        except Exception: continue
    return frame, new_masks

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.cached_masks = None

    def recv(self, frame):
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h_orig, w_orig = img.shape[:2]

        img_proc = cv2.resize(img, (640, 480))
        results = face_mesh_tool.process(cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB))
        
        if results.multi_face_landmarks:
            # Skip frame for UNet to reduce lag
            if self.frame_count % 2 == 0:
                img_proc, self.cached_masks = apply_hybrid_lens(img_proc, results.multi_face_landmarks[0].landmark, lens_img, None)
            else:
                img_proc, _ = apply_hybrid_lens(img_proc, landmarks=results.multi_face_landmarks[0].landmark, lens_texture=lens_img, cached_masks=self.cached_masks)
            
        return VideoFrame.from_ndarray(cv2.resize(img_proc, (w_orig, h_orig)), format="bgr24")

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
webrtc_streamer(key="ttdeye-final", video_processor_factory=VideoProcessor, rtc_configuration=RTC_CONFIG, async_processing=True)

# import cv2
# import numpy as np
# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
# import tensorflow as tf
# from av import VideoFrame
# import mediapipe as mp
# from mediapipe.python.solutions import face_mesh as mp_face_mesh

# # --- 1. Resources Loading ---
# @st.cache_resource
# def load_assets():
#     # Model ko pehle bytes mein read karein (Fixes Mmap error)
#     try:
#         with open("iris_pure_float32.tflite", "rb") as f:
#             model_content = f.read()
        
#         # Interpreter ko model_content (bytes) se load karein
#         interpreter = tf.lite.Interpreter(model_content=model_content)
#         interpreter.allocate_tensors()
#     except Exception as e:
#         st.error(f"Model Load Error: {e}")
#         return None, None

#     # Lens texture loading
#     lens_img = cv2.imread("images/1.png", cv2.IMREAD_UNCHANGED)
#     if lens_img is None:
#         st.error("Lens image not found! Check images/1.png path.")
        
#     return interpreter, lens_img

# interpreter, lens_img = load_assets()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# face_mesh_tool = mp_face_mesh.FaceMesh(
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# def predict_mask_with_model(crop):
#     # Performance Hint: UNet input size fixed at 384
#     img = cv2.resize(crop, (384, 384)).astype(np.float32) / 255.0
#     interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img, axis=0))
#     interpreter.invoke()
#     pred = interpreter.get_tensor(output_details[0]['index'])[0]
#     mask = (np.squeeze(pred) > 0.3).astype(np.uint8) * 255
#     return cv2.resize(mask, (crop.shape[1], crop.shape[0]))

# def apply_hybrid_lens(frame, landmarks, lens_texture):
#     h, w = frame.shape[:2]
#     LEFT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
#     RIGHT_EYE_POINTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

#     eye_configs = [
#         (468, 159, 145, 469, LEFT_EYE_POINTS), 
#         (473, 386, 374, 474, RIGHT_EYE_POINTS) 
#     ]

#     for iris_idx, top_idx, bot_idx, edge_idx, eye_pts in eye_configs:
#         try:
#             cx, cy = int(landmarks[iris_idx].x * w), int(landmarks[iris_idx].y * h)
#             ex, ey = int(landmarks[edge_idx].x * w), int(landmarks[edge_idx].y * h)
#             r = int(np.sqrt((cx - ex)**2 + (cy - ey)**2) * 1.25)
            
#             y1, y2, x1, x2 = max(0, cy-r), min(h, cy+r), max(0, cx-r), min(w, cx+r)
#             crop = frame[y1:y2, x1:x2].copy()
#             if crop.size == 0: continue
#             ch, cw = crop.shape[:2]

#             model_mask = predict_mask_with_model(crop) 
#             geo_mask = np.zeros((ch, cw), dtype=np.uint8)
#             cv2.circle(geo_mask, (cw//2, ch//2), int(r * 0.95), 255, -1)

#             eye_poly = np.array([[(landmarks[p].x*w - x1), (landmarks[p].y*h - y1)] for p in eye_pts], dtype=np.int32)
#             occlusion_mask = np.zeros((ch, cw), dtype=np.uint8)
#             cv2.fillPoly(occlusion_mask, [eye_poly], 255)

#             final_mask = cv2.bitwise_and(cv2.bitwise_or(geo_mask, model_mask), occlusion_mask)
#             final_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)

#             lens_res = cv2.resize(lens_texture, (cw, ch), interpolation=cv2.INTER_AREA)
#             if lens_res.shape[2] == 4:
#                 alpha = (lens_res[:, :, 3].astype(float) / 255.0) * (final_mask.astype(float) / 255.0)
#                 alpha_3d = cv2.merge([alpha] * 3)
#                 fg = lens_res[:, :, :3].astype(float) * alpha_3d
#                 bg = crop.astype(float) * (1.0 - alpha_3d)
#                 frame[y1:y2, x1:x2] = cv2.add(fg, bg).astype(np.uint8)
#         except: continue
#     return frame

# class VideoProcessor(VideoTransformerBase):
#     def __init__(self):
#         self.frame_count = 0
#         self.last_results = None # Purane landmarks save karne ke liye

#     def recv(self, frame):
#         self.frame_count += 1
#         img = frame.to_ndarray(format="bgr24")
#         img = cv2.flip(img, 1)
#         h_orig, w_orig = img.shape[:2]

#         # Resolution ko process ke liye 480p rakhein (Balance)
#         img_proc = cv2.resize(img, (640, 480))
#         rgb = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)
        
#         # 1. Face Mesh har frame par chalayein taaki tracking na tute
#         results = face_mesh_tool.process(rgb)
        
#         if results.multi_face_landmarks:
#             # 2. Lens sirf har alternative frame par calculate karein (Lag fix)
#             # Lekin display har frame par hoga
#             landmarks = results.multi_face_landmarks[0].landmark
#             img_proc = apply_hybrid_lens(img_proc, landmarks, lens_img)
            
#         # Output wapas original size par
#         img_final = cv2.resize(img_proc, (w_orig, h_orig))
#         return VideoFrame.from_ndarray(img_final, format="bgr24")

# RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# webrtc_streamer(
#     key="ttdeye-v4-fast",
#     video_processor_factory=VideoProcessor,
#     rtc_configuration=RTC_CONFIG,
#     async_processing=True, # UI freeze hone se bachata hai
#     media_stream_constraints={
#         "video": {
#             "width": {"ideal": 640},
#             "frameRate": {"ideal": 20} # Stable FPS
#         },
#         "audio": False
#     }
# )


# import cv2
# import numpy as np
# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
# import tensorflow as tf
# from av import VideoFrame
# import mediapipe as mp
# from mediapipe.python.solutions import face_mesh as mp_face_mesh

# # --- 1. Resources Loading ---
# @st.cache_resource
# def load_assets():
#     interpreter = tf.lite.Interpreter(model_path="iris_pure_float32.tflite")
#     interpreter.allocate_tensors()
#     lens_img = cv2.imread("images/1.png", cv2.IMREAD_UNCHANGED)
#     return interpreter, lens_img

# interpreter, lens_img = load_assets()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# face_mesh_tool = mp_face_mesh.FaceMesh(
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# def predict_mask_with_model(crop):
#     img = cv2.resize(crop, (384, 384)).astype(np.float32) / 255.0
#     interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img, axis=0))
#     interpreter.invoke()
#     pred = interpreter.get_tensor(output_details[0]['index'])[0]
#     mask = (np.squeeze(pred) > 0.3).astype(np.uint8) * 255
#     return cv2.resize(mask, (crop.shape[1], crop.shape[0]))

# # --- 3. Lens Application (1st Logic: Professional Blending) ---
# def apply_hybrid_lens(frame, landmarks, lens_texture):
#     h, w = frame.shape[:2]
#     LEFT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
#     RIGHT_EYE_POINTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

#     eye_configs = [
#         (468, 159, 145, 469, LEFT_EYE_POINTS), 
#         (473, 386, 374, 474, RIGHT_EYE_POINTS) 
#     ]

#     for iris_idx, top_idx, bot_idx, edge_idx, eye_pts in eye_configs:
#         try:
#             cx, cy = int(landmarks[iris_idx].x * w), int(landmarks[iris_idx].y * h)
#             ex, ey = int(landmarks[edge_idx].x * w), int(landmarks[edge_idx].y * h)
#             r = int(np.sqrt((cx - ex)**2 + (cy - ey)**2) * 1.25)
            
#             y1, y2, x1, x2 = max(0, cy-r), min(h, cy+r), max(0, cx-r), min(w, cx+r)
#             crop = frame[y1:y2, x1:x2].copy()
#             if crop.size == 0: continue
#             ch, cw = crop.shape[:2]

#             # 1. Model & Geo Mask
#             model_mask = predict_mask_with_model(crop) 
#             geo_mask = np.zeros((ch, cw), dtype=np.uint8)
#             cv2.circle(geo_mask, (cw//2, ch//2), int(r * 0.95), 255, -1)

#             # 2. Eyelid Occlusion
#             eye_poly = np.array([[(landmarks[p].x*w - x1), (landmarks[p].y*h - y1)] for p in eye_pts], dtype=np.int32)
#             occlusion_mask = np.zeros((ch, cw), dtype=np.uint8)
#             cv2.fillPoly(occlusion_mask, [eye_poly], 255)

#             # 3. Hybrid Mask with Soft Edges (Gaussian Blur 7x7 for natural look)
#             final_mask = cv2.bitwise_and(cv2.bitwise_or(geo_mask, model_mask), occlusion_mask)
#             final_mask = cv2.GaussianBlur(final_mask, (7, 7), 0)

#             # 4. High Quality Resizing (LANCZOS4 preserves lens details)
#             lens_res = cv2.resize(lens_texture, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
            
#             if lens_res.shape[2] == 4:
#                 # 5. Advanced Blending Logic
#                 # Lens ki apni transparency + Mask ki transparency
#                 alpha_tex = (lens_res[:, :, 3].astype(float) / 255.0)
#                 alpha_mask = (final_mask.astype(float) / 255.0)
#                 alpha_final = alpha_tex * alpha_mask
                
#                 alpha_3d = cv2.merge([alpha_final] * 3)
                
#                 # Colors separation
#                 lens_bgr = lens_res[:, :, :3].astype(float)
                
#                 # Composition: (Lens * Alpha) + (Original Eye * (1 - Alpha))
#                 fg = lens_bgr * alpha_3d
#                 bg = crop.astype(float) * (1.0 - alpha_3d)
                
#                 frame[y1:y2, x1:x2] = cv2.add(fg, bg).astype(np.uint8)

#         except Exception:
#             continue
#     return frame

# class VideoProcessor(VideoTransformerBase):
#     def __init__(self):
#         self.frame_count = 0

#     def recv(self, frame):
#         self.frame_count += 1
#         img = frame.to_ndarray(format="bgr24")
#         img = cv2.flip(img, 1)
#         h_orig, w_orig = img.shape[:2]

#         # Resolution for processing
#         img_proc = cv2.resize(img, (640, 480))
#         rgb = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)
        
#         results = face_mesh_tool.process(rgb)
        
#         if results.multi_face_landmarks:
#             landmarks = results.multi_face_landmarks[0].landmark
#             img_proc = apply_hybrid_lens(img_proc, landmarks, lens_img)
            
#         img_final = cv2.resize(img_proc, (w_orig, h_orig))
#         return VideoFrame.from_ndarray(img_final, format="bgr24")

# RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# webrtc_streamer(
#     key="ttdeye-pro-version",
#     video_processor_factory=VideoProcessor,
#     rtc_configuration=RTC_CONFIG,
#     async_processing=True,
#     media_stream_constraints={
#         "video": {"width": {"ideal": 640}, "frameRate": {"ideal": 20}},
#         "audio": False
#     }
# )



# import cv2
# import numpy as np
# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
# import tensorflow as tf
# from av import VideoFrame
# import mediapipe as mp
# from mediapipe.python.solutions import face_mesh as mp_face_mesh

# # --- 1. Resources Loading ---
# @st.cache_resource
# def load_assets():
#     try:
#         # Bytes reading for Mmap fix on Streamlit Cloud
#         with open("iris_pure_float32.tflite", "rb") as f:
#             model_content = f.read()
#         interpreter = tf.lite.Interpreter(model_content=model_content)
#         interpreter.allocate_tensors()
#     except Exception as e:
#         st.error(f"Error loading TFLite model: {e}")
#         return None, None

#     # Load lens with Alpha Channel
#     lens_img = cv2.imread("images/3.png", cv2.IMREAD_UNCHANGED)
#     if lens_img is None:
#         st.error("Lens image not found in images/1.png")
#     return interpreter, lens_img

# interpreter, lens_img = load_assets()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# face_mesh_tool = mp_face_mesh.FaceMesh(
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# def predict_mask_with_model(crop):
#     # UNet processing (Fixed 384x384)
#     img = cv2.resize(crop, (384, 384)).astype(np.float32) / 255.0
#     interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img, axis=0))
#     interpreter.invoke()
#     pred = interpreter.get_tensor(output_details[0]['index'])[0]
    
#     mask = (np.squeeze(pred) > 0.3).astype(np.uint8) * 255
#     return cv2.resize(mask, (crop.shape[1], crop.shape[0]))

# # def apply_hybrid_lens(frame, landmarks, lens_texture):
# #     h, w = frame.shape[:2]
    
# #     # Eyelid points for precise occlusion
# #     LEFT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
# #     RIGHT_EYE_POINTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# #     # Iris Config: (Center, Top, Bottom, Vertical_Edge, Eyelid_Points)
# #     # 468/473 center hain, 471/476 horizontal edges hain
# #     eye_configs = [
# #         (468, 159, 145, 471, LEFT_EYE_POINTS), 
# #         (473, 386, 374, 476, RIGHT_EYE_POINTS) 
# #     ]

# #     for iris_idx, top_idx, bot_idx, edge_idx, eye_pts in eye_configs:
# #         try:
# #             # 1. Precise Coordinate Calculation
# #             cx, cy = int(landmarks[iris_idx].x * w), int(landmarks[iris_idx].y * h)
# #             ex, ey = int(landmarks[edge_idx].x * w), int(landmarks[edge_idx].y * h)
            
# #             # Radius ko thora barhayein (1.25 ya 1.3 multiplier) taaki iris cover ho
# #             r = int(np.sqrt((cx - ex)**2 + (cy - ey)**2) * 1.3) 
            
# #             y1, y2, x1, x2 = max(0, cy-r), min(h, cy+r), max(0, cx-r), min(w, cx+r)
# #             crop = frame[y1:y2, x1:x2].copy()
# #             if crop.size == 0: continue
# #             ch, cw = crop.shape[:2]

# #             # 2. Refined Masking
# #             # Geometric circle ko center mein align karein
# #             geo_mask = np.zeros((ch, cw), dtype=np.uint8)
# #             cv2.circle(geo_mask, (cw//2, ch//2), int(r * 0.95), 255, -1)

# #             # Eyelid clipping (palkon ke liye)
# #             eye_poly = np.array([[(landmarks[p].x*w - x1), (landmarks[p].y*h - y1)] for p in eye_pts], dtype=np.int32)
# #             occlusion_mask = np.zeros((ch, cw), dtype=np.uint8)
# #             cv2.fillPoly(occlusion_mask, [eye_poly], 255)

# #             # Final Mask: Model mask aur Geometric mask ko combine karein
# #             # Model mask agar accurate nahi toh sirf geo_mask aur occlusion use karein test ke liye
# #             final_mask = cv2.bitwise_and(geo_mask, occlusion_mask)
# #             final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)

# #             # 3. High Quality Texture Blending
# #             # Resize lens using AREA or LANCZOS4 for better texture preservation
# #             lens_res = cv2.resize(lens_texture, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
            
# #             if lens_res.shape[2] == 4:
# #                 # Alpha calculations
# #                 alpha_tex = (lens_res[:, :, 3].astype(float) / 255.0)
# #                 alpha_mask = (final_mask.astype(float) / 255.0)
                
# #                 # Combined alpha for smooth edges
# #                 combined_alpha = cv2.merge([alpha_tex * alpha_mask] * 3)
                
# #                 # Lens image (foreground) aur Eye crop (background) ka blend
# #                 fg = lens_res[:, :, :3].astype(float) * combined_alpha
# #                 bg = crop.astype(float) * (1.0 - combined_alpha)
                
# #                 # Final pixel update
# #                 frame[y1:y2, x1:x2] = cv2.add(fg, bg).astype(np.uint8)

# #         except Exception: continue
# #     return frame

# # def apply_hybrid_lens(frame, landmarks, lens_texture):
# #     h, w = frame.shape[:2]
    
# #     LEFT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
# #     RIGHT_EYE_POINTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# #     # Iris Center and Radius Points
# #     eye_configs = [
# #         (468, 471, LEFT_EYE_POINTS), # Left Eye
# #         (473, 476, RIGHT_EYE_POINTS)  # Right Eye
# #     ]

# #     for iris_idx, edge_idx, eye_pts in eye_configs:
# #         try:
# #             # 1. Coordinate Calculation with high precision
# #             cx, cy = int(landmarks[iris_idx].x * w), int(landmarks[iris_idx].y * h)
# #             ex, ey = int(landmarks[edge_idx].x * w), int(landmarks[edge_idx].y * h)
            
# #             # Radius multiplier ko thora kam kiya (1.2) aur padding rakhi precision ke liye
# #             r = int(np.sqrt((cx - ex)**2 + (cy - ey)**2) * 1.2) 
            
# #             y1, y2, x1, x2 = max(0, cy-r), min(h, cy+r), max(0, cx-r), min(w, cx+r)
# #             crop = frame[y1:y2, x1:x2].copy()
# #             if crop.size == 0: continue
# #             ch, cw = crop.shape[:2]

# #             # 2. Masking with Soft Edges (Anti-Aliasing)
# #             geo_mask = np.zeros((ch, cw), dtype=np.uint8)
# #             # 0.9 multiplier for tighter fit
# #             cv2.circle(geo_mask, (cw//2, ch//2), int(r * 0.9), 255, -1)

# #             eye_poly = np.array([[(landmarks[p].x*w - x1), (landmarks[p].y*h - y1)] for p in eye_pts], dtype=np.int32)
# #             occ_mask = np.zeros((ch, cw), dtype=np.uint8)
# #             cv2.fillPoly(occ_mask, [eye_poly], 255)

# #             final_mask = cv2.bitwise_and(geo_mask, occ_mask)
# #             # Gaussian blur ko 5x5 se 3x3 kiya taaki edges bohot zyada fade na hon
# #             final_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)

# #             # 3. ADVANCED TEXTURE PROCESSING
# #             # Interpolation ko INTER_CUBIC kiya for smoother look on small crops
# #             lens_res = cv2.resize(lens_texture, (cw, ch), interpolation=cv2.INTER_CUBIC)
            
# #             # DOOR SE TEXTURE CLEAR RAKHNE KE LIYE: Sharpening Filter
# #             # Yeh line door se print ko ajeeb hone se bachayegi
# #             kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
# #             lens_res_sharp = cv2.filter2D(lens_res[:,:,:3], -1, kernel)

# #             if lens_res.shape[2] == 4:
# #                 # Alpha channels blending
# #                 alpha_tex = (lens_res[:, :, 3].astype(float) / 255.0)
# #                 alpha_mask = (final_mask.astype(float) / 255.0)
                
# #                 # Combined alpha with Gamma Adjustment (0.8) for deeper blend
# #                 combined_alpha = np.power(alpha_tex * alpha_mask, 0.8)
# #                 combined_alpha_3d = cv2.merge([combined_alpha] * 3)
                
# #                 # Foreground (Lens) and Background (Eye)
# #                 fg = lens_res_sharp.astype(float) * combined_alpha_3d
# #                 bg = crop.astype(float) * (1.0 - combined_alpha_3d)
                
# #                 frame[y1:y2, x1:x2] = cv2.add(fg, bg).astype(np.uint8)

# #         except Exception: continue
# #     return frame
# def apply_hybrid_lens(frame, landmarks, lens_texture):
#     h, w = frame.shape[:2]
    
#     # Eyelid points for precise clipping (palkon ke liye)
#     LEFT_EYE_PTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
#     RIGHT_EYE_PTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

#     eye_configs = [
#         (468, 471, LEFT_EYE_PTS), 
#         (473, 476, RIGHT_EYE_PTS) 
#     ]

#     for iris_idx, edge_idx, eye_pts in eye_configs:
#         try:
#             # 1. Exact Center and Radius
#             cx, cy = int(landmarks[iris_idx].x * w), int(landmarks[iris_idx].y * h)
#             ex, ey = int(landmarks[edge_idx].x * w), int(landmarks[edge_idx].y * h)
#             r = int(np.sqrt((cx - ex)**2 + (cy - ey)**2) * 1.2)
            
#             # ROI (Region of Interest)
#             y1, y2, x1, x2 = max(0, cy-r), min(h, cy+r), max(0, cx-r), min(w, cx+r)
#             crop = frame[y1:y2, x1:x2]
#             if crop.size == 0: continue
#             ch, cw = crop.shape[:2]

#             # 2. FAST MASKING (MediaPipe Eyelid + Iris Circle)
#             # Pehle poora iris ka circle banayein
#             iris_mask = np.zeros((ch, cw), dtype=np.uint8)
#             cv2.circle(iris_mask, (cw//2, ch//2), int(r * 0.9), 255, -1)

#             # Phir palkon (eyelids) ka mask banayein
#             eyelid_mask = np.zeros((ch, cw), dtype=np.uint8)
#             eye_poly = np.array([[(landmarks[p].x*w - x1), (landmarks[p].y*h - y1)] for p in eye_pts], dtype=np.int32)
#             cv2.fillPoly(eyelid_mask, [eye_poly], 255)

#             # Final mask: Sirf wo jagah jahan iris aur khuli aankh dono hain
#             final_mask = cv2.bitwise_and(iris_mask, eyelid_mask)
#             final_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)

#             # 3. FAST BLENDING
#             # Resize lens to match iris crop
#             lens_res = cv2.resize(lens_texture, (cw, ch), interpolation=cv2.INTER_LINEAR)
            
#             if lens_res.shape[2] == 4:
#                 # Mask normalization
#                 mask_alpha = final_mask.astype(float) / 255.0
#                 tex_alpha = (lens_res[:, :, 3].astype(float) / 255.0) * mask_alpha
#                 alpha_3d = cv2.merge([tex_alpha] * 3)

#                 # Overlay logic
#                 fg = lens_res[:, :, :3].astype(float) * alpha_3d
#                 bg = crop.astype(float) * (1.0 - alpha_3d)
                
#                 # Update frame directly for speed
#                 frame[y1:y2, x1:x2] = cv2.add(fg, bg).astype(np.uint8)

#         except Exception as e:
#             continue
#     return frame
# # def apply_hybrid_lens(frame, landmarks, lens_texture):
# #     h, w = frame.shape[:2]
    
# #     LEFT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
# #     RIGHT_EYE_POINTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# #     eye_configs = [
# #         (468, 471, LEFT_EYE_POINTS), 
# #         (473, 476, RIGHT_EYE_POINTS) 
# #     ]

# #     for iris_idx, edge_idx, eye_pts in eye_configs:
# #         try:
# #             cx, cy = int(landmarks[iris_idx].x * w), int(landmarks[iris_idx].y * h)
# #             ex, ey = int(landmarks[edge_idx].x * w), int(landmarks[edge_idx].y * h)
            
# #             # Perfect Scale: 1.25 is ideal for covering natural iris
# #             r = int(np.sqrt((cx - ex)**2 + (cy - ey)**2) * 1.25) 
            
# #             y1, y2, x1, x2 = max(0, cy-r), min(h, cy+r), max(0, cx-r), min(w, cx+r)
# #             crop = frame[y1:y2, x1:x2].copy()
# #             if crop.size == 0: continue
# #             ch, cw = crop.shape[:2]

# #             # 1. SOFT MASKING (Anti-Aliasing)
# #             geo_mask = np.zeros((ch, cw), dtype=np.uint8)
# #             cv2.circle(geo_mask, (cw//2, ch//2), int(r * 0.95), 255, -1)

# #             eye_poly = np.array([[(landmarks[p].x*w - x1), (landmarks[p].y*h - y1)] for p in eye_pts], dtype=np.int32)
# #             occ_mask = np.zeros((ch, cw), dtype=np.uint8)
# #             cv2.fillPoly(occ_mask, [eye_poly], 255)

# #             final_mask = cv2.bitwise_and(geo_mask, occ_mask)
# #             # Higher blur for natural edge merging
# #             final_mask = cv2.GaussianBlur(final_mask, (7, 7), 0)

# #             # 2. HIGH QUALITY TEXTURE PREPARATION
# #             lens_res = cv2.resize(lens_texture, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
            
# #             # Sharpening to maintain detail at distance
# #             kernel = np.array([[0, -0.1, 0], [-0.1, 1.4, -0.1], [0, -0.1, 0]])
# #             lens_rgb = cv2.filter2D(lens_res[:,:,:3], -1, kernel)

# #             # if lens_res.shape[2] == 4:
# #             #     # 3. NATURAL BLENDING LOGIC
# #             #     alpha_tex = (lens_res[:, :, 3].astype(float) / 255.0)
# #             #     alpha_mask = (final_mask.astype(float) / 255.0)
                
# #             #     # Dynamic Opacity: 0.85 makes it look less like "painted" on
# #             #     combined_alpha = (alpha_tex * alpha_mask) * 0.85
# #             #     alpha_3d = cv2.merge([combined_alpha] * 3)

# #             #     # MULTIPLY BLENDING: Asli reflections aur shadows ko upar laane ke liye
# #             #     # Yeh step "flat" look ko khatam karta hai
# #             #     eye_float = crop.astype(float) / 255.0
# #             #     lens_float = lens_rgb.astype(float) / 255.0
                
# #             #     # Blend: Multiply mode helps eye details show through the lens colors
# #             #     multiplied = cv2.multiply(eye_float, lens_float) * 1.5 # Boost brightness
# #             #     multiplied = np.clip(multiplied, 0, 1) * 255.0

# #             #     # Final Composite
# #             #     fg = multiplied * alpha_3d
# #             #     bg = crop.astype(float) * (1.0 - alpha_3d)
                
# #             #     frame[y1:y2, x1:x2] = cv2.add(fg, bg).astype(np.uint8)
# #             if lens_res.shape[2] == 4:
# #                 # 1. Alpha aur Mask ko merge karein
# #                 alpha_tex = (lens_res[:, :, 3].astype(float) / 255.0)
# #                 alpha_mask = (final_mask.astype(float) / 255.0)
                
# #                 # Combined alpha: 0.9 tak rakhein taaki color solid nazar aaye
# #                 combined_alpha = (alpha_tex * alpha_mask) * 0.9 
# #                 alpha_3d = cv2.merge([combined_alpha] * 3)

# #                 # 2. COLOR VIBRANCY LOGIC (Alpha Blending)
# #                 # Multiply ke bajaye standard alpha blending use karein taaki lens ka asli color dikhay
# #                 lens_rgb = lens_res[:, :, :3].astype(float)
# #                 eye_float = crop.astype(float)

# #                 # Formula: (Lens * Alpha) + (Aankh * (1 - Alpha))
# #                 # Is se lens ka apna color user ko wazay dikhai dega
# #                 composite = (lens_rgb * alpha_3d) + (eye_float * (1.0 - alpha_3d))
                
# #                 # 3. ADD HIGHLIGHTS (Reflections ko wapas laane ke liye)
# #                 # Aankh ke bright spots (reflections) ko lens ke upar "Screen" karein
# #                 gray_eye = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
# #                 highlights = np.where(gray_eye > 0.7, gray_eye, 0) # Sirf bright spots pakrein
# #                 highlights_3d = cv2.merge([highlights] * 3) * 50 # Intensity adjust karein
                
# #                 final_output = cv2.add(composite, highlights_3d)
# #                 frame[y1:y2, x1:x2] = np.clip(final_output, 0, 255).astype(np.uint8)

# #         except Exception: continue
# #     return frame

# class VideoProcessor(VideoTransformerBase):
#     def recv(self, frame):
#         img = frame.to_ndarray(format="bgr24")
#         img = cv2.flip(img, 1)
#         h_orig, w_orig = img.shape[:2]

#         # Process at 640x480 for real-time speed
#         img_proc = cv2.resize(img, (640, 480))
#         rgb = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)
        
#         results = face_mesh_tool.process(rgb)
#         if results.multi_face_landmarks:
#             img_proc = apply_hybrid_lens(img_proc, results.multi_face_landmarks[0].landmark, lens_img)
            
#         # Scaling back to original camera resolution
#         img_final = cv2.resize(img_proc, (w_orig, h_orig))
#         return VideoFrame.from_ndarray(img_final, format="bgr24")

# RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# webrtc_streamer(
#     key="ttdeye-final-pro",
#     video_processor_factory=VideoProcessor,
#     rtc_configuration=RTC_CONFIG,
#     async_processing=True,
#     media_stream_constraints={
#         "video": {"width": {"ideal": 640}, "frameRate": {"ideal": 20}},
#         "audio": False
#     }
# )

