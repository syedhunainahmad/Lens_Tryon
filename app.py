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



import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import tensorflow as tf
from av import VideoFrame
import mediapipe as mp
from mediapipe.python.solutions import face_mesh as mp_face_mesh

# --- 1. Resources Loading ---
@st.cache_resource
def load_assets():
    try:
        # Bytes reading for Mmap fix on Streamlit Cloud
        with open("iris_pure_float32.tflite", "rb") as f:
            model_content = f.read()
        interpreter = tf.lite.Interpreter(model_content=model_content)
        interpreter.allocate_tensors()
    except Exception as e:
        st.error(f"Error loading TFLite model: {e}")
        return None, None

    # Load lens with Alpha Channel
    lens_img = cv2.imread("images/1.png", cv2.IMREAD_UNCHANGED)
    if lens_img is None:
        st.error("Lens image not found in images/1.png")
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
    # UNet processing (Fixed 384x384)
    img = cv2.resize(crop, (384, 384)).astype(np.float32) / 255.0
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img, axis=0))
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0]
    
    mask = (np.squeeze(pred) > 0.3).astype(np.uint8) * 255
    return cv2.resize(mask, (crop.shape[1], crop.shape[0]))

def apply_hybrid_lens(frame, landmarks, lens_texture):
    h, w = frame.shape[:2]
    
    # Eyelid points for precise occlusion (palkon ke peeche)
    LEFT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_POINTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    # Iris logic (Center, Top, Bottom, Edge, Eyelid)
    eye_configs = [
        (468, 159, 145, 469, LEFT_EYE_POINTS), 
        (473, 386, 374, 474, RIGHT_EYE_POINTS) 
    ]

    for iris_idx, top_idx, bot_idx, edge_idx, eye_pts in eye_configs:
        try:
            # 1. Coordinate Calculation
            cx, cy = int(landmarks[iris_idx].x * w), int(landmarks[iris_idx].y * h)
            ex, ey = int(landmarks[edge_idx].x * w), int(landmarks[edge_idx].y * h)
            
            # Iris radius calculate karke thora margin dena
            r = int(np.sqrt((cx - ex)**2 + (cy - ey)**2) * 1.15)
            
            y1, y2, x1, x2 = max(0, cy-r), min(h, cy+r), max(0, cx-r), min(w, cx+r)
            crop = frame[y1:y2, x1:x2].copy()
            if crop.size == 0: continue
            ch, cw = crop.shape[:2]

            # 2. Hybrid Mask Creation
            model_mask = predict_mask_with_model(crop) 
            geo_mask = np.zeros((ch, cw), dtype=np.uint8)
            cv2.circle(geo_mask, (cw//2, ch//2), int(r * 0.95), 255, -1)

            # Eyelid clipping
            eye_poly = np.array([[(landmarks[p].x*w - x1), (landmarks[p].y*h - y1)] for p in eye_pts], dtype=np.int32)
            occlusion_mask = np.zeros((ch, cw), dtype=np.uint8)
            cv2.fillPoly(occlusion_mask, [eye_poly], 255)

            # Final mask combining UNet and Geometry
            final_mask = cv2.bitwise_and(cv2.bitwise_or(geo_mask, model_mask), occlusion_mask)
            final_mask = cv2.GaussianBlur(final_mask, (3, 3), 0) # Smooth but Sharp

            # 3. High-Quality Texture Blending
            # LANCZOS4 ensures textures remain clear when resized
            lens_res = cv2.resize(lens_texture, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
            
            if lens_res.shape[2] == 4:
                # Get Alpha channels
                alpha_tex = (lens_res[:, :, 3].astype(float) / 255.0)
                alpha_mask = (final_mask.astype(float) / 255.0)
                
                # Multiply both for combined transparency
                alpha_final = alpha_tex * alpha_mask
                alpha_3d = cv2.merge([alpha_final] * 3)
                
                fg = lens_res[:, :, :3].astype(float) * alpha_3d
                bg = crop.astype(float) * (1.0 - alpha_3d)
                
                frame[y1:y2, x1:x2] = cv2.add(fg, bg).astype(np.uint8)

        except Exception:
            continue
    return frame

class VideoProcessor(VideoTransformerBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h_orig, w_orig = img.shape[:2]

        # Process at 640x480 for real-time speed
        img_proc = cv2.resize(img, (640, 480))
        rgb = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)
        
        results = face_mesh_tool.process(rgb)
        if results.multi_face_landmarks:
            img_proc = apply_hybrid_lens(img_proc, results.multi_face_landmarks[0].landmark, lens_img)
            
        # Scaling back to original camera resolution
        img_final = cv2.resize(img_proc, (w_orig, h_orig))
        return VideoFrame.from_ndarray(img_final, format="bgr24")

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_streamer(
    key="ttdeye-final-pro",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIG,
    async_processing=True,
    media_stream_constraints={
        "video": {"width": {"ideal": 640}, "frameRate": {"ideal": 20}},
        "audio": False
    }
)
