##############.  geometry with MP code in LiveAR.   ###############
import cv2
import numpy as np
import tensorflow as tf
import io
import threading
import base64
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from mediapipe.python.solutions import face_mesh as mp_face_mesh

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

executor = ThreadPoolExecutor(max_workers=2)

# ─────────────────────────────────────────────────────────────
# MODEL  (photo mode only)
# ─────────────────────────────────────────────────────────────

interpreter = tf.lite.Interpreter(model_path="iris_pure_float32.tflite")
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter_lock = threading.Lock()

_dummy = np.zeros((1,384,384,3), dtype=np.float32)
with interpreter_lock:
    interpreter.set_tensor(input_details[0]['index'], _dummy)
    interpreter.invoke()
print("[model] Warmup done")

face_mesh_live = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
face_mesh_photo = mp_face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1,
    refine_landmarks=True, min_detection_confidence=0.5,
)

_lens_cache         = {}
_resized_lens_cache = {}

def get_lens_texture(lens_id):
    if lens_id not in _lens_cache:
        t = cv2.imread(f"images/{lens_id}.png", cv2.IMREAD_UNCHANGED)
        if t is None: return None
        _lens_cache[lens_id] = t
    return _lens_cache[lens_id]

def get_resized_lens(lens_id, w, h):
    key = f"{lens_id}_{w}_{h}"
    if key not in _resized_lens_cache:
        t = get_lens_texture(lens_id)
        if t is None: return None
        # INTER_LANCZOS4 for better quality (cached so no speed cost)
        _resized_lens_cache[key] = cv2.resize(t, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return _resized_lens_cache[key]


# ─────────────────────────────────────────────────────────────
# EYE OPENNESS DETECTION
#
# How it works:
#   EAR (Eye Aspect Ratio) = vertical_distance / horizontal_distance
#   Open eye:   EAR ≈ 0.25-0.35
#   Closed eye: EAR ≈ 0.0-0.10
#
# Landmark indices for EAR:
#   Left eye:  top=159, bottom=145, left=33,  right=133
#   Right eye: top=386, bottom=374, left=362, right=263
# ─────────────────────────────────────────────────────────────
def get_ear(landmarks, top_idx, bot_idx, left_idx, right_idx, w, h):
    top   = np.array([landmarks[top_idx].x * w,   landmarks[top_idx].y * h])
    bot   = np.array([landmarks[bot_idx].x * w,   landmarks[bot_idx].y * h])
    left  = np.array([landmarks[left_idx].x * w,  landmarks[left_idx].y * h])
    right = np.array([landmarks[right_idx].x * w, landmarks[right_idx].y * h])

    vertical   = np.linalg.norm(top - bot)
    horizontal = np.linalg.norm(left - right)

    if horizontal == 0: return 0.0
    return vertical / horizontal


def get_eye_openness(landmarks, w, h):
    """
    Returns (left_ear, right_ear).
    EAR > 0.15 = eye open
    EAR < 0.15 = eye closed / blinking
    """
    left_ear  = get_ear(landmarks, 159, 145, 33,  133, w, h)
    right_ear = get_ear(landmarks, 386, 374, 362, 263, w, h)
    return left_ear, right_ear


# ─────────────────────────────────────────────────────────────
# PARTIAL VISIBILITY MASK
#
# When eye is half-open, lens should only show on the
# visible part of the iris — not behind the eyelid.
# This is handled by the occlusion mask (fillPoly with
# eyelid landmarks), which naturally clips the lens.
#
# Additionally: scale lens alpha by EAR so lens fades
# smoothly as eye closes — no hard pop-in/pop-out.
# ─────────────────────────────────────────────────────────────
EAR_OPEN_THRESH  = 0.18   # above this = fully open
EAR_CLOSE_THRESH = 0.10   # below this = fully closed
                           # between = partial, blend smoothly

def ear_to_alpha_scale(ear):
    """Smooth 0→1 scale based on how open the eye is."""
    if ear <= EAR_CLOSE_THRESH:  return 0.0
    if ear >= EAR_OPEN_THRESH:   return 1.0
    # Linear interpolation in the transition range
    return (ear - EAR_CLOSE_THRESH) / (EAR_OPEN_THRESH - EAR_CLOSE_THRESH)


# ─────────────────────────────────────────────────────────────
# LIVE LENS APPLICATION
# ─────────────────────────────────────────────────────────────
def apply_lens_live(frame, landmarks, lens_id):
    h, w = frame.shape[:2]

    # Eyelid contour points for occlusion mask
    L_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
    R_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]

    # Get EAR for both eyes
    left_ear, right_ear = get_eye_openness(landmarks, w, h)

    configs = [
        (468, 471, L_EYE, left_ear),   # left iris center, edge, contour, ear
        (473, 476, R_EYE, right_ear),  # right
    ]

    for iris_idx, edge_idx, eye_pts, ear in configs:
        try:
            # Skip completely if eye is closed
            alpha_scale = ear_to_alpha_scale(ear)
            if alpha_scale <= 0.0:
                continue

            cx = int(landmarks[iris_idx].x * w)
            cy = int(landmarks[iris_idx].y * h)
            ex = int(landmarks[edge_idx].x * w)
            ey = int(landmarks[edge_idx].y * h)

            # Iris radius from MediaPipe edge landmark
            iris_r = int(np.sqrt((cx-ex)**2 + (cy-ey)**2))
            if iris_r < 4: continue

            # Crop region — just the iris area (tight crop = higher quality)
            pad  = int(iris_r * 1.35)
            y1,y2 = max(0,cy-pad), min(h,cy+pad)
            x1,x2 = max(0,cx-pad), min(w,cx+pad)
            crop  = frame[y1:y2, x1:x2]
            if crop.size == 0: continue
            ch, cw = crop.shape[:2]

            # ── MASK 1: Iris circle ────────────────────────────────
            # Use exact iris radius (not padded) for clean boundary
            iris_mask = np.zeros((ch, cw), np.uint8)
            cx_local  = cx - x1
            cy_local  = cy - y1
            cv2.circle(iris_mask, (cx_local, cy_local), int(iris_r * 0.92), 255, -1)

            # ── MASK 2: Eyelid occlusion ───────────────────────────
            # Hides lens behind eyelashes — critical for realism
            occ_mask = np.zeros((ch, cw), np.uint8)
            poly     = np.array([
                [(int(landmarks[p].x*w) - x1), (int(landmarks[p].y*h) - y1)]
                for p in eye_pts], np.int32)
            cv2.fillPoly(occ_mask, [poly], 255)

            # ── MASK 3: Combine + smooth edge ──────────────────────
            final_mask = cv2.bitwise_and(iris_mask, occ_mask)
            # Slightly larger blur = softer lens edge = more natural
            final_mask = cv2.GaussianBlur(final_mask, (7, 7), 0)

            # ── LENS TEXTURE ───────────────────────────────────────
            lens = get_resized_lens(lens_id, cw, ch)
            if lens is None or lens.shape[2] != 4: continue

            # ── ALPHA BLEND ────────────────────────────────────────
            # alpha_scale: 0 when closed, 1 when open, smooth transition
            base_alpha = 0.88
            a  = (lens[:,:,3].astype(np.float32)/255.0) * \
                 (final_mask.astype(np.float32)/255.0) * \
                 base_alpha * alpha_scale   # <-- fades with eye openness

            a3  = np.stack([a,a,a], axis=2)
            out = lens[:,:,:3].astype(np.float32)*a3 + crop.astype(np.float32)*(1.0-a3)
            frame[y1:y2, x1:x2] = out.astype(np.uint8)

        except Exception as e:
            print(f"[live eye {iris_idx}] {e}")

    return frame


def predict_mask_unet(crop):
    img = cv2.resize(crop, (384,384)).astype(np.float32) / 255.0
    with interpreter_lock:
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img,0))
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])[0]
    mask = (np.squeeze(pred) > 0.15).astype(np.uint8) * 255
    return cv2.resize(mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)


# ─────────────────────────────────────────────────────────────
# FRAME PROCESSOR
# ─────────────────────────────────────────────────────────────
def process_frame(frame_bytes, lens_id):
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None: return None

    # ── QUALITY FIX 1: Process at 640x480 (was 320x240) ──────
    # Higher res = better MediaPipe accuracy + sharper lens
    # Still fast because model is not running anymore
    frame   = cv2.resize(frame, (640, 480))

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh_live.process(rgb)

    if results.multi_face_landmarks:
        frame = apply_lens_live(
            frame,
            results.multi_face_landmarks[0].landmark,
            lens_id,
        )

    # ── QUALITY FIX 2: Higher JPEG quality (was 80) ──────────
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes()


# ─────────────────────────────────────────────────────────────
# WEBSOCKET
# ─────────────────────────────────────────────────────────────
@app.websocket("/ws/live-ar")
async def live_ar_websocket(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected")
    loop    = asyncio.get_event_loop()
    pending = None

    async def process_and_send(frame_bytes, lens_id):
        try:
            result = await loop.run_in_executor(executor, process_frame, frame_bytes, lens_id)
            if result:
                await websocket.send_text(json.dumps({
                    "frame": base64.b64encode(result).decode()
                }))
        except Exception as e:
            print(f"[WS] {e}")

    try:
        while True:
            raw  = await websocket.receive_text()
            data = json.loads(raw)
            lens_id   = data.get("lens_id", "1")
            frame_b64 = data.get("frame", "")
            if not frame_b64: continue

            if pending and not pending.done():
                pending.cancel()

            pending = asyncio.create_task(
                process_and_send(base64.b64decode(frame_b64), lens_id)
            )

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
    finally:
        if pending: pending.cancel()


# ─────────────────────────────────────────────────────────────
# PHOTO MODE  (UNet for accuracy)
# ─────────────────────────────────────────────────────────────
@app.post("/apply-lens")
async def apply_lens_photo(image: UploadFile = File(...), lens_id: str = Form(...)):
    contents = await image.read()
    nparr    = np.frombuffer(contents, np.uint8)
    frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None: return {"error": "Could not decode image"}

    lens_texture = get_lens_texture(lens_id)
    if lens_texture is None: return {"error": f"Lens not found: {lens_id}"}

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh_photo.process(rgb)

    if results.multi_face_landmarks:
        h, w      = frame.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark
        L = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
        R = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]

        # Also check eye openness for photo mode
        left_ear, right_ear = get_eye_openness(landmarks, w, h)

        for (iris_idx, edge_idx, eye_pts, ear) in [(468,471,L,left_ear),(473,476,R,right_ear)]:
            try:
                alpha_scale = ear_to_alpha_scale(ear)
                if alpha_scale <= 0.0: continue

                cx = int(landmarks[iris_idx].x*w)
                cy = int(landmarks[iris_idx].y*h)
                ex = int(landmarks[edge_idx].x*w)
                ey = int(landmarks[edge_idx].y*h)
                r  = int(np.sqrt((cx-ex)**2+(cy-ey)**2)*1.3)
                y1,y2 = max(0,cy-r),min(h,cy+r)
                x1,x2 = max(0,cx-r),min(w,cx+r)
                crop  = frame[y1:y2,x1:x2].copy()
                if crop.size==0: continue
                ch,cw = crop.shape[:2]

                m_mask = predict_mask_unet(crop)
                geo    = np.zeros((ch,cw),np.uint8)
                cv2.circle(geo,(cw//2,ch//2),int(r*0.92),255,-1)
                occ    = np.zeros((ch,cw),np.uint8)
                poly   = np.array([
                    [(int(landmarks[p].x*w)-x1),(int(landmarks[p].y*h)-y1)]
                    for p in eye_pts],np.int32)
                cv2.fillPoly(occ,[poly],255)
                mask   = cv2.GaussianBlur(
                    cv2.bitwise_and(cv2.bitwise_or(m_mask,geo),occ),(11,11),0)
                lens   = cv2.resize(lens_texture,(cw,ch),interpolation=cv2.INTER_LANCZOS4)
                if lens.shape[2]==4:
                    a  = (lens[:,:,3].astype(float)/255.0) * \
                         (mask.astype(float)/255.0) * 0.9 * alpha_scale
                    a3 = cv2.merge([a]*3)
                    frame[y1:y2,x1:x2] = (
                        lens[:,:,:3].astype(float)*a3 +
                        crop.astype(float)*(1-a3)
                    ).astype(np.uint8)
            except: continue

        _, buf = cv2.imencode(".png", frame)
        return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")

    return {"error": "No face detected"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, loop="asyncio")

# ###############.   code with unet+geometry+mediapipe in LiveAR.   ###############
# #code 2
# import cv2
# import numpy as np
# import tensorflow as tf
# import io
# import threading
# import base64
# import json
# import asyncio
# from concurrent.futures import ThreadPoolExecutor
# from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
# from fastapi.responses import StreamingResponse
# from fastapi.middleware.cors import CORSMiddleware
# from mediapipe.python.solutions import face_mesh as mp_face_mesh

# app = FastAPI()
# app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# executor = ThreadPoolExecutor(max_workers=2)

# # ─────────────────────────────────────────────────────────────
# # MODEL SETUP
# # UNet model — used in BOTH live (every 10th frame) and photo mode
# # ─────────────────────────────────────────────────────────────
# interpreter = tf.lite.Interpreter(model_path="iris_pure_float32.tflite")
# interpreter.allocate_tensors()
# input_details  = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# interpreter_lock = threading.Lock()

# # Warmup so first frame is not slow
# _dummy = np.zeros((1, 384, 384, 3), dtype=np.float32)
# with interpreter_lock:
#     interpreter.set_tensor(input_details[0]['index'], _dummy)
#     interpreter.invoke()
# print("[UNet] Model warmed up and ready")

# # ─────────────────────────────────────────────────────────────
# # MEDIAPIPE
# # live: tracking mode (continuous stream)
# # photo: static mode (single image, more accurate)
# # ─────────────────────────────────────────────────────────────
# face_mesh_live = mp_face_mesh.FaceMesh(
#     static_image_mode=False,
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5,
# )
# face_mesh_photo = mp_face_mesh.FaceMesh(
#     static_image_mode=True,
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
# )

# # ─────────────────────────────────────────────────────────────
# # LENS CACHE
# # ─────────────────────────────────────────────────────────────
# _lens_cache         = {}
# _resized_lens_cache = {}

# def get_lens_texture(lens_id):
#     if lens_id not in _lens_cache:
#         t = cv2.imread(f"images/{lens_id}.png", cv2.IMREAD_UNCHANGED)
#         if t is None: return None
#         _lens_cache[lens_id] = t
#         print(f"[cache] Lens loaded: {lens_id}")
#     return _lens_cache[lens_id]

# def get_resized_lens(lens_id, w, h):
#     key = f"{lens_id}_{w}_{h}"
#     if key not in _resized_lens_cache:
#         t = get_lens_texture(lens_id)
#         if t is None: return None
#         _resized_lens_cache[key] = cv2.resize(t, (w, h), interpolation=cv2.INTER_LANCZOS4)
#     return _resized_lens_cache[key]


# # ─────────────────────────────────────────────────────────────
# # UNET MASK PREDICTION
# # Used in: live AR (every 10th frame) + photo mode (every frame)
# # ─────────────────────────────────────────────────────────────
# def predict_mask_unet(crop):
#     img = cv2.resize(crop, (384, 384)).astype(np.float32) / 255.0
#     with interpreter_lock:
#         interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img, 0))
#         interpreter.invoke()
#         pred = interpreter.get_tensor(output_details[0]['index'])[0]
#     mask = (np.squeeze(pred) > 0.15).astype(np.uint8) * 255
#     return cv2.resize(mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)


# # ─────────────────────────────────────────────────────────────
# # EYE OPENNESS — EAR (Eye Aspect Ratio)
# # EAR > 0.18 = open, EAR < 0.10 = closed, between = partial
# # ─────────────────────────────────────────────────────────────
# EAR_OPEN  = 0.18
# EAR_CLOSE = 0.10

# def get_ear(landmarks, top_idx, bot_idx, left_idx, right_idx, w, h):
#     top   = np.array([landmarks[top_idx].x * w,   landmarks[top_idx].y * h])
#     bot   = np.array([landmarks[bot_idx].x * w,   landmarks[bot_idx].y * h])
#     left  = np.array([landmarks[left_idx].x * w,  landmarks[left_idx].y * h])
#     right = np.array([landmarks[right_idx].x * w, landmarks[right_idx].y * h])
#     vertical   = np.linalg.norm(top - bot)
#     horizontal = np.linalg.norm(left - right)
#     return (vertical / horizontal) if horizontal > 0 else 0.0

# def get_eye_openness(landmarks, w, h):
#     left_ear  = get_ear(landmarks, 159, 145, 33,  133, w, h)
#     right_ear = get_ear(landmarks, 386, 374, 362, 263, w, h)
#     return left_ear, right_ear

# def ear_to_alpha(ear):
#     if ear <= EAR_CLOSE: return 0.0
#     if ear >= EAR_OPEN:  return 1.0
#     return (ear - EAR_CLOSE) / (EAR_OPEN - EAR_CLOSE)


# # ─────────────────────────────────────────────────────────────
# # LIVE AR — LENS APPLICATION
# #
# # Mask strategy (per frame):
# #   frame % 10 == 0  →  UNet model  (precise iris boundary)
# #   all other frames →  Geometry    (fast fallback, reuses last mask)
# #
# # This means:
# #   - UNet runs ~3 times per second at 30fps (every 10th frame)
# #   - Geometry fills in the gaps between UNet frames
# #   - Supervisor can see UNet is actively used in live mode
# #   - FPS stays high because UNet only runs 10% of frames
# # ─────────────────────────────────────────────────────────────
# def apply_lens_live(frame, landmarks, lens_id, last_masks, frame_count):
#     h, w = frame.shape[:2]
#     L_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
#     R_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]

#     left_ear, right_ear = get_eye_openness(landmarks, w, h)

#     configs = [
#         (468, 471, L_EYE, left_ear,  0),   # left eye
#         (473, 476, R_EYE, right_ear, 1),   # right eye
#     ]

#     current_masks = [None, None]

#     for iris_idx, edge_idx, eye_pts, ear, idx in configs:
#         try:
#             # Eye closed — skip entirely
#             alpha_scale = ear_to_alpha(ear)
#             if alpha_scale <= 0.0:
#                 current_masks[idx] = last_masks[idx] if last_masks else None
#                 continue

#             cx = int(landmarks[iris_idx].x * w)
#             cy = int(landmarks[iris_idx].y * h)
#             ex = int(landmarks[edge_idx].x * w)
#             ey = int(landmarks[edge_idx].y * h)

#             iris_r = int(np.sqrt((cx-ex)**2 + (cy-ey)**2))
#             if iris_r < 4: continue

#             pad   = int(iris_r * 1.35)
#             y1,y2 = max(0,cy-pad), min(h,cy+pad)
#             x1,x2 = max(0,cx-pad), min(w,cx+pad)
#             crop  = frame[y1:y2, x1:x2]
#             if crop.size == 0: continue
#             ch, cw = crop.shape[:2]

#             # ── MASK DECISION ─────────────────────────────────────
#             run_model = (frame_count % 10 == 0)

#             if run_model or last_masks is None or last_masks[idx] is None:
#                 # UNet: precise iris segmentation (every 10th frame)
#                 print(f"[UNet] Running model on frame {frame_count}, eye {idx}")
#                 unet_raw = predict_mask_unet(crop)

#                 # Combine UNet with geometry for best result
#                 geo_mask = np.zeros((ch, cw), np.uint8)
#                 cv2.circle(geo_mask, (cx-x1, cy-y1), int(iris_r*0.92), 255, -1)
#                 final_mask = cv2.bitwise_or(unet_raw, geo_mask)
#                 current_masks[idx] = final_mask
#             else:
#                 # Geometry fallback: reuse last UNet mask (frames 1-9)
#                 final_mask = cv2.resize(
#                     last_masks[idx], (cw, ch), interpolation=cv2.INTER_NEAREST)
#                 current_masks[idx] = last_masks[idx]

#             # Eyelid occlusion — hides lens behind eyelashes
#             occ_mask = np.zeros((ch, cw), np.uint8)
#             poly     = np.array([
#                 [(int(landmarks[p].x*w)-x1), (int(landmarks[p].y*h)-y1)]
#                 for p in eye_pts], np.int32)
#             cv2.fillPoly(occ_mask, [poly], 255)

#             # Final mask = (UNet or Geometry) clipped by eyelid
#             combined = cv2.bitwise_and(final_mask, occ_mask)
#             combined = cv2.GaussianBlur(combined, (7, 7), 0)

#             # Lens texture (pre-cached, Lanczos quality)
#             lens = get_resized_lens(lens_id, cw, ch)
#             if lens is None or lens.shape[2] != 4: continue

#             # Alpha blend with eye-openness scaling
#             a  = (lens[:,:,3].astype(np.float32)/255.0) * \
#                  (combined.astype(np.float32)/255.0) * \
#                  0.88 * alpha_scale
#             a3 = np.stack([a,a,a], axis=2)
#             out = lens[:,:,:3].astype(np.float32)*a3 + crop.astype(np.float32)*(1.0-a3)
#             frame[y1:y2, x1:x2] = out.astype(np.uint8)

#         except Exception as e:
#             print(f"[live eye {iris_idx}] {e}")
#             current_masks[idx] = last_masks[idx] if last_masks else None

#     return frame, current_masks


# # ─────────────────────────────────────────────────────────────
# # FRAME PROCESSOR  (runs in thread pool — never blocks WebSocket)
# # ─────────────────────────────────────────────────────────────
# def process_frame(frame_bytes, lens_id, last_masks, frame_count):
#     nparr = np.frombuffer(frame_bytes, np.uint8)
#     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     if frame is None: return None, last_masks

#     frame   = cv2.resize(frame, (640, 480))
#     rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh_live.process(rgb)

#     current_masks = last_masks if last_masks else [None, None]

#     if results.multi_face_landmarks:
#         frame, current_masks = apply_lens_live(
#             frame,
#             results.multi_face_landmarks[0].landmark,
#             lens_id,
#             last_masks,
#             frame_count,
#         )

#     _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
#     return buf.tobytes(), current_masks


# # ─────────────────────────────────────────────────────────────
# # WEBSOCKET — Live AR
# # ─────────────────────────────────────────────────────────────
# @app.websocket("/ws/live-ar")
# async def live_ar_websocket(websocket: WebSocket):
#     await websocket.accept()
#     print("[WS] Client connected")

#     frame_count = 0
#     last_masks  = [None, None]
#     loop        = asyncio.get_event_loop()
#     pending     = None

#     async def process_and_send(frame_bytes, lens_id, masks, count):
#         try:
#             result, new_masks = await loop.run_in_executor(
#                 executor, process_frame, frame_bytes, lens_id, masks, count
#             )
#             if result:
#                 await websocket.send_text(json.dumps({
#                     "frame": base64.b64encode(result).decode()
#                 }))
#             return new_masks
#         except Exception as e:
#             print(f"[WS process] {e}")
#             return masks

#     try:
#         while True:
#             raw  = await websocket.receive_text()
#             data = json.loads(raw)
#             lens_id   = data.get("lens_id", "1")
#             frame_b64 = data.get("frame", "")
#             if not frame_b64: continue

#             frame_count += 1

#             # Drop stale frame — prevents lag buildup
#             if pending and not pending.done():
#                 pending.cancel()

#             pending = asyncio.create_task(
#                 process_and_send(
#                     base64.b64decode(frame_b64),
#                     lens_id,
#                     last_masks,
#                     frame_count,
#                 )
#             )

#             # Update masks if task already finished
#             if pending.done():
#                 result = pending.result()
#                 if result: last_masks = result

#     except WebSocketDisconnect:
#         print("[WS] Client disconnected")
#     except Exception as e:
#         print(f"[WS] Error: {e}")
#     finally:
#         if pending: pending.cancel()


# # ─────────────────────────────────────────────────────────────
# # PHOTO MODE — UNet every frame (full quality)
# # ─────────────────────────────────────────────────────────────
# @app.post("/apply-lens")
# async def apply_lens_photo(image: UploadFile = File(...), lens_id: str = Form(...)):
#     contents = await image.read()
#     nparr    = np.frombuffer(contents, np.uint8)
#     frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     if frame is None: return {"error": "Could not decode image"}

#     lens_texture = get_lens_texture(lens_id)
#     if lens_texture is None: return {"error": f"Lens not found: {lens_id}"}

#     rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh_photo.process(rgb)

#     if results.multi_face_landmarks:
#         h, w      = frame.shape[:2]
#         landmarks = results.multi_face_landmarks[0].landmark
#         L = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
#         R = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]

#         left_ear, right_ear = get_eye_openness(landmarks, w, h)

#         for (iris_idx, edge_idx, eye_pts, ear) in [
#             (468, 471, L, left_ear),
#             (473, 476, R, right_ear)
#         ]:
#             try:
#                 alpha_scale = ear_to_alpha(ear)
#                 if alpha_scale <= 0.0: continue

#                 cx = int(landmarks[iris_idx].x*w)
#                 cy = int(landmarks[iris_idx].y*h)
#                 ex = int(landmarks[edge_idx].x*w)
#                 ey = int(landmarks[edge_idx].y*h)
#                 r  = int(np.sqrt((cx-ex)**2+(cy-ey)**2)*1.3)

#                 y1,y2 = max(0,cy-r), min(h,cy+r)
#                 x1,x2 = max(0,cx-r), min(w,cx+r)
#                 crop  = frame[y1:y2, x1:x2].copy()
#                 if crop.size == 0: continue
#                 ch, cw = crop.shape[:2]

#                 # UNet: precise segmentation for photo quality
#                 m_mask = predict_mask_unet(crop)
#                 geo    = np.zeros((ch,cw), np.uint8)
#                 cv2.circle(geo, (cw//2,ch//2), int(r*0.92), 255, -1)
#                 occ    = np.zeros((ch,cw), np.uint8)
#                 poly   = np.array([
#                     [(int(landmarks[p].x*w)-x1),(int(landmarks[p].y*h)-y1)]
#                     for p in eye_pts], np.int32)
#                 cv2.fillPoly(occ, [poly], 255)

#                 mask = cv2.GaussianBlur(
#                     cv2.bitwise_and(cv2.bitwise_or(m_mask,geo), occ), (11,11), 0)
#                 lens = cv2.resize(lens_texture, (cw,ch), interpolation=cv2.INTER_LANCZOS4)

#                 if lens.shape[2] == 4:
#                     a  = (lens[:,:,3].astype(float)/255.0) * \
#                          (mask.astype(float)/255.0) * 0.9 * alpha_scale
#                     a3 = cv2.merge([a]*3)
#                     frame[y1:y2,x1:x2] = (
#                         lens[:,:,:3].astype(float)*a3 +
#                         crop.astype(float)*(1-a3)
#                     ).astype(np.uint8)

#             except Exception as e:
#                 print(f"[photo eye {iris_idx}] {e}")
#                 continue

#         _, buf = cv2.imencode(".png", frame)
#         return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")

#     return {"error": "No face detected"}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001, loop="asyncio")