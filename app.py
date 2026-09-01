import math
import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"
)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

INDEX_TIP = 8
THUMB_TIP = 4
PINCH_PX = 35

FILTERS = ["MATRIX", "PIXELATE", "THERMAL", "GLITCH"]
filter_idx = 0
was_pinching = False

# Recording Variables
is_recording = False
out_writer = None
OUTPUT_FILE = "retro_portal_output.mp4"


def ensure_model():
  if os.path.isfile(MODEL_PATH):
    return
  print("Downloading hand landmarker model...")
  urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def order_quad(points):
  pts = np.array(points, dtype=np.float32)
  s = pts.sum(axis=1)
  d = np.diff(pts, axis=1).ravel()
  return np.array(
      [
          pts[np.argmin(s)],
          pts[np.argmin(d)],
          pts[np.argmax(s)],
          pts[np.argmax(d)],
      ],
      dtype=np.int32,
  )


def filter_matrix(frame):
  h, w = frame.shape[:2]
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  blocks = 32
  small = cv2.resize(
      gray,
      (blocks, max(1, int(blocks * h / w))),
      interpolation=cv2.INTER_LINEAR,
  )
  pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

  matrix_bgr = np.zeros_like(frame)
  matrix_bgr[:, :, 1] = pixelated
  matrix_bgr[:, :, 0] = (pixelated * 0.15).astype(np.uint8)

  grid = matrix_bgr.copy()
  grid[::8, :] = 0
  grid[:, ::8] = 0

  return cv2.addWeighted(matrix_bgr, 0.7, grid, 0.3, 0)


def filter_pixelate(frame, blocks=28):
  h, w = frame.shape[:2]
  small = cv2.resize(
      frame,
      (blocks, max(1, int(blocks * h / w))),
      interpolation=cv2.INTER_LINEAR,
  )
  return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def filter_thermal(frame):
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def filter_glitch(frame):
  shift = 10
  b, g, r = cv2.split(frame)
  out = cv2.merge([np.roll(b, shift, axis=1), g, np.roll(r, -shift, axis=1)])
  out[::4] = (out[::4] * 0.55).astype(np.uint8)
  return out


FILTER_FNS = {
    "MATRIX": filter_matrix,
    "PIXELATE": filter_pixelate,
    "THERMAL": filter_thermal,
    "GLITCH": filter_glitch,
}


def apply_portal(frame, quad, filtered):
  h, w = frame.shape[:2]
  mask = np.zeros((h, w), dtype=np.uint8)
  cv2.fillConvexPoly(mask, quad, 255)
  mask_f = cv2.GaussianBlur(mask, (21, 21), 0).astype(np.float32) / 255.0
  mask_3 = mask_f[..., None]
  blended = (
      filtered.astype(np.float32) * mask_3
      + frame.astype(np.float32) * (1.0 - mask_3)
  ).astype(np.uint8)

  overlay = blended.copy()
  cv2.polylines(overlay, [quad], True, (0, 255, 255), 6, cv2.LINE_AA)
  cv2.polylines(overlay, [quad], True, (255, 255, 255), 2, cv2.LINE_AA)
  for pt in quad:
    cv2.circle(overlay, tuple(pt), 8, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(overlay, tuple(pt), 4, (255, 255, 255), -1, cv2.LINE_AA)
  cv2.addWeighted(overlay, 0.85, blended, 0.15, 0, blended)
  return blended, mask


ensure_model()

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
timestamp_ms = 0
frame_count = 0

# Grab width and height for video writer
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30.0

try:
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    frame = cv2.flip(frame, 1)
    frame_count += 1

    rgb_frame = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms += 33
    results = landmarker.detect_for_video(mp_image, timestamp_ms)

    corners = []
    pinching_now = False

    if results.hand_landmarks:
      for hand_landmarks in results.hand_landmarks:
        ix = int(hand_landmarks[INDEX_TIP].x * w)
        iy = int(hand_landmarks[INDEX_TIP].y * h)
        tx = int(hand_landmarks[THUMB_TIP].x * w)
        ty = int(hand_landmarks[THUMB_TIP].y * h)
        corners.append((ix, iy))
        corners.append((tx, ty))

        thumb = hand_landmarks[THUMB_TIP]
        index = hand_landmarks[INDEX_TIP]
        pinch_dist = math.hypot((thumb.x - index.x) * w, (thumb.y - index.y) * h)
        if pinch_dist < PINCH_PX:
          pinching_now = True

    if pinching_now and not was_pinching:
      filter_idx = (filter_idx + 1) % len(FILTERS)
    was_pinching = pinching_now

    name = FILTERS[filter_idx]

    if len(corners) == 4:
      quad = order_quad(corners)
      area = cv2.contourArea(quad.astype(np.float32))
      if area > 2500:
        filtered = FILTER_FNS[name](frame)
        frame, _ = apply_portal(frame, quad, filtered)

        top = quad[np.argmin(quad[:, 1])]
        label = f"PORTAL: {name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        lx = int(np.clip(top[0] - tw // 2, 8, w - tw - 8))
        ly = int(np.clip(top[1] - 18, th + 10, h - 10))
        cv2.putText(
            frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4
        )
        cv2.putText(
            frame,
            label,
            (lx, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

    # UI Instructions
    cv2.putText(
        frame,
        "Pinch: Change Filter | 'r': Record | 'q': Quit",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Active Filter: {name}",
        (12, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    # Recording Status Indicator
    if is_recording:
      if (frame_count // 15) % 2 == 0:  # Flashing indicator
        cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1)
      cv2.putText(
          frame,
          "REC",
          (w - 85, 36),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.6,
          (0, 0, 255),
          2,
      )
      out_writer.write(frame)

    cv2.imshow("RetroLens Python Portal", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
      break
    elif key == ord("r"):
      is_recording = not is_recording
      if is_recording:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (w, h))
        print(f"Started recording to {OUTPUT_FILE}")
      else:
        if out_writer:
          out_writer.release()
          out_writer = None
        print("Recording stopped & saved.")

finally:
  if out_writer:
    out_writer.release()
  cap.release()
  landmarker.close()
  cv2.destroyAllWindows()