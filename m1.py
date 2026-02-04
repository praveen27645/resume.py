import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from collections import deque

# Haptic Free Music Player Ready
# Initialize MediaPipe Hands and Face Mesh
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.8, min_tracking_confidence=0.8)

cap = cv2.VideoCapture(0)

# Volume Control Variables
VOLUME_MIN = 0
VOLUME_MAX = 100
UPDATE_INTERVAL = 0.1
SAFE_ZONE_THRESHOLD = 20
SMOOTHING_WINDOW = 5
prev_volume = 50
last_update_time = time.time()
finger_distances = deque(maxlen=SMOOTHING_WINDOW)
volume_control_enabled = False # Flag to toggle volume control

# Gesture detection variables
TOUCH_THRESHOLD = 0.03
mute_toggle = False
prev_action_time = 0
first_right_tap_time = 0
first_left_tap_time = 0

action_text = "" # To display detected action on the screen

# Function to display action text on video feed
def display_action(frame, text, position=(50, 400)):
  cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

def detect_touch(landmarks, face_landmarks, target_points):
  if not all(idx in face_landmarks for idx in target_points):
    return False
  target_x = np.mean([face_landmarks[idx][0] for idx in target_points])
  target_y = np.mean([face_landmarks[idx][1] for idx in target_points])
  index_finger_x, index_finger_y = landmarks[8]
  distance = np.linalg.norm(np.array([target_x, target_y]) - np.array([index_finger_x, index_finger_y]))
  return distance < TOUCH_THRESHOLD

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break
  frame = cv2.flip(frame, 1)
  frame_height, frame_width, _ = frame.shape
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  
  results_hands = hands.process(frame_rgb)
  results_face = face_mesh.process(frame_rgb)
  
  hand_landmarks = []
  face_landmarks = {}
  left_hand_detected = False
  right_hand_detected = False
  left_palm_detected = False
  
  if results_face.multi_face_landmarks:
    for face_landmark in results_face.multi_face_landmarks:
      for idx, lm in enumerate(face_landmark.landmark):
        face_landmarks[idx] = (lm.x, lm.y)

  if results_hands.multi_hand_landmarks:
    for hand_landmark in results_hands.multi_hand_landmarks:
      landmarks = [(lm.x, lm.y) for lm in hand_landmark.landmark]
      hand_landmarks.append(landmarks)
      mp_drawing.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

      if landmarks[5][0] < 0.5:
        left_hand_detected = True
        left_hand_landmarks = landmarks

        palm_width = np.linalg.norm(np.array(landmarks[0]) - np.array(landmarks[17]))
        finger_spread = np.linalg.norm(np.array(landmarks[5]) - np.array(landmarks[17]))

        if finger_spread > palm_width * 0.8:
          left_palm_detected = True
          palm_center_x = int((landmarks[0][0] + landmarks[9][0]) / 2 * frame_width)
          palm_center_y = int((landmarks[0][1] + landmarks[9][1]) / 2 * frame_height)
          palm_radius = int(palm_width * frame_width * 0.5)
          cv2.circle(frame, (palm_center_x, palm_center_y), palm_radius, (255, 0, 0), 2)

      else:
        right_hand_detected = True
        right_hand_landmarks = landmarks

  # Toggle Volume Control On/Off using Left Palm
  if left_palm_detected:
    if time.time() - prev_action_time > 1:
      volume_control_enabled = not volume_control_enabled
      prev_action_time = time.time()
      if volume_control_enabled:
        print("Volume control enabled.")
      else:
        print("Volume control disabled.")

  # Volume Control (Only When Enabled)
  if volume_control_enabled and left_hand_detected:
    thumb_tip = left_hand_landmarks[4]
    index_tip = left_hand_landmarks[8]
    
    thumb_x, thumb_y = int(thumb_tip[0] * frame_width), int(thumb_tip[1] * frame_height)
    index_x, index_y = int(index_tip[0] * frame_width), int(index_tip[1] * frame_height)
    
    finger_distance = np.linalg.norm(np.array([thumb_x, thumb_y]) - np.array([index_x, index_y]))
    finger_distances.append(finger_distance)
    avg_distance = np.mean(finger_distances)

    if avg_distance > SAFE_ZONE_THRESHOLD:
      new_volume = np.interp(avg_distance, [5, 150], [VOLUME_MIN, VOLUME_MAX])
      new_volume = int(np.clip(new_volume, VOLUME_MIN, VOLUME_MAX))

      current_time = time.time()
      if current_time - last_update_time > UPDATE_INTERVAL:
        step = max(1, abs(new_volume - prev_volume) // 5)
        if new_volume > prev_volume:
          pyautogui.press('volumeup', presses=step)
          print(f"Volume increased to {min(prev_volume + step, VOLUME_MAX)}")
          prev_volume = min(prev_volume + step, VOLUME_MAX)
          action_text = f"Volume: {prev_volume}"
        elif new_volume < prev_volume:
          pyautogui.press('volumedown', presses=step)
          print(f"Volume decreased to {max(prev_volume - step, VOLUME_MIN)}")
          prev_volume = max(prev_volume - step, VOLUME_MIN)
          action_text = f"Volume: {prev_volume}"
        last_update_time = current_time

    # Draw Volume Bar
    bar_x = 50
    bar_y = 100
    bar_height = 200
    volume_bar = int((new_volume / VOLUME_MAX) * bar_height)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 20, bar_y + bar_height), (200, 200, 200), 2)
    cv2.rectangle(frame, (bar_x, bar_y + bar_height - volume_bar), (bar_x + 20, bar_y + bar_height), (0, 255, 0), -1)

  # Music Controls (Right Hand)
  current_time = time.time()
  if face_landmarks and hand_landmarks:
    mouth_touched = any(detect_touch(landmarks, face_landmarks, [13, 14]) for landmarks in hand_landmarks)
    right_ear_touched = any(detect_touch(landmarks, face_landmarks, [234]) for landmarks in hand_landmarks)
    left_ear_touched = any(detect_touch(landmarks, face_landmarks, [454]) for landmarks in hand_landmarks)

    if mouth_touched and not mute_toggle:
      pyautogui.press("volumemute")
      print("Volume muted.")
      mute_toggle = True
      action_text = "Mute Toggled"
    elif not mouth_touched:
      mute_toggle = False

    if left_ear_touched:
      if current_time - prev_action_time > 0.7:
        pyautogui.press("playpause")
        print("Play/Pause toggled.")
        prev_action_time = current_time
        action_text = "Play/Pause"

    if right_ear_touched:
      if first_right_tap_time == 0:
        first_right_tap_time = current_time
      elif current_time - first_right_tap_time < 0.4:
        pyautogui.press("prevtrack")
        print("Previous track.")
        first_right_tap_time = 0
        prev_action_time = current_time
        action_text = "Previous Track"
      else:
        first_right_tap_time = current_time

    if left_ear_touched:
      if first_left_tap_time == 0:
        first_left_tap_time = current_time
      elif current_time - first_left_tap_time < 0.4:
        pyautogui.press("nexttrack")
        print("Next track.")
        first_left_tap_time = 0
        prev_action_time = current_time
        action_text = "Next Track"
      else:
        first_left_tap_time = current_time

  # Draw Volume Access Status
  access_text = "Volume Access: ON" if volume_control_enabled else "Volume Access: OFF"
  access_color = (0, 255, 0) if volume_control_enabled else (0, 0, 255)
  cv2.putText(frame, access_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, access_color, 2)

  # Display detected action on screen
  display_action(frame, action_text)
  
  cv2.imshow("Haptic-Free Music Player with Hand Gesture Recognition", frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()