import cv2
import numpy as np
import pygame
import dlib
from scipy.spatial import distance as dist
from ultralytics import YOLO
import tensorflow as tf

# Initialize pygame for sound
pygame.mixer.init()
pygame.mixer.music.load("alert_sound.mp3")  # Replace with your sound file path

# Eye Aspect Ratio Calculation (Drowsiness Detection)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Drowsiness detection constants
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 7
COUNTER = 0
ALERT_ON = False

# Load dlib face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# Load YOLOv8 model (pre-trained)
model = YOLO('yolov8n.pt')  # Use 'yolov8s.pt' for higher accuracy, 'yolov8n.pt' is lighter

# Open video capture for object detection (use a video file for car detection)
video_cap = cv2.VideoCapture('sv3.mp4')

# Open webcam for drowsiness detection (use 0 for webcam)
webcam_cap = cv2.VideoCapture(0)

# Create dlib's correlation tracker
tracker = dlib.correlation_tracker()

# Function to draw bounding boxes and alert for objects (vehicles)
def draw_boxes_and_alert(frame, detections):
    height, width, _ = frame.shape
    boxes = []

    # Define the proximity area around the vehicle's hood
    hood_top = height - 170
    hood_bottom = height - 0
    hood_left = width // 6
    hood_right = 5 * width // 6
    proximity_top = hood_top - 170
    proximity_bottom = hood_bottom + 50
    proximity_left = hood_left - 150
    proximity_right = hood_right + 150

    # Draw the hood and proximity area rectangles
    cv2.rectangle(frame, (hood_left, hood_top), (hood_right, hood_bottom), (0, 0, 255), 3)
    cv2.putText(frame, "Ego Vehicle Hood", (hood_left, hood_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.rectangle(frame, (proximity_left, proximity_top), (proximity_right, proximity_bottom), (0, 255, 255), 2)

    for detection in detections:
        for i in range(len(detection.boxes)):
            # Extract bounding box, confidence, and class
            x1, y1, x2, y2 = map(int, detection.boxes[i].xyxy[0])
            conf = detection.boxes[i].conf[0]
            label = detection.names[int(detection.boxes[i].cls[0])]

            if x1 > hood_left and y1 > hood_top and x2 < hood_right and y2 < hood_bottom:
                continue

            # Check if object is too close to the vehicle (within proximity area)
            too_close = False
            if proximity_left <= x1 <= proximity_right and proximity_top <= y1 <= proximity_bottom:
                too_close = True
                side = "LEFT" if x1 < width // 2 else "RIGHT"
                cv2.putText(frame, f"{side}", (width // 2 - 100, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                color = (0, 0, 255)  # Red for "too close"
            else:
                color = (0, 255, 0)  # Green for valid objects

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label}: {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if len(boxes) == 0:
                tracker.start_track(frame, dlib.rectangle(x1, y1, x2, y2))
                boxes.append((x1, y1, x2, y2))

    if len(boxes) > 0:
        tracker.update(frame)
        tracked_position = tracker.get_position()
        cv2.rectangle(frame, (int(tracked_position.left()), int(tracked_position.top())),
                      (int(tracked_position.right()), int(tracked_position.bottom())), (255, 0, 0), 2)

# Start processing video frames
while video_cap.isOpened() and webcam_cap.isOpened():
    ret_video, frame_video = video_cap.read()
    ret_webcam, frame_webcam = webcam_cap.read()
    
    if not ret_video or not ret_webcam:
        break

    # Perform YOLO object detection on the video frame
    results = model(frame_video, device='mps')  # Use Metal Performance Shaders for Apple GPU

    # Draw bounding boxes and check proximity for detected objects
    draw_boxes_and_alert(frame_video, results)

    # Convert webcam frame to grayscale for drowsiness detection
    gray = cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Drowsiness detection
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame_webcam, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not ALERT_ON:
                    pygame.mixer.music.play(-1)  # Loop alert sound
                    ALERT_ON = True
        else:
            COUNTER = 0
            if ALERT_ON:
                pygame.mixer.music.stop()  # Stop sound
                ALERT_ON = False

    # Display both frames with detections and alerts
    cv2.imshow("Car Object Detection (Video)", frame_video)
    cv2.imshow("Webcam Drowsiness Detection", frame_webcam)

    # Handle key presses for playback control
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):  # Quit
        break

# Release resources and close windows
video_cap.release()
webcam_cap.release()
cv2.destroyAllWindows()
