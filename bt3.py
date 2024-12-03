import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import dlib  # Import dlib for object tracking

# Check if GPU is available and being used
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load YOLOv8 model (pre-trained) for object detection
model = YOLO('yolov8n.pt')  # Use 'yolov8s.pt' or 'yolov8m.pt' for more accuracy
frame_step = 10

# Open video capture (use 0 for webcam or provide a video file path)
cap = cv2.VideoCapture('sv2.mp4')
paused = False
playback_speed = 1.0  # Default playback speed (1.0 = normal speed)
frame_delay = int(1000 / cap.get(cv2.CAP_PROP_FPS) * playback_speed)

# Create dlib's correlation tracker
tracker = dlib.correlation_tracker()

# Define function to draw bounding boxes and track objects
def draw_boxes_and_alert(frame, detections):
    height, width, _ = frame.shape
    boxes = []

    # Define the hood rectangle (This is for the front hood of the car)
    hood_top = height - 170  # Top position, making it taller
    hood_bottom = height - 0  # Bottom position, keeping it large vertically
    hood_left = width // 6  # Move left further (1/6 of width)
    hood_right = 5 * width // 6  # Move right further (5/6 of width), to make it much wider

    # Define extra proximity boundary around the hood (to detect objects that come too close)
    proximity_top = hood_top - 170  # Adding some distance above the hood
    proximity_bottom = hood_bottom + 50  # Adding some distance below the hood
    proximity_left = hood_left - 150  # Adding some distance to the left
    proximity_right = hood_right + 150  # Adding some distance to the right

    # Draw the hood rectangle (this will be ignored during object detection)
    cv2.rectangle(frame, (hood_left, hood_top), (hood_right, hood_bottom), (0, 0, 255), 3)
    cv2.putText(frame, "Ego Vehicle Hood", (hood_left, hood_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # Draw the proximity boundary (extra area around the hood for "too close" detection)
    cv2.rectangle(frame, (proximity_left, proximity_top), (proximity_right, proximity_bottom), (0, 255, 255), 2)

    for detection in detections:
        for i in range(len(detection.boxes)):
            # Extract bounding box coordinates, confidence, and class label
            x1, y1, x2, y2 = map(int, detection.boxes[i].xyxy[0])
            conf = detection.boxes[i].conf[0]
            label = detection.names[int(detection.boxes[i].cls[0])]

            # Skip if the object is inside the hood rectangle
            if x1 > hood_left and y1 > hood_top and x2 < hood_right and y2 < hood_bottom:
                continue  # Skip this object if it is inside the hood

            # Check if the object is within the proximity boundary
            too_close = False
            if proximity_left <= x1 <= proximity_right and proximity_top <= y1 <= proximity_bottom:
                too_close = True
                # Determine left or right side proximity
                if x1 < width // 2:
                    side = "LEFT"
                else:
                    side = "RIGHT"
                # Flash the "LEFT" or "RIGHT" text on the screen
                cv2.putText(frame, f"{side}", (width // 2 - 100, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                color = (0, 0, 255)  # Red for objects too close
            else:
                color = (0, 255, 0)  # Green for valid objects

            # Draw bounding box and label for this object
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label}: {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Initialize the tracker for this object (only for the first detection)
            if len(boxes) == 0:
                tracker.start_track(frame, dlib.rectangle(x1, y1, x2, y2))
                boxes.append((x1, y1, x2, y2))

    # Update tracker for the current frame
    if len(boxes) > 0:
        tracker.update(frame)
        tracked_position = tracker.get_position()
        cv2.rectangle(frame, (int(tracked_position.left()), int(tracked_position.top())),
                      (int(tracked_position.right()), int(tracked_position.bottom())), (255, 0, 0), 2)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame
    results = model(frame, device='mps')  # Use Metal Performance Shaders (MPS) for Apple GPU

    # Draw bounding boxes and labels for all detected objects
    draw_boxes_and_alert(frame, results)

    # Display the frame with detections and tracking
    cv2.imshow('Car Object Detection with Tracking', frame)

    # Break loop if 'q' is pressed
    key = cv2.waitKey(frame_delay) & 0xFF
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if key == ord('q'):  # Quit playback
        break
    elif key == ord(' '):  # Pause/Resume playback
        paused = not paused
    elif key == ord('r'):  # Rewind video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    elif key == ord('+'):  # Increase playback speed
        playback_speed = max(0.1, playback_speed - 0.1)  # Decrease frame delay
        frame_delay = int(1000 / cap.get(cv2.CAP_PROP_FPS) * playback_speed)
    elif key == ord('-'):  # Decrease playback speed
        playback_speed = playback_speed + 0.1  # Increase frame delay
        frame_delay = int(1000 / cap.get(cv2.CAP_PROP_FPS) * playback_speed)
    elif key == ord('f'):  # Step forward
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(current_frame + frame_step, total_frames - 1))
    elif key == ord('b'):  # Step backward
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(current_frame - frame_step, 0))

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
