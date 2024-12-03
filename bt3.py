import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

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


# Define function to draw bounding boxes on frames
def draw_boxes_and_alert(frame, detections):
    height, width, _ = frame.shape
    alert_distance_horizontal = width * 0.01  # Horizontal proximity threshold (e.g., left or right)
    alert_distance_vertical = height * 0.3  # Vertical proximity threshold (e.g., front)

    for detection in detections:
        for i in range(len(detection.boxes)):
            # Extract bounding box coordinates, confidence, and class label
            x1, y1, x2, y2 = map(int, detection.boxes[i].xyxy[0])
            conf = detection.boxes[i].conf[0]
            label = detection.names[int(detection.boxes[i].cls[0])]

            box_height = y2 - y1

            # Calculate bounding box center and size
            box_center_x = (x1 + x2) // 2
            box_center_y = (y1 + y2) // 2
            box_width = x2 - x1
            box_height = y2 - y1

            if (
                box_center_y > height * 0.7  # Near bottom
                and abs(box_center_x - width // 2) < width * 0.3  # Near center horizontally
                and (x2 - x1) * (y2 - y1) > (width * height) * 0.2  # Large area
            ):
                continue 

            # Check if object is "too close" horizontally or vertically
            too_close_horizontal = (
                box_center_x < alert_distance_horizontal or box_center_x > width - alert_distance_horizontal
            )
            too_close_vertical = box_center_y > height - alert_distance_vertical

            # Trigger alert if object is too close in either dimension
            too_close = too_close_horizontal or too_close_vertical

            # Draw bounding box with red if too close, green otherwise
            color = (0, 0, 255) if too_close else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Add label, confidence score, and alert if too close
            text = f"{label}: {conf:.2f}"
            if too_close:
                print(f"ALERT: {label} is too close! Position: ({box_center_x}, {box_center_y})")
                text += " (TOO CLOSE!)"
                cv2.putText(frame, "WARNING: Object Too Close!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


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

    # Display the frame with detections
    cv2.imshow('Car Object Detection', frame)

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
