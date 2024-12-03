import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # Make sure to use the correct YOLO model files
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Enable GPU acceleration (Metal/OpenCL for macOS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)  # Metal support for macOS M1/M2
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)     # OpenCL for GPU acceleration

# Load video or webcam feed
cap = cv2.VideoCapture(r'/Users/mayankrajanand/svideo.mp4')  # Replace with "camera" for webcam input
fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video

# Initial pause state
paused = False

# Thresholds for alert (this will depend on your scenario and calibration)
MIN_VEHICLE_SIZE = 5000  # Adjust this based on experimentation or camera calibration

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

    # Resize image (optional for faster inference)
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Post-processing the YOLO output and drawing bounding boxes
    class_ids = []
    confidences = []
    boxes = []
    height, width, channels = frame.shape
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression to reduce overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes on the detected objects and check if vehicles are too close
    close_alert = False
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(class_ids[i])
            confidence = str(round(confidences[i], 2))
            area = w * h  # Calculate the area of the bounding box

            # Check if the detected object is a vehicle (class_id for vehicle may need to be checked)
            if area > MIN_VEHICLE_SIZE:
                close_alert = True  # Alert if vehicle size exceeds threshold

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show an alert if vehicles are too close
    if close_alert:
        cv2.putText(frame, "Alert: Vehicles Too Close!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the processed frame
    cv2.imshow("Frame", frame)

    # Handle key press events
    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):  # Pause playback
        paused = not paused  # Toggle pause
    elif key == ord('f'):  # Forward playback
        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        new_pos = current_pos + int(fps * 5)  # Forward 5 seconds
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
    elif key == ord('b'):  # Backward playback
        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        new_pos = max(current_pos - int(fps * 5), 0)  # Backward 5 seconds, but not below 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
    elif key == 27:  # 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
