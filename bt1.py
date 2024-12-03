import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load the class names from coco.names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture
cap = cv2.VideoCapture(r'/Users/mayankrajanand/sv2.mp4')  # Replace with your video file
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define exclusion region (bottom 15% of the frame)
exclusion_region = int(frame_height * 0.85)

# Playback control variables
paused = False
frame_position = 0

while True:
    if not paused or frame_position != 0:
        ret, frame = cap.read()
        if not ret:
            break
        frame_position = 0

        # Prepare the frame for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Initialize lists to hold detection details
        class_ids, confidences, boxes = [], [], []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Detection threshold
                    # Scale bounding box to frame size
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    w = int(detection[2] * frame_width)
                    h = int(detection[3] * frame_height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Append details if not in exclusion region
                    if y + h <= exclusion_region:
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

        # Non-max suppression to reduce overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                confidence = confidences[i]
                class_id = class_ids[i]

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Alert if a vehicle is too close
                if classes[class_id] in ["car", "truck", "bus"] and h > frame_height * 0.5:
                    cv2.putText(frame, "VEHICLE TOO CLOSE!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Show the frame
    cv2.imshow("Crash Detection System", frame)

    # Handle key presses for playback control
    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):  # Quit
        break
    elif key == ord("p"):  # Pause or play
        paused = not paused
    elif key == ord("f"):  # Forward by 5 seconds
        frame_position += fps * 5
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + fps * 5)
    elif key == ord("b"):  # Backward by 5 seconds
        frame_position -= fps * 5
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cap.get(cv2.CAP_PROP_POS_FRAMES) - fps * 5))

cap.release()
cv2.destroyAllWindows()
