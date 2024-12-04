import cv2
import dlib
from scipy.spatial import distance as dist
import numpy as np
import pygame  # Import pygame for sound

# Initialize pygame and load the sound
pygame.mixer.init()
pygame.mixer.music.load("alert_sound.mp3")  # Replace with the path to your sound file

def eye_aspect_ratio(eye):
    # Compute the distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Compute the distance between the horizontal eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Thresholds for EAR and frames to consider drowsiness
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 9

# Initialize frame counter and drowsiness alert
COUNTER = 0
ALERT_ON = False  # Flag to check if alert is already playing

# Load the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Grab the indexes of the facial landmarks for the left and right eye, respectively
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Average the EAR for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # Visualize the eye landmarks
        #cv2.polylines(frame, [cv2.convexHull(np.array(leftEye))], True, (0, 255, 0), 1)
        #cv2.polylines(frame, [cv2.convexHull(np.array(rightEye))], True, (0, 255, 0), 1)

        # Check if EAR is below threshold
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            # If eyes are closed for a sufficient number of frames, trigger drowsiness alert
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Play alert sound if not already playing
                if not ALERT_ON:
                    pygame.mixer.music.play(-1)  # Loop the alert sound
                    ALERT_ON = True
        else:
            COUNTER = 0
            if ALERT_ON:
                pygame.mixer.music.stop()  # Stop the alert sound
                ALERT_ON = False

    cv2.imshow("Frame", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
