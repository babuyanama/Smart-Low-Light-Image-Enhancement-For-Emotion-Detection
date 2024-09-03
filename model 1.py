import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and find faces
        results = face_detection.process(rgb_frame)

        # Convert back to BGR for display
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Draw rectangle around the face
                cv2.rectangle(frame, bbox, (0, 255, 0), 2)

                # Analyze emotions
                try:
                    face_img = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                    results = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                    emotion = results[0]['dominant_emotion']
                    
                    # Display emotion text
                    cv2.putText(frame, emotion, (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                except:
                    print("Error in emotion detection")

        # Display the resulting frame
        cv2.imshow('Face Recognition and Emotion Detection', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
#live emotion detection
