import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import mediapipe as mp

# Load and preprocess data
def load_data(data_path, csv_path):
    data = []
    labels = []
    df = pd.read_csv(csv_path)
    
    for set_id in range(len(df)):
        folder_path = os.path.join(data_path, f"set_{set_id}")
        for emotion in ['angry', 'happy', 'neutral', 'sad']:
            img_path = os.path.join(folder_path, f"{emotion}.jpg")
            print(f"Checking if image exists at path: {img_path}")
            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (48, 48))
                data.append(img)
                labels.append(emotion)
            else:
                print(f"Image {img_path} does not exist.")
    
    if not data:
        print("No data found. Please check your dataset paths and files.")
        exit()
    
    return np.array(data), np.array(labels)

# Prepare the model
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )
    return history

# Live emotion detection
def detect_emotion(frame, face_detection, model, emotion_labels):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            
            face = frame[y:y+h, x:x+w]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48))
            face = np.expand_dims(face, axis=[0, -1])
            
            emotion_pred = model.predict(face)
            emotion_label = emotion_labels[np.argmax(emotion_pred)]
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion_label, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame

# Main function
def main():
    # Load and preprocess data
    data_path = "C:/Users/hp/Desktop/FRED/images"
    csv_path = "C:/Users/hp/Desktop/FRED/emotions.csv"
    X, y = load_data(data_path, csv_path)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = to_categorical(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Normalize data
    X_train = X_train.astype('float32') / 255
    X_val = X_val.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    
    # Reshape data for CNN
    X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
    X_val = X_val.reshape(X_val.shape[0], 48, 48, 1)
    X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
    
    # Create and train model
    model = create_model((48, 48, 1), len(le.classes_))
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy}")
    
    # Live detection
    mp_face_detection = mp.solutions.face_detection
    cap = cv2.VideoCapture(0)
    
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            frame = detect_emotion(frame, face_detection, model, le.classes_)
            cv2.imshow('Emotion Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
#training and testing and live emotion detection 