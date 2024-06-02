import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
import json

imagePath = r"dataset\roger_federer\5c6becd02628985d2a2ee2a2.jfif"  # Path to your image
model_path = r"sports_celebrity_classification.pkl"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load the model (assuming scikit-learn is upgraded)
model = joblib.load(model_path)

with open("class_dictionary.json") as f:
    class_dict=json.load(f)

class_dict={v:k for (k,v) in class_dict.items()}

def get_cropped_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) >= 2:
            return roi_color

    return None

# Get cropped image
croppedImg = get_cropped_image(imagePath)

if croppedImg is not None: 
    # Resize the image to 90x90 pixels
    fixed_size = (90, 90)
    resized_img = cv2.resize(croppedImg, fixed_size)

    # Convert to grayscale if needed
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # Flatten the image to 8100 features
    flattened_img = gray.flatten()

    # Predict using the model
    prediction = model.predict([flattened_img])
    print("Celibrity :", class_dict[prediction[0]])


else:
    print("No face detected in the image or the image is invalid.")
