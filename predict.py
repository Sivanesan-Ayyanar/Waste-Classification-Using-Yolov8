from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model with the best weights
model = YOLO(r'C:\Users\vs632\PycharmProjects\Metal_Or_Plastic_Detection_Using_Yolov8\runs\classify\train2\weights\best.pt')

# Function to classify and display the image with results
def classify_image(image_path):
    # Load the image using PIL
    img = Image.open(image_path).convert("RGB")

    # Convert the image to a numpy array for prediction
    img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Make prediction
    results = model.predict(img_array)

    # Extract the first result (since results is a list)
    result = results[0]

    # Extract probabilities and class names
    probs = result.probs
    class_names = result.names

    # Determine the predicted class
    predicted_class = class_names[probs.top1]
    confidence = probs.top1conf.item()

    # Display the results
    print(f"Predicted Class: {predicted_class} with confidence {confidence:.2f}")

    # Display the image with Matplotlib
    plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    plt.title(f"{predicted_class}: {confidence:.2f}")
    plt.axis('off')
    plt.show()

# Example usage
classify_image(r'C:\Users\vs632\OneDrive\Desktop\SEM 6\recycle-aluminum-metal-crushed-can-waste-background-photo.jpg')


