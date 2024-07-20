from ultralytics import YOLO

# Initialize the model
model = YOLO('yolov8n-cls.pt')

# Train the model
results = model.train(data=r'C:\Users\vs632\OneDrive\Desktop\Dataset_For_Waste_Classification', epochs=10, imgsz=64)

# Optionally, print the results
print(results)
