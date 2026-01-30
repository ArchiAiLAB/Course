from ultralytics import YOLO
model = YOLO(r"C:\Users\User\Documents\...\weights\best.pt") 
results = model(
    source = r'C:\Users\User\Documents\...\test_set\dogs',
    imgsz = 224,
    save = True)  