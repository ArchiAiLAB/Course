from ultralytics import YOLO
model = YOLO("user_best.pt")  
results = model(
    source = r'C:\Users\User\Documents\...\test\images',
    imgsz = 640,
    save = True,
    save_txt = True    
    )  