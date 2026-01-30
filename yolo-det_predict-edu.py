from ultralytics import YOLO
model = YOLO("yolov8n.pt") 
results = model(
    source = r'C:\Users\User\Documents\...\coco128\images\train2017',
    imgsz = 224,
    save = True,
    save_txt = True    
    )  