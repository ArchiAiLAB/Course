from ultralytics import YOLO
if __name__ == "__main__":
    model = YOLO("yolov8n-cls.pt") 
    results = model.train(data=r'C:\Users\User\Documents\...\training_set', 
                        epochs=5, 
                        imgsz=224)