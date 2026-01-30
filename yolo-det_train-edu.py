import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_WARNINGS"] = "FALSE"  
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt") 
    path = r'C:\Users\User\Documents\...\dataset.yaml'
    results = model.train(data= path,
                          project= "Course",
                          name="ch9-train_",
                          epochs=5, 
                          imgsz=640)
    
if __name__ == '__main__':
    # This is the "main guard"
    from multiprocessing import freeze_support
    freeze_support()
    main()