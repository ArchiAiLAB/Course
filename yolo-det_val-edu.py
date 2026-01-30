import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_WARNINGS"] = "FALSE"  
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

def main():
    model = YOLO(r"C:\Users\User\Documents\...\Course\ch9-train_\weights\best.pt") 
    metrics = model.val()  
    metrics.box.map  
    metrics.box.map50  
    metrics.box.map75  
    metrics.box.maps  

if __name__ == '__main__':
    # This is the "main guard"
    from multiprocessing import freeze_support
    freeze_support()
    main()