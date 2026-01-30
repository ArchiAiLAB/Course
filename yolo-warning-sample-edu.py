import cv2
import os
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
video_path = "*.mp4"
video_name = os.path.splitext(os.path.basename(video_path))[0]
cap = cv2.VideoCapture(video_path)

zone_top_left = (600, 420)
zone_bottom_right = (700, 600)

fps = cap.get(cv2.CAP_PROP_FPS)
os.makedirs("alert_frames", exist_ok=True)
alert_flag = False

while cap.isOpened():
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    alert = False

    for box in results.boxes:
        if int(box.cls[0]) != 0:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if not (x2 < zone_top_left[0] or x1 > zone_bottom_right[0] or y2 < zone_top_left[1] or y1 > zone_bottom_right[1]):
            alert = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.rectangle(frame, zone_top_left, zone_bottom_right, (0, 255, 0), 2)

    time_sec = frame_idx / fps
    label = f" Person in zone at {time_sec:.2f}s!" if alert else "Zone clear"
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if alert else (0, 255, 0), 2)

    if alert and not alert_flag:
        fname = f"alert_frames/{video_name}_alert_{time_sec:.2f}s.jpg"
        cv2.imwrite(fname, frame)
        alert_flag = True
    elif not alert:
        alert_flag = False

    cv2.imshow("YOLO Alert Fixed", frame)
    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()
