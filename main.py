from ultralytics import YOLO
import cv2

video = "Traffic IP Camera video.mp4"

cap = cv2.VideoCapture(video)

ret, frame = cap.read()

model = YOLO("yolov8m.pt")
names = model.names

threshold = 0.5
# mask created in Canvas
mask = cv2.imread("mask.png")

# finding shape of each frame
height, width, _ = frame.shape
# resize mask image according to frame size
mask = cv2.resize(mask, (width, height))

while ret:
    imgRegion = cv2.bitwise_and(frame, mask)
    results = model(imgRegion)[0]
    ret, frame = cap.read()
    # putting created mask (in Canvas) on each frame

    for result in results.boxes.data.tolist():
        classes_to_detect = ["truck", "car", "bus"]
        x1, y1, x2, y2, score, class_id = result
        if score > threshold and results.names[int(class_id)] in classes_to_detect:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1 + 10), int(y1 + 40)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # cv2.imshow("Video", frame)
    cv2.imshow("Masked video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
