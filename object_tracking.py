import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Webcam
cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    # Object detection + tracking
    results = model.track(
        frame,
        persist=True
    )

    # Draw results automatically
    annotated_frame = results[0].plot()

    cv2.imshow(
        "Object Detection and Tracking",
        annotated_frame
    )

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()