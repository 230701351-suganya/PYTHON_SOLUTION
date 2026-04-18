import cv2
from ultralytics import YOLO

model = YOLO("yolov8l.pt")

cap = cv2.VideoCapture("videos/bus4.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

cv2.namedWindow("Bus Crowd Detection", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video ended or frame not read.")
        break

    height, width, _ = frame.shape

    # Keep aspect ratio
    max_width = 900
    scale = max_width / width
    new_width = int(width * scale)
    new_height = int(height * scale)

    frame = cv2.resize(frame, (new_width, new_height))

    results = model(frame, conf=0.25)

    person_count = 0

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == 0:
                person_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Person {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    if person_count <= 10:
        status = "Low Crowd"
        color = (0, 255, 0)
    elif person_count <= 20:
        status = "Medium Crowd"
        color = (0, 255, 255)
    else:
        status = "Overcrowded"
        color = (0, 0, 255)

    cv2.putText(frame, f"People Count: {person_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(frame, f"Crowd Status: {status}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    if status == "Overcrowded":
        cv2.putText(frame, "ALERT: BUS IS OVERCROWDED!", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Resize window to match frame
    cv2.resizeWindow("Bus Crowd Detection", new_width, new_height)
    cv2.imshow("Bus Crowd Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()