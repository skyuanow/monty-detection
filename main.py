import cv2
import datetime
from plyer import notification
from ultralytics import YOLO

model = YOLO('best.pt')
model.overrides['verbose'] = False  # Suppress verbose output

webcamera = cv2.VideoCapture(0)
timestamp = None
capturedBefore = False

def sendNotification():
    print("DOG DETECTED")
    notification.notify(
        title="Monty is here",
        message="Do you want to let him in?",
        timeout=10
    )

while True:
    success, frame = webcamera.read()
    
    results = model.track(frame, classes=0, conf=0.6, imgsz=480)
    detected = results[0].boxes

    if detected:
        timestamp = datetime.datetime.now()

        # first time being seen on camera
        if (not capturedBefore):
            print('FIRST TIME ON CAM WOW')
            sendNotification()
            capturedBefore = True
    else:
        # not the first time
        if (capturedBefore and timestamp):
            # Find the time gap
            diff = (datetime.datetime.now() - timestamp).total_seconds()
            if (diff > 3):
                capturedBefore = False

    cv2.imshow("Live Camera", results[0].plot())

    if cv2.waitKey(1) == ord('q'):
        break

webcamera.release()
cv2.destroyAllWindows()
