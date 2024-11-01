import cv2
import datetime
from plyer import notification
from ultralytics import YOLO

model = YOLO('best.pt')
webcamera = cv2.VideoCapture(0)
hasDog = False
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
    
    results = model.track(frame, classes=0, conf=0.5, imgsz=480)
    detected = results[0].boxes

    if detected and not hasDog:
        currentTime = datetime.datetime.now()

        # first time being seen on camera
        if (not capturedBefore):
            sendNotification()
            capturedBefore = True
            hasDog = True

        # not the first time
        if (capturedBefore and not hasDog):
            # Find the time gap
            diff = (currentTime - timestamp).total_seconds()
            if (diff > 3):
                sendNotification()
                hasDog = True
    
    if not detected:
        if hasDog:
            timestamp = datetime.datetime.now()
        hasDog = False

    cv2.imshow("Live Camera", results[0].plot())

    if cv2.waitKey(1) == ord('q'):
        break

webcamera.release()
cv2.destroyAllWindows()
