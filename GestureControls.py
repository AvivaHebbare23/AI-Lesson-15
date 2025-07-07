import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to captre image.")
        break

    #convert hsv for colour filtering
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 500: #ignore small contours
            #draw box around detected hand
            x, y, w, h = cv2.boundingRect(max_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #centre of hand for further tracking
            centre_x = int(x + w / 2)
            centre_y = int(y + h / 2)
            cv2.circle(frame, (centre_x, centre_y), 5, (0, 0, 255), -1) #red dot in centre

    cv2.imshow("Original Frame", frame)
    cv2.imshow("Filtered Frame", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()