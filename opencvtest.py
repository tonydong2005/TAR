import cv2
import numpy as np

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert the frame to HSV (Hue, Saturation, Value)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Narrower lower red range in HSV
    lower_red1 = np.array([0, 150, 150])
    upper_red1 = np.array([5, 255, 255])

    # Narrower upper red range in HSV
    lower_red2 = np.array([175, 150, 150])
    upper_red2 = np.array([180, 255, 255])
    
    # Create a mask for the red color
    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    # Create other mask
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)

    # Combine the two masks
    red_mask = mask1 + mask2

    # Find contours in the red mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding box around each contour
    for contour in contours:
        if cv2.contourArea(contour) > 2000:  # Filter by area to avoid noise
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Red Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()