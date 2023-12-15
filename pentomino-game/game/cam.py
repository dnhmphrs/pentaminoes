import cv2
import numpy as np

width = 1920
height = 1080

cam = cv2.VideoCapture(4)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Check if autofocus is supported
if cam.isOpened():
    # Try to enable autofocus
    autofocus_enabled = cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    if autofocus_enabled:
        print("Autofocus enabled")
    else:
        print("Autofocus not supported or could not be enabled")

while True:
    ret, frame = cam.read()

    if ret:

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Create a sharpening kernel
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])

        # Apply the kernel to the image
        sharpened = cv2.filter2D(frame, -1, kernel)

        # Convert image to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # Split the LAB image into L, A, and B channels
        l, a, b = cv2.split(lab)

        # Apply contrast adjustment to the L channel
        l_adjusted = cv2.equalizeHist(l)

        # Merge the adjusted L channel with the original A and B channels
        lab_adjusted = cv2.merge((l_adjusted, a, b))

        # Convert the LAB image back to BGR color space
        adjusted_image = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)

        # Display the original and blurred images
        cv2.imshow('Original', frame)
        cv2.imshow('Sharp', sharpened)
        cv2.imshow('Contrast', adjusted_image)

    if cv2.waitKey(1) == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
