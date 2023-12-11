import cv2
import cv2.aruco as aruco

# Define ArUco marker dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)
# Define camera intrinsic parameters (obtained through calibration)
# camera_matrix = # Your camera matrix
# dist_coeffs = # Your distortion coefficients

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use the appropriate camera index or video file path
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
while True:
    # Read frame from video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for marker detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

    if ids is not None:
        # # Estimate marker poses
        # rvecs, tvecs, _ = detector.estimatePose(corners)

        # Draw axis and markers on the frame
        aruco.drawDetectedMarkers(frame, corners, ids)
        aruco.drawDetectedMarkers(frame, rejectedImgPoints, borderColor=(100, 0, 240))
        # for i in range(len(ids)):
        #     detector.drawAxis(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_length)

    # Display the frame
    cv2.imshow("AR Marker Tracking", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
