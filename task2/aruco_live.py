import cv2
import numpy as np

# --------- CAMERA PARAMETERS ---------
K = np.array([
    [2253.8865,    0.0,    705.2908],
    [0.0,      2271.9448, 1162.9481],
    [0.0,         0.0,       1.0]
], dtype=np.float32)

distortion = np.array([-0.02340005, 0.44409153, 0.02065355, -0.0144623, -1.32082603], dtype=np.float32)

marker_length = 0.05  # marker size in meters (5 cm)

# --------- STREAM SETUP ---------
phone_ip = '192.168.1.152'  # Replace with your phone's IP address
port = 4747                # Common default for IP Webcam apps
stream_url = f"http://{phone_ip}:{port}/video"

cap = cv2.VideoCapture(stream_url)

# --------- ARUCO SETUP ---------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Generate a marker to display
marker_id = 0
marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 400)

cv2.imshow('Show this Marker', marker_img)
cv2.waitKey(1)

# --------- MAIN LOOP ---------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from phone camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimate pose of each marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, K, distortion)

        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(frame, K, distortion, rvec, tvec, 0.03)

            # Print pose
            R, _ = cv2.Rodrigues(rvec)
            print("\nRotation Matrix:\n", R)
            print("Translation Vector (meters):\n", tvec.ravel())

    cv2.imshow('Phone Camera Pose Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --------- CLEANUP ---------
cap.release()
cv2.destroyAllWindows()
