'''python callibrate.py -v arucoCallibrate.mp4 -c 80'''

import numpy as np
import cv2
from cv2 import aruco
import argparse

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to the calibration video")
ap.add_argument("-c", "--captures", required=True, type=int, help="minimum number of valid captures required")
args = vars(ap.parse_args())

# ArUco dictionary
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_100)

# Initialize video capture
cap = cv2.VideoCapture(args["video"])

# Store detected corners and their IDs
all_corners = []  # List of all corners detected in all frames
all_ids = []      # List of all ids detected in all frames

valid_captures = 0
image_size = None

while valid_captures < args["captures"] and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(gray, ARUCO_DICT)

    if ids is not None:
        all_corners.append(corners)
        all_ids.append(ids)
        valid_captures += 1

        # Display the detected markers
        aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.imshow('Detected ArUco markers', frame)
        if cv2.waitKey(0) == ord('q'):
            break

    #set the image size        
    if image_size is None:
        image_size = gray.shape[::-1]

cv2.destroyAllWindows()
cap.release()

if valid_captures < args["captures"]:
    print("Calibration was unsuccessful. Not enough valid captures.")
    exit()

# Prepare data for calibration
object_points = []
# all markers are the same size and we know their real world dimensions
marker_length = 0.048  # marker length of 4.8 cm

#3D coordinates of the corners of an ArUco marker. 
object_marker_points = np.array([
    [0, 0, 0],
    [marker_length, 0, 0],
    [marker_length, marker_length, 0],
    [0, marker_length, 0]
], dtype=np.float32)

# iterate simultaneously over two lists: all_corners and all_ids to present each row of the array in 3D coordinates of the corners 
for corners, ids in zip(all_corners, all_ids):
    # repeating it as many times as there are IDs detected in the frame.
    objp = np.array([object_marker_points for _ in range(len(ids))], dtype=np.float32)
    object_points.append(objp)

# Flatten the list of corners and object points
all_corners = [corner for sublist in all_corners for corner in sublist]
all_ids = [id_ for sublist in all_ids for id_ in sublist]
object_points = [op for sublist in object_points for op in sublist]

print("Starting Calibration")
# Camera calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objectPoints=object_points,
    imagePoints=all_corners,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None)

print("Calibration successful.")
print("Camera intrinsic parameters matrix:\n{}".format(camera_matrix))
print("Camera distortion coefficients:\n{}".format(dist_coeffs))

# Save calibration results
np.save("K_matrix", camera_matrix)
np.save("distortion", dist_coeffs)

# np.save("a", camera_matrix)
# np.save("b", dist_coeffs)
