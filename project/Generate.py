import cv2
from cv2 import aruco

def generate_aruco_markers():
    # Parameters
    aruco_dict_type = aruco.DICT_5X5_100
    marker_size = 200  # size of marker images in pixels
    aruco_dict = aruco.Dictionary_get(aruco_dict_type)

    # Generate and save each marker
    for i in range(100):  # there are 100 markers in DICT_5X5_100
        # Create an image for the marker
        marker_image = aruco.drawMarker(aruco_dict, i, marker_size)
        # Save the image
        cv2.imwrite(f'Aruco/aruco_marker_5x5_100_id_{i}.png', marker_image)

generate_aruco_markers()
