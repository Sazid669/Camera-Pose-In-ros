import cv2
from cv2 import aruco
import numpy as np

def create_charuco_board():
    # Parameters for the Charuco board
    CHARUCOBOARD_ROWCOUNT = 7
    CHARUCOBOARD_COLCOUNT = 5
    ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_100)
    square_length = 200  # in pixels
    marker_length = 150  # in pixels, must be less than square_length

    # Create the CharucoBoard object
    charuco_board = aruco.CharucoBoard_create(
        squaresX=CHARUCOBOARD_COLCOUNT,
        squaresY=CHARUCOBOARD_ROWCOUNT,
        squareLength=square_length,
        markerLength=marker_length,
        dictionary=ARUCO_DICT)
    
    # Create the board image
    board_image = charuco_board.draw((CHARUCOBOARD_COLCOUNT * square_length, CHARUCOBOARD_ROWCOUNT * square_length))

    # Save the image or show it
    cv2.imwrite('charuco_board_5x5_100.png', board_image)
    cv2.imshow('Charuco Board', board_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function to create and display the board
create_charuco_board()
