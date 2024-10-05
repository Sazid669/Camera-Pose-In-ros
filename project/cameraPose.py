'''
Sample Usage:-
python cameraPose.py --K_Matrix K_matrix.npy --D_Coeff distortion.npy --type DICT_5X5_100
'''
import cv2
import numpy as np
import argparse
import sys
from utils import ARUCO_DICT
import networkx as nx
import matplotlib.pyplot as plt

# Class to hold marker information
class MarkerNode:
    def __init__(self, marker_id, rvec, tvec, corners):
        self.marker_id = marker_id  # ID of the ArUco marker
        self.rvec = rvec           # Rotation vector for this marker
        self.tvec = tvec           # Translation vector for this marker
        self.corners = corners

# Function to invert rotation and translation vectors
def invert_pose(rvecs, tvecs):
    # Convert rotation vector to a rotation matrix
    rmat, _ = cv2.Rodrigues(rvecs) 
    # Invert the rotation matrix
    rmat_inv = np.linalg.inv(rmat)
    # Reshape tvecs to ensure it is a three-element vector (3, 1)
    tvecs = tvecs.reshape((3, 1))
    # Calculate the inverse translation vector
    tvec_inv = -np.dot(rmat_inv, tvecs)
    # Convert the inverted rotation matrix back to a rotation vector
    rvec_inv, _ = cv2.Rodrigues(rmat_inv)
    return rvec_inv, tvec_inv

# Function to estimate pose of ArUco markers in a frame
def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    # Convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Get the ArUco dictionary
    aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    # Create default parameters for detector
    parameters = cv2.aruco.DetectorParameters_create()
    # Detect markers
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters,
        cameraMatrix=matrix_coefficients, distCoeff=distortion_coefficients)
    marker_poses = []
    # If markers are detected
    if ids is not None and len(corners) > 0:
        # Process each marker
        for i, corner in enumerate(corners):
            # Estimate pose of each marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corner, 0.048, matrix_coefficients, distortion_coefficients)
            # Draw the detected markers
            cv2.aruco.drawDetectedMarkers(frame, corners)
            # Draw axis for each marker
            cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
            # Append the pose data to the list
            marker_poses.append(MarkerNode(ids[i][0], rvec, tvec,corner))
    return frame, marker_poses

# Main function
def main():
    # Argument parser for command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUco tag to detect")
    args = vars(ap.parse_args())

    # Check if the specified ArUco dictionary type is supported
    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUco tag type '{args['type']}' is not supported")
        sys.exit(0)

    # Load camera calibration parameters
    k = np.load(args["K_Matrix"])
    d = np.load(args["D_Coeff"])
    aruco_dict_type = ARUCO_DICT[args["type"]]
    # Open video file
    video = cv2.VideoCapture('pose.MOV')
    
    # Create a graph to store marker relationships
    graph = nx.Graph()

    # Initialize accumulators for averaging camera pose
    cumulative_rvec = np.zeros((3, 1))
    cumulative_tvec = np.zeros((3, 1))
    count = 0

    # Process video frames
    while True:
        ret, frame = video.read()
        if not ret:
            break

        output, marker_poses = pose_estimation(frame, aruco_dict_type, k, d)
        for node in marker_poses:
            '''Adds each detected marker as a node in a graph (graph). The node is identified by node.marker_id, and stores the marker's rotation and translation vectors as attributes. '''
            graph.add_node(node.marker_id, rvec=node.rvec, tvec=node.tvec) 
            # Invert the rotation and translation vectors to calculate the average camera pose
            rvec_inv, tvec_inv = invert_pose(node.rvec, node.tvec)
            # summing the inverted vector to calculate the average camera pose
            cumulative_rvec += rvec_inv
            cumulative_tvec += tvec_inv
            count += 1
            # Compute average camera pose from accumulated poses
            if count > 1:
                average_rvec = cumulative_rvec / count
                average_tvec = cumulative_tvec / count
                print("Real time Camera Pose (Rotation Vector):", average_rvec)
                print("Real time Camera Pose (Translation Vector):", average_tvec)

        # Add edges based on the Euclidean distance between markers
        for i in range(len(marker_poses)):
            for j in range(i + 1, len(marker_poses)):
                dist = np.linalg.norm(marker_poses[i].tvec - marker_poses[j].tvec)
                dist = round(dist,2)
                if dist < 0.5:  # You can adjust this threshold as needed
                    graph.add_edge(marker_poses[i].marker_id, marker_poses[j].marker_id, weight=dist)
        

        cv2.imshow('Estimated Pose', output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if count > 1:
        average_rvec = cumulative_rvec / count
        average_tvec = cumulative_tvec / count
        print("Average Camera Pose (Rotation Vector):", average_rvec)
        print("Average Camera Pose (Translation Vector):", average_tvec)

    video.release()
    cv2.destroyAllWindows()
    draw_graph(graph)
    

def draw_graph(graph):
    # Position nodes using the Spring layout
    pos = nx.spring_layout(graph)

    # Draw the graph using the positions calculated
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', edge_color='red', node_size=700, font_size=10, font_weight='bold')

    # Draw edge labels to show weights (distances)
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    plt.title('Aruco Markers Graph')
    plt.show()

if __name__ == '__main__':
    main()
