#!/usr/bin/env python

import rospy
import cv2
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
import cv2.aruco as aruco
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation
from utils import ARUCO_DICT
from deadReckoning_node import DeadReckoningNode
from tf.transformations import euler_from_quaternion
from marker import *
from geometry_msgs.msg import PoseStamped,Point, Quaternion
class MarkerNode:
    def __init__(self, marker_id, rvec, tvec):
        self.marker_id = marker_id
        self.rvec = rvec
        self.tvec = tvec

class ArucoDetectorNode:
    def __init__(self):
        rospy.init_node('aruco_detector')
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/turtlebot/kobuki/sensors/realsense/color/image_color", Image, self.image_callback)
        self.pose_pub = rospy.Publisher("/turtlebot/kobuki/aruco_position_perception", Float64MultiArray, queue_size=10)
        self.perception_markers_pub_aruco_marker = rospy.Publisher("aruco/visualization_aruco_perception", MarkerArray, queue_size=10)
        self.odom_sub = rospy.Subscriber("turtlebot/kobuki/odom", Odometry, self.odom_callback, queue_size=10)
        self.state = np.zeros(3)  # Initialize the state attribute with a default value

        self.aruco_dict_type = aruco.Dictionary_get(ARUCO_DICT["DICT_ARUCO_ORIGINAL"])
        self.camera_matrix = np.array([[1396.8086675255468, 0.0, 960.0],
                                       [0.0, 1396.8086675255468, 540.0],
                                       [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.marker_nodes = []
        self.marker={}
        self.marker_handler = MarkerHandler()
        # self.graph = ArucoGraph()

    def image_callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Initialize accumulators for averaging camera pose
        cumulative_rvec = np.zeros((3, 1))
        cumulative_tvec = np.zeros((3, 1))
        count = 0

        frame, marker_poses = self.pose_estimation(frame, self.aruco_dict_type, self.camera_matrix, self.dist_coeffs)
        for node in marker_poses:
            # self.graph.update_graph(node, marker_poses)
            self.marker_nodes.append(node)
            rvec_inv, tvec_inv = self.invert_pose(node.rvec, node.tvec)
            cumulative_rvec += rvec_inv
            cumulative_tvec += tvec_inv
            count += 1

        if count > 1:
            average_rvec = cumulative_rvec / count
            average_tvec = cumulative_tvec / count
            rospy.logwarn(f"Real time Camera Pose (Rotation Vector):,{average_rvec}")
            rospy.logwarn(f"Real time Camera Pose (Translation Vector):, {average_tvec}")

        # cv2.imshow('Estimated Pose', frame)
        # cv2.waitKey(1)

    def invert_pose(self, rvecs, tvecs):
        rmat, _ = cv2.Rodrigues(rvecs)
        rmat_inv = np.linalg.inv(rmat)
        tvecs = tvecs.reshape((3, 1))
        tvec_inv = -np.dot(rmat_inv, tvecs)
        rvec_inv, _ = cv2.Rodrigues(rmat_inv)
        return rvec_inv, tvec_inv

    def pose_estimation(self, frame, aruco_dict, camera_matrix, dist_coeffs):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        parameters = aruco.DetectorParameters_create()
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters, cameraMatrix=camera_matrix, distCoeff=dist_coeffs)
        marker_poses = []
        if ids is not None:
            for i, corner in enumerate(corners):
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corner, 0.15, camera_matrix, dist_coeffs)
                cv2.aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))
                cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
                marker_poses.append(MarkerNode(ids[i][0], rvec, tvec))
                x, y, z = tvec[0][0][0], tvec[0][0][1], tvec[0][0][2]
                cv2.putText(frame, f"X: {x:.2f}", (10, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Y: {y:.2f}", (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Z: {z:.2f}", (10, 85), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
               
                self.aruco_callback(x,y,z,ids[i][0])
                

        return frame, marker_poses

    def odom_callback(self, odom_msg):
        quaternion = (
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z,
            odom_msg.pose.pose.orientation.w
        )
        _, _, yaw = euler_from_quaternion(quaternion)
        self.state = np.array([
            odom_msg.pose.pose.position.x,
            odom_msg.pose.pose.position.y,
            yaw
        ])

    def aruco_callback(self,x,y,z,marker_id):
        self.aruco_x, self.aruco_y, self.aruco_z, self.aruco_id = x,y,z,marker_id
        aruco_position = np.array([self.aruco_x,self.aruco_y, self.aruco_z])
      
        CxF = np.array([[self.aruco_z], [self.aruco_x], [self.aruco_y]])
      
        RxF = self.robot_frame_to_feature_frame(CxF)
        
        NxB = self.state[0:3]
        NxF = self.g(NxB, RxF[0:2])
      
        self.marker_handler.add_marker(self.aruco_id,NxF)
        self.marker[self.aruco_id]=NxF

        self.idx = self.marker_handler.get_index(self.aruco_id)
    
        rospy.loginfo(f"observed Aruco List, {self.marker_handler.observed_arucos}")
        if self.aruco_id in self.marker_handler.observed_arucos:
            self.visualization_marker(self.idx,NxF)
    def visualization_marker(self, idx, aruco_position):
        marker_array = MarkerArray()
        aruco_id = list(self.marker_handler.observed_arucos.keys())[idx]

        # Cube marker
        marker = Marker()
        marker.header.frame_id = "world_ned"  # Ensure this frame exists in your TF tree
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.id = aruco_id
        marker.header.stamp = rospy.Time.now()
        marker.pose.position.x = aruco_position[0, 0]
        marker.pose.position.y = aruco_position[1, 0]
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration(0)
        marker_array.markers.append(marker)

        # Text marker
        text_marker = Marker()
        text_marker.header.frame_id = "world_ned"  # Ensure this frame exists in your TF tree
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.id = aruco_id + 20000
        text_marker.header.stamp = rospy.Time.now()
        text_marker.pose.position.x = aruco_position[0, 0]
        text_marker.pose.position.y = aruco_position[1, 0]
        text_marker.pose.position.z = 0.5
        text_marker.pose.orientation.w = 1.0
        text_marker.scale.z = 0.20
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        text_marker.text = f"ArUco {aruco_id}"
        marker_array.markers.append(text_marker)


        if len(self.marker) > 1:
            arucos=list(self.marker.keys())
            positions = list(self.marker.values())
            for i in range(len(positions) - 1):
                dist = np.linalg.norm(positions[i] - positions[i + 1])
                
                # Create line marker from one marker to the next
                line_marker = Marker()
                line_marker.header.frame_id = "world_ned"
                line_marker.header.stamp = rospy.Time.now()
                line_marker.ns = "line_markers"
                line_marker.id = 10000 + i  # Unique ID for each marker line
                line_marker.type = Marker.LINE_STRIP
                line_marker.action = Marker.ADD
                line_marker.scale.x = 0.04  # Line width
                line_marker.color.r = 1.0
                line_marker.color.g = 0.0
                line_marker.color.b = 1.0
                line_marker.color.a = 1.0
                line_marker.lifetime = rospy.Duration(0)

                # Add points to the line
                p1 = Point(positions[i][0], positions[i][1], 0.0)
                p2 = Point(positions[i + 1][0], positions[i + 1][1], 0.0)
                line_marker.points = [p1, p2]
                rospy.loginfo(f"distance between aruco {arucos[i]} and aruco {arucos[i+1]}:{dist}")
                marker_array.markers.append(line_marker)
                 # Create text marker for the distance
                dist_marker = Marker()
                dist_marker.header.frame_id = "world_ned"
                dist_marker.type = Marker.TEXT_VIEW_FACING
                dist_marker.action = Marker.ADD
                dist_marker.id = 30000 + i  # Unique ID for each distance text
                dist_marker.header.stamp = rospy.Time.now()
                # Position the text marker at the midpoint of the line
                dist_marker.pose.position.x = (positions[i][0] + positions[i + 1][0]) / 2
                dist_marker.pose.position.y = (positions[i][1] + positions[i + 1][1]) / 2
                dist_marker.pose.position.z = 0.35  # Slightly above the line
                dist_marker.scale.z = 0.22  # Text size
                dist_marker.color.r = 1.0
                dist_marker.color.g = 1.0
                dist_marker.color.b = 0.0
                dist_marker.color.a = 1.0
                dist_marker.text = f"{dist:.2f}m"  # Distance text
                marker_array.markers.append(dist_marker)

        self.perception_markers_pub_aruco_marker.publish(marker_array)


    def robot_frame_to_feature_frame(self, CxF):
        CxF_x = CxF[0, 0]
        CxF_y = CxF[1, 0]
        CxF_z = CxF[2, 0]
        RxC_x = 0.122
        RxC_y = -0.033
        RxC_z = 0.082
        RxC_yaw = 0.0
        RxF_x = RxC_x + CxF_x * np.cos(RxC_yaw) - CxF_y * np.sin(RxC_yaw)
        RxF_y = RxC_y + CxF_y * np.sin(RxC_yaw) + CxF_y * np.cos(RxC_yaw)
        RxF_z = RxC_z + CxF_z
        return np.array([[RxF_x], [RxF_y], [RxF_z]])

    def boxplus(self, NxB, BxF):
        x, y, theta = NxB[0] ,NxB[1], NxB[2]
        BxF_x, BxF_y = BxF[0, 0], BxF[1, 0]
        NxF_x = x + np.cos(theta) * BxF_x - np.sin(theta) * BxF_y
        NxF_y = y + np.sin(theta) * BxF_x + np.cos(theta) * BxF_y
        return np.array([[NxF_x], [NxF_y]])

    def g(self, NxB, znp):
        return self.boxplus(NxB, znp)

if __name__ == '__main__':
    try:
        node = ArucoDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
        
