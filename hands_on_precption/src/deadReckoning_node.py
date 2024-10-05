#!/usr/bin/env python
import math
import rospy
import tf
import numpy as np
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import Odometry,Path
from sensor_msgs.msg import JointState, Imu
from visualization_msgs.msg import Marker, MarkerArray
import tf.transformations
from tf.transformations import *
from math import *
from geometry_msgs.msg import PoseStamped,Point, Quaternion

# Defines a ROS node for dead reckoning based on wheel encoders and IMU data
class DeadReckoningNode:
    def __init__(self):
        rospy.init_node('dead_reckoning_node', anonymous=True)  # Initialize ROS node

        # Initialize wheel velocities and flags to check if velocities are received
        self.left_wheel_velocity = 0.0
        self.right_wheel_velocity = 0.0
        self.left_wheel_velocity_received = False
        self.right_wheel_velocity_received = False
        
    
        # Robot parameters: wheel radius and distance between wheels
        self.wheel_radius = 0.035
        self.wheel_base_distance = 0.235
        #Robot Dimension
        self.xB_dim=3
        #Feature Dimension
        self.xF_dim=2
       
      

        # TF broadcaster to publish transformations between coordinate frames
        self.odom_broadcaster = tf.TransformBroadcaster()
        
        # Robot parameters: wheel radius and distance between wheels
        self.wheel_radius = 0.035
        self.wheel_base_distance = 0.235

        # Initial robot state [x, y, theta] and its covariance matrix
        self.xk= np.zeros((3,)) 
        self.Pk = np.eye((3)) * 0.0
        
        # Odometry noise covariance matrix
        self.Qk = np.diag(np.array([0.00 ** 2, 0.00 ** 2,  0.001**2])) 
        
        # Publisher for odometry messages
        self.odom_pub = rospy.Publisher("turtlebot/kobuki/odom", Odometry, queue_size=10)
        #Publish MarkerArray for visualization
        self.path_pub = rospy.Publisher("turtlebot/kobuki/path_update", Path, queue_size=10)
        self.path_marker_pub = rospy.Publisher("turtlebot/kobuki/path_marker_update", Marker, queue_size=10)
        self.path = Path()
        self.path.header.frame_id = "world_ned"
        self.marker_array = MarkerArray()
        self.marker_id = 0
        
        
        self.last_time = rospy.Time.now()  # Tracks the last time a message was received
       
        # Subscribe to joint states and IMU data topics
        self.js_sub=rospy.Subscriber("turtlebot/joint_states", JointState, self.joint_states_callback, queue_size=10)
        self.imu_sub= rospy.Subscriber("turtlebot/kobuki/sensors/imu", Imu, self.imu_callback, queue_size=10)

    # Normalizes an angle to be within the range of [-pi, pi]
    def wrap_angle(self,angle):
        #corrects an angle to be within the range of [-pi, pi]
        return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )

    # Callback for processing joint states messages
    def joint_states_callback(self, msg):
        # rospy.loginfo(f"Received joint states message: {msg}")
      
        # Identify names of the left and right wheel joints
        self.left_wheel_name='turtlebot/kobuki/wheel_left_joint'
        self.right_wheel_name='turtlebot/kobuki/wheel_right_joint'
        
      
        # Extract wheel velocities from the message
        if msg.name[0]==self.left_wheel_name:
            self.left_wheel_velocity=msg.velocity[0]
         
            self.left_wheel_velocity_received=True
            
            
        elif msg.name[0]==self.right_wheel_name:
            self.right_wheel_velocity=msg.velocity[0]
          
            
        # Ensure both wheel velocities are received before proceeding
        
        if self.left_wheel_velocity_received:
            
        

            # Calculate linear and angular velocities
            
            self.left_linear_velocity=self.left_wheel_velocity*self.wheel_radius
            self.right_linear_velocity=self.right_wheel_velocity*self.wheel_radius
            
            self.linear_velocity=(self.left_linear_velocity+self.right_linear_velocity)/2
            self.angular_velocity=(self.left_linear_velocity-self.right_linear_velocity)/self.wheel_base_distance
            
            
            # # Compute new odometry based on velocities and time elapsed
            self.current_time=rospy.Time.from_sec(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
            self.time=(self.current_time-self.last_time).to_sec()
            self.last_time=self.current_time
            
            
            
             
            ## Update robot's pose based on motion model
            
            self.xk[0]=self.xk[0]+np.cos(self.xk[2])*self.linear_velocity*self.time
            self.xk[1]=self.xk[1]+np.sin(self.xk[2])*self.linear_velocity*self.time
            
            #Angle normalization
            self.xk[2] = self.wrap_angle(self.xk[2] + self.angular_velocity*self.time)
           
           
            
            # Prepare for prediction step by updating the state and covariance         
           
            self.xk, self.Pk=self.prediction(self.xk,self.Pk,self.linear_velocity,self.angular_velocity,self.time)
            
           
           
            self.publish_odometry(self.xk, self.Pk)
            self.update_path()
          
            
          
            # Reset flags for next iteration
            
            self.left_wheel_velocity_received = False
    
            
            
    #Prediction Step      
    def prediction(self,xk,Pk,v,w,t):
        
        #Mean
        self.xk=self.xk.copy()
      
        
        #Covariance
        #Compute Jacobian of motion model respect to xk
        Ak= np.array([[1.0,    0.0,    -np.sin(xk[2])*v*t],
                    [0.0,    1.0,     np.cos(xk[2])*v*t],
                    [0.0,    0.0,     1.0]])
        # #Compute Jacobian of process noise respect to noise
        
        Wk = np.array([[np.cos(self.xk[2])*t*0.5*self.wheel_radius, np.cos(self.xk[2])*t*0.5*self.wheel_radius,0.0],    
                            [np.sin(self.xk[2])*t*0.5*self.wheel_radius, np.sin(self.xk[2])*t*0.5*self.wheel_radius,0.0],   
                                
                            [(t*self.wheel_radius)/self.wheel_base_distance, -(t*self.wheel_radius)/self.wheel_base_distance,1.0]])     
    
        Qk=self.Qk
        
        #Compute covariance based on Jacobians and noise
        self.Pk=Ak@Pk@Ak.T + Wk@Qk@Wk.T
        
        return self.xk, self.Pk
  
####Sensor gives Quaternion form


    def imu_callback(self, msg):
       
        # Extract the quaternion from the IMU message
        quaternion = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)

        # convert the orientation message received from quaternion to euler
        _, _ , yaw_measurement = euler_from_quaternion(quaternion)
       
        
        self.update(yaw_measurement)
       
    # Kalman filter update
    def update(self,yaw_measurement):

        #Actual measurement
        self.zk= np.array([yaw_measurement])
      
        #Jacobian of the observation model with respect to the state vector.
        Hk=np.array([[0,0,1]])
        
        
        # Jacobian of the observation model with respect to the noise vector.
        Vk = np.diag([0.001])  

        # Covariance matrix of the noise vector
        Rk = np.array([[0.00001]])  
        
        
        #expected observation
        self.h=np.array([self.xk[2]])
        
        #Different between actual and predicted measurement
        # innovation
        
        innovation=self.wrap_angle(self.zk-self.h)
        
       
        # Compute Kalman gain
        Kk = self.Pk @ Hk.T @ np.linalg.inv(Hk @ self.Pk @ Hk.T + Vk @ Rk @ Vk.T)

       
        self.xk  = self.xk + Kk @ (innovation)
        I        = np.eye((len(self.xk)))
        self.Pk  = (I - Kk @ Hk) @ self.Pk @ (I - Kk @ Hk).T
        
        return self.xk, self.Pk


    #Publish Odometry
    def publish_odometry(self, xk, Pk):
      
        ## Convert the yaw angle to a quaternion
       
        """
        Publishes odometry data to ROS.

        :param xk: Current state vector [x, y, theta].
        :param Pk: Covariance matrix of the state.
        :param linear_velocity: Linear velocity of the robot.
        :param angular_velocity: Angular velocity of the robot.
        :param current_time: Current ROS time, used as the timestamp for the odometry message.
        """
        
        # Convert the yaw angle to a quaternion for representing 3D orientations.
        self.q = quaternion_from_euler(0, 0, xk[2])

        # Initialize an odometry message.
        odom = Odometry()
        odom.header.stamp =self.current_time
        odom.header.frame_id = "world_ned"
        odom.child_frame_id = "turtlebot/kobuki/base_footprint"
       
        # Set the position in the odometry message.
        odom.pose.pose.position.x = xk[0]
        odom.pose.pose.position.y = xk[1]
       
        # Set the orientation in the odometry message.
        odom.pose.pose.orientation.x = self.q[0]
        odom.pose.pose.orientation.y = self.q[1]
        odom.pose.pose.orientation.z = self.q[2]
        odom.pose.pose.orientation.w = self.q[3]

        # Set the velocities in the odometry message.
        odom.twist.twist.linear.x = self.linear_velocity
        odom.twist.twist.angular.z = self.angular_velocity

        
        # Convert covariance matrix from np.array to list
        P_list = Pk.tolist()  
        # Update the diagonal elements directly for variance
        odom.pose.covariance[0] = P_list[0][0]  # Variance in x
        odom.pose.covariance[7] = P_list[1][1]  # Variance in y
        odom.pose.covariance[35] = P_list[2][2]  # Variance in yaw

        # Update the off-diagonal elements for covariance between variables
        odom.pose.covariance[1] = P_list[0][1]  # Covariance between x and y
        odom.pose.covariance[6] = P_list[1][0]  # Covariance between y and x 

        odom.pose.covariance[5] = P_list[0][2]  # Covariance between x and yaw
        odom.pose.covariance[30] = P_list[2][0]  # Covariance between yaw and x

        odom.pose.covariance[11] = P_list[1][2]  # Covariance between y and yaw
        odom.pose.covariance[31] = P_list[2][1]  # Covariance between yaw and y
         

        
        self.odom_pub.publish(odom)
        
        
        # Publish the transform over tf
        self.odom_broadcaster.sendTransform((self.xk[0], self.xk[1], 0.0), self.q, rospy.Time.now(), odom.child_frame_id, odom.header.frame_id)
      
    
        
    def update_path(self):
        """ Update the path with the current position of the robot"""
        
        quaternion = quaternion_from_euler(0, 0, self.xk[2])
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "world_ned"
        pose.pose.position.x = self.xk[0]
        pose.pose.position.y = self.xk[1]
        pose.pose.orientation = Quaternion(*quaternion)
        self.path.poses.append(pose)
        self.path.header.stamp = rospy.Time.now()
        self.path_pub.publish(self.path)

        point = Point(pose.pose.position.x, pose.pose.position.y,0.0)
        marker = Marker()
        marker.header.frame_id = "world_ned"
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)
        marker.points.append(point)
        self.path_marker_pub.publish(marker)
        
        
        
if __name__ == '__main__':
    try:
        # Call the DeadReckoning function 
        DeadReckoningNode()
        # Keep the program running until rospy is shut down
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

