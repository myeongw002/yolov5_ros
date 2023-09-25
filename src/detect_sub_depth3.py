#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from detection_msgs.msg import BoundingBox, BoundingBoxes
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Int64
from erp42_serial.msg import ESerial, setState
import cv2
from cv_bridge import CvBridge
import numpy as np
from scipy.interpolate import splprep, splev
import math
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import traceback


class Cone_Detector:
    def __init__(self):
        self.K = np.array([684.1913452148438, 0.0, 482.5340881347656, 
                  0.0, 684.1913452148438, 255.77565002441406, 
                  0.0, 0.0, 1.0]).reshape((3,3))
    # Create an instance of CvBridge
        self.cv_bridge = CvBridge()
        self.yellow_cones = []
        self.blue_cones = []
        self.XYZ_yellow = np.array([])
        self.XYZ_blue = np.array([])
        self.img_center = None
        self.ros_topic_func()
        
        
    def ros_topic_func(self):
        self.yellow_marker_publisher = rospy.Publisher("/path/yellow_marker", Marker, queue_size=10)
        self.blue_marker_publisher = rospy.Publisher("/path/blue_marker", Marker, queue_size=10)
        self.center_marker_publisher = rospy.Publisher("/path/center_paths_marker", Marker, queue_size=10)
        # Subscribe to the BoundingBoxes topic
        box_topic = rospy.get_param("~input_box_topic", "/yolov5/detections")
        self.sub_boxes = rospy.Subscriber(box_topic, BoundingBoxes, self.bounding_boxes_callback, queue_size=1)
        # Subscribe to the depth image topic
        depth_image_topic = rospy.get_param("~depth_image_topic", 'usb_cam/image/raw')
        self.deth_image_sub = rospy.Subscriber(depth_image_topic, Image, self.depth_callback, queue_size=1)
        rospy.wait_for_message(depth_image_topic, Image)
        rospy.wait_for_message(box_topic, BoundingBoxes)
        
        
    def publish_marker(self,points, color):
        marker_msg = Marker()
        marker_msg.header.frame_id = "base_link"  # Set the frame_id according to your robot's frame
        marker_msg.header.stamp = rospy.Time.now()
        marker_msg.ns = "interpolated_paths"
        marker_msg.id = 0
        marker_msg.type = Marker.LINE_STRIP
        marker_msg.action = Marker.ADD
        marker_msg.pose.orientation.w = 1.0
        marker_msg.scale.x = 0.1  # Line width
        marker_msg.color.a = 1.0
        marker_msg.color.r = color[0]
        marker_msg.color.g = color[1]
        marker_msg.color.b = color[2]
        for point in points:
            p = Point()
            p.x = point[2] #real world x = depth image z
            p.y = -point[0]  #real world y  = depth image -x
            p.z = 0#-point[1] #real world z = depth image y
            marker_msg.points.append(p)
        return marker_msg



    def bounding_boxes_callback(self,msg):
        # Clear the previous cone lists
        self.yellow_cones.clear()
        self.blue_cones.clear()

        # Extract yellow and blue cones based on the label names
        for bounding_box in msg.bounding_boxes:
            if bounding_box.Class == "yellow_cone":
                self.yellow_cones.append(bounding_box)
            elif bounding_box.Class == "blue_cone":
                self.blue_cones.append(bounding_box)

        # Sort the cones by ycenter in descending order
        self.yellow_cones.sort(key=lambda box: (box.ymax, abs(box.xmax - self.img_center)), reverse=True)
        self.blue_cones.sort(key=lambda box: (box.ymax, abs(box.xmin - self.img_center)), reverse=True)
        
        #print('detected yellow',len(self.yellow_cones))
        #print('detected blue',len(self.blue_cones))




    def cal_XYZ(self, cones, depth_image):
        depth_pixel_array = []
        center_list = []  # Use a list for efficient appending
        valid_depth_indices = []  # Keep track of valid depth indices

        for idx, cone in enumerate(cones):
            xp = cone.xmax if cone.Class == "yellow_cone" else cone.xmin
            yp = cone.ymax

            # Extract the depth value from the depth image for the cone's center
            depth_val = depth_image[int(yp), int(xp)]
            
            if np.isfinite(depth_val):  # Check if depth value is valid (not infinite)
                center_list.append([xp, yp, 1])
                depth_pixel_array.append(depth_val)
                valid_depth_indices.append(idx)

        center_array = np.array(center_list)
        homo = np.linalg.inv(self.K) @ center_array.T
        xz = homo[0, :] * homo[0, :]
        yz = homo[1, :] * homo[1, :]

        Z = np.array(depth_pixel_array) / np.sqrt(xz + yz + np.ones(xz.shape[0]))
        X = homo[0, :] * Z
        Y = homo[1, :] * Z
        XYZ = np.vstack((np.vstack((X, Y)), Z)).T

        return XYZ, center_array



    def depth_callback(self,msg):
        self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.img_center = self.depth_image.shape[1] / 2
        self.XYZ_yellow = np.array([])
        self.XYZ_blue = np.array([])
        try:
            self.XYZ_yellow, _ = self.cal_XYZ(self.yellow_cones, self.depth_image)

            #print('len', len(self.yellow_cones))
            #print('rows', self.XYZ_yellow.shape[0])
            #print('calc yellow',len(yellow_center))
        except:
            self.XYZ_yellow = np.array([])
            
        try:
            self.XYZ_blue, _ = self.cal_XYZ(self.blue_cones, self.depth_image)
            #print('calc blue',len(blue_center))
        except:
            self.XYZ_blue = np.array([])
          
    def publish_marker(self,points, color):
        marker_msg = Marker()
        marker_msg.header.frame_id = "base_link"  # Set the frame_id according to your robot's frame
        marker_msg.header.stamp = rospy.Time.now()
        marker_msg.ns = "interpolated_paths"
        marker_msg.id = 0
        marker_msg.type = Marker.LINE_STRIP
        marker_msg.action = Marker.ADD
        marker_msg.pose.orientation.w = 1.0
        marker_msg.scale.x = 0.1  # Line width
        marker_msg.color.a = 1.0
        marker_msg.color.r = color[0]
        marker_msg.color.g = color[1]
        marker_msg.color.b = color[2]
        for point in points:
            p = Point()
            p.x = point[2] #real world x = depth image z
            p.y = -point[0]  #real world y  = depth image -x
            p.z = 0#-point[1] #real world z = depth image y
            marker_msg.points.append(p)
        return marker_msg

    
    def main(self):
        
        distance = float(rospy.get_param("~distance", 2.2))
        distance = math.sqrt(distance)
        rate = rospy.Rate(15)
        while not rospy.is_shutdown():
            print('detected yellow',len(self.yellow_cones))
            print('detected blue',len(self.blue_cones))
            print('XYZ yellow', self.XYZ_yellow.shape[0])
            print('XYZ blue', self.XYZ_blue.shape[0])
            yellow_maker = self.publish_marker(self.XYZ_yellow, (1.0, 1.0, 0.0))  
            # Red for interpolated yellow path
            blue_maker = self.publish_marker(self.XYZ_blue, (0.0, 0.0, 1.0))   
            # Blue for interpolated blue path
     
            # Green for centerline                     
            self.yellow_marker_publisher.publish(yellow_maker)          
            self.blue_marker_publisher.publish(blue_maker)

                

                        
            rate.sleep()        
        

if __name__ == '__main__':
    rospy.init_node('yolo_sub')
    cone_detector = Cone_Detector()
    cone_detector.main()
        
            
