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
        self.plt = plt
        self.plt.ion()
        self.figure, self.ax = self.plt.subplots(figsize=(8,6))
        self.point = np.array([[0,0]])
        self.line, = self.ax.plot(self.point[:,0],self.point[:,1],'og')
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
        
        print('detected yellow',len(self.yellow_cones))
        print('detected blue',len(self.blue_cones))




    def cal_XYZ(self,cones, depth_image):
        depth_pixel_array = []
        center_array = np.empty((0,3))
        for i in range(len(cones)):
            if cones[i].Class == "yellow_cone":
                xp = cones[i].xmax
                yp = cones[i].ymax
            else:
                xp = cones[i].xmin
                yp = cones[i].ymax 
            center_array = np.append(center_array, np.array([[xp,yp,1]]),axis=0)
            # depth 이미지에서 콘의 거리[m] 값 추출
            index_depth = depth_image[cones[i].ymin:cones[i].ymax, cones[i].xmin:cones[i].xmax]
            index_depth = index_depth[np.isfinite(index_depth)]
            lange = np.min(index_depth)
            depth_pixel_array.append(lange)
            
        # print(yel_center_array.T)
        homo = np.linalg.inv(self.K) @ center_array.T
        xz = homo[0,:]*homo[0,:]; yz = homo[1,:]*homo[1,:]

        Z = np.array(depth_pixel_array)/np.sqrt(xz+yz+np.ones(xz.shape[0]))
        X = homo[0,:]*Z
        Y = homo[1,:]*Z
        XYZ = np.vstack(( np.vstack((X,Y)), Z)).T

        return XYZ, center_array


    def depth_callback(self,msg):
        self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.img_center = self.depth_image.shape[1] / 2
        
        # try:
        #     self.XYZ_yellow, _ = self.cal_XYZ(self.yellow_cones, self.depth_image)

        #     #print('len', len(self.yellow_cones))
        #     #print('rows', self.XYZ_yellow.shape[0])
        #     #print('calc yellow',len(yellow_center))
        # except:
        #     pass
            
        # try:
        #     self.XYZ_blue, _ = self.cal_XYZ(self.blue_cones, self.depth_image)
        #     #print('calc blue',len(blue_center))
        # except:
        #     pass


    def pixel2meter(self,cones, Y):
        center_array = np.empty((0,3))
        for i in range(len(cones)):
            if cones[i].Class == "yellow_cone":
                xp = cones[i].xmax
                yp = cones[i].ymax
            else:
                xp = cones[i].xmin
                yp = cones[i].ymax
            center_array = np.append(center_array, np.array([[xp, yp, 1]]), axis=0)
        Z = self.K[1,1]*Y/(center_array[:,1]-self.K[1,2])
        X = (center_array[:,0]-self.K[0,2])/(self.K[0,0]) * Z
        XYZ = np.vstack(( np.vstack((X,np.ones(Z.shape[0])*Y)), Z)).T
        return XYZ


    def main(self):
   
        distance = float(rospy.get_param("~distance", 2.2))
        distance = math.sqrt(distance)
        rate = rospy.Rate(15)
        while not rospy.is_shutdown():
            self.XYZ_yellow = self.pixel2meter(self.yellow_cones, 1) # Y = 바닥으로부터 카메라 높이 meter 1미터로 넣어뒀으니 측정해서 넣기
            self.XYZ_blue = self.pixel2meter(self.blue_cones, 1) # Y = 바닥으로부터 카메라 높이 meter 

            try:
                self.line =self. ax.plot(self.XYZ_yellow[:,0], self.XYZ_yellow[:,2],'oy')
                self.line = self.ax.plot(self.XYZ_blue[:,0], self.XYZ_blue[:,2],'ob')

                self.figure.canvas.draw()
                self.figure.canvas.flush_events()
                self.plt.cla()
                
            except Exception as e:
                tb = traceback.format_exc()  # Get the traceback information
                error_line = tb.split("\n")[-3]  # Extract the error line from the traceback
                print(f"Error: {e} on {error_line}")
                        
            rate.sleep()        
        

if __name__ == '__main__':
    rospy.init_node('yolo_sub')
    cone_detector = Cone_Detector()
    cone_detector.main()
        
            
