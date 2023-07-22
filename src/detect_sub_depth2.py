#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from detection_msgs.msg import BoundingBox, BoundingBoxes
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Int64
from erp42_serial.msg import ESerial
import cv2
from cv_bridge import CvBridge
import numpy as np
from scipy.interpolate import splprep, splev
import math
import matplotlib.pyplot as plt
from visualization_msgs.msg import Marker


K = np.array([684.1913452148438, 0.0, 482.5340881347656, 
              0.0, 684.1913452148438, 255.77565002441406, 
              0.0, 0.0, 1.0]).reshape((3,3))

plt.ion()
figure, ax = plt.subplots(figsize=(8,6))
point = np.array([[0,0]])
line, = ax.plot(point[:,0],point[:,1],'og')


# Create an instance of CvBridge
cv_bridge = CvBridge()
yellow_cones = []
blue_cones = []



class MovingAverage:
    def __init__(self, n):
        self.samples = n
        self.data = []
        self.weights = list(range(1, n+1))
        self.steer_pub = rospy.Publisher('erp42_serial', ESerial, queue_size=10)
        
    def add_sample(self, new_sample):
        if len(self.data) < self.samples:
            self.data.append(new_sample)
        else:
            self.data = self.data[1:] + [new_sample]
            
    def get_wmm(self):
        s = 0
        for i, x in enumerate(self.data):
            s += x * self.weights[i]
        return float(s) / sum(self.weights[:len(self.data)])

    def calc_steer(self,steer):
        erp_serial = ESerial()
        self.add_sample(steer)
        erp_serial.steer = int(self.get_wmm())
        erp_serial.speed = 40
        self.steer_pub.publish(erp_serial)
        
        print(erp_serial.steer     , end='\r')


def publish_marker(points, color):
    marker_msg = Marker()
    marker_msg.header.frame_id = "zed_camera_center"  # Set the frame_id according to your robot's frame
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
        p.x = point[2] #zed x = depth z
        p.y = -point[0]  #zed y  = depth -x
        p.z = -point[1] #zed z = depth y
        marker_msg.points.append(p)
    return marker_msg



def bounding_boxes_callback(msg):
    global yellow_cones
    global blue_cones
    # Clear the previous cone lists
    yellow_cones.clear()
    blue_cones.clear()

    # Extract yellow and blue cones based on the label names
    for bounding_box in msg.bounding_boxes:
        if bounding_box.Class == "yellow_cone":
            yellow_cones.append(bounding_box)
        elif bounding_box.Class == "blue_cone":
            blue_cones.append(bounding_box)

    # Sort the cones by ycenter in descending order
    yellow_cones.sort(key=lambda box: box.ymax, reverse=True)
    blue_cones.sort(key=lambda box: box.ymax, reverse=True)
    

def interpolate_and_plot_path(XYZ_coordinates):
    m = XYZ_coordinates.shape[0]
    if m <= 1:
        return  # Not enough data points for interpolation

    # Choose the degree of the spline based on the number of data points
    k = min(2, m - 1)  # Use k=2 for quadratic interpolation if enough data points, else use the highest degree possible

    # Perform spline interpolation on the XZ coordinates
    tck, u = splprep([XYZ_coordinates[:, 0], XYZ_coordinates[:, 1], XYZ_coordinates[:, 2]], k=k, s=0)

    # Evaluate the interpolated path at desired intervals (e.g., 100 points along the path)
    u_new = np.linspace(0, 1, 5)
    interpolated_points = np.array(splev(u_new, tck)).T

    # Plot the interpolated path on the XZ plane
    ax.plot(interpolated_points[:, 0], interpolated_points[:, 2], '-r')
    
    return interpolated_points
    


def cal_XYZ(cones, depth_image):
    depth_pixel_array = []
    center_array = np.empty((0,3))
    for i in range(len(cones)):
        if cones[0].Class == "yellow_cone":
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
    homo = np.linalg.inv(K) @ center_array.T
    xz = homo[0,:]*homo[0,:]; yz = homo[1,:]*homo[1,:]

    Z = np.array(depth_pixel_array)/np.sqrt(xz+yz+np.ones(xz.shape[0]))
    X = homo[0,:]*Z
    Y = homo[1,:]*Z
    XYZ = np.vstack(( np.vstack((X,Y)), Z)).T

    return XYZ, center_array



XYZ_yellow = np.array([])
XYZ_blue = np.array([])
def depth_callback(msg):
    global cv_bridge, XYZ_yellow, XYZ_blue
    depth_image = cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    try:
        XYZ_yellow, center_yellow = cal_XYZ(yellow_cones, depth_image)
    except:
        pass
    try:
        XYZ_blue, center_blue = cal_XYZ(blue_cones, depth_image)
    except:
        pass


if __name__ == '__main__':
    rospy.init_node('yolo_sub')
  
    yellow_marker_publisher = rospy.Publisher("/path/yellow_marker", Marker, queue_size=10)
    blue_marker_publisher = rospy.Publisher("/path/blue_marker", Marker, queue_size=10)
    center_marker_publisher = rospy.Publisher("/path/center_paths_marker", Marker, queue_size=10)
    # Subscribe to the BoundingBoxes topic
    box_topic = rospy.get_param("~input_box_topic", "/yolov5/detections")
    sub_boxes = rospy.Subscriber(box_topic, BoundingBoxes, bounding_boxes_callback)
    # Subscribe to the depth image topic
    depth_image_topic = rospy.get_param("~depth_image_topic", 'usb_cam/image/raw')
    deth_image = rospy.Subscriber(depth_image_topic, Image, depth_callback)  
    
    steer_avg = MovingAverage(5)
    rospy.wait_for_message(box_topic, BoundingBoxes)
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():

        if XYZ_yellow.shape[0] != 0 and XYZ_blue.shape[0] != 0:
            line = ax.plot(XYZ_yellow[:,0],XYZ_yellow[:,2],'oy')
            line = ax.plot(XYZ_blue[:,0],XYZ_blue[:,2],'ob')
            
            interpolated_yellow = interpolate_and_plot_path(np.vstack((XYZ_yellow[:, 0],XYZ_yellow[:, 1], XYZ_yellow[:, 2])).T)
            interpolated_blue = interpolate_and_plot_path(np.vstack((XYZ_blue[:, 0],XYZ_blue[:, 1], XYZ_blue[:, 2])).T)
            
            if interpolated_yellow is not None and interpolated_blue is not None:
                # Calculate the centerline between the interpolated paths
                centerline = (interpolated_yellow + interpolated_blue) / 2

                # Plot the centerline on the XZ plane
                ax.plot(centerline[:, 0], centerline[:, 2], '-g')
                yellow_maker = publish_marker(interpolated_yellow, (1.0, 1.0, 0.0))  # Red for interpolated yellow path
                blue_maker = publish_marker(interpolated_blue, (0.0, 0.0, 1.0))   # Blue for interpolated blue path
                center_marker = publish_marker(centerline, (0.0, 1.0, 0.0))          # Green for centerline                     
                yellow_marker_publisher.publish(yellow_maker)          
                blue_marker_publisher.publish(blue_maker)
                center_marker_publisher.publish(center_marker)
                steer_avg.calc_steer(2000*centerline[0][0])
               
            figure.canvas.draw()
            figure.canvas.flush_events()
            plt.cla()
        rate.sleep()
    
        





