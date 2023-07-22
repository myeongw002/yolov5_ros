#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage, Image
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

# Create an instance of CvBridge
cv_bridge = CvBridge()
# Initialize the lists for yellow and blue cones
yellow_cones = []
blue_cones = []
image = None
steer = 0


#depth plotting
K = np.array([684.1913452148438, 0.0, 482.5340881347656, 
              0.0, 684.1913452148438, 255.77565002441406, 
              0.0, 0.0, 1.0]).reshape((3,3))
plt.ion()
figure, ax = plt.subplots(figsize=(8,6))
point = np.array([[0,0]])
line, = ax.plot(point[:,0],point[:,1],'og')




class MovingAverage:
    def __init__(self, n):
        self.samples = n
        self.data = []
        self.weights = list(range(1, n+1))
        
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



steer_mov_avg = MovingAverage(5)
def calc_steer(steer):
    erp_serial = ESerial()
    steer_mov_avg.add_sample(steer)
    avg_steer = int(steer_mov_avg.get_wmm())
    erp_serial.steer = avg_steer
    erp_serial.speed = 40
    steer_pub.publish(erp_serial)
    
    print(avg_steer, end='\r')




def draw_cones(image, cones):
    if len(cones) > 1:
        if cones[0].Class == 'yellow_cone':
            for i in range(len(cones) - 1):
                if i + 1 < len(cones):
                    line_point1 = (cones[i].xmax, cones[i].ymax)
                    line_point2 = (cones[i + 1].xmax, cones[i+1].ymax)
                    cv2.line(image, line_point1, line_point2, (0, 255, 255), 10)
                    
        elif cones[0].Class == 'blue_cone':
            for i in range(len(cones) - 1):
                if i + 1 < len(cones):
                    line_point1 = (cones[i].xmin, cones[i].ymax)
                    line_point2 = (cones[i + 1].xmin, cones[i+1].ymax)                    
                    cv2.line(image, line_point1, line_point2, (255, 0, 0), 10)
                    
    
    elif len(cones) == 1:
        if cones[0].Class == 'yellow_cone':
            point = (cones[0].xmax, cones[0].ymax)
            cv2.circle(image, point, radius=5, color=(0, 255, 255), thickness=-1)
        else:
            point = (cones[0].xmin, cones[0].ymax)
            cv2.circle(image, point, radius=5, color=(255, 0, 0), thickness=-1)    
    else:
        pass
    
    
        
def draw_path(image, yellow_cones, blue_cones):
    # Find points on the line connecting the yellow cones
    yellow_line_points = []
    blue_line_points = []
    
    car_center_x = int(image.shape[1]*0.5)
    car_center_y = image.shape[0]
    
    for i in range(len(yellow_cones) - 1):
        if i + 1 < len(yellow_cones):
            pt1 = Point()
            pt1.x = (yellow_cones[i].xmin + yellow_cones[i].xmax) / 2
            pt1.y = yellow_cones[i].ymax

            pt2 = Point()
            pt2.x = (yellow_cones[i+1].xmin + yellow_cones[i+1].xmax) / 2
            pt2.y = yellow_cones[i+1].ymax

            yellow_line_points.append(pt1)
            yellow_line_points.append(pt2)

            # Calculate the slope of the line connecting the two yellow cones
            dx = pt2.x - pt1.x
            dy = pt2.y - pt1.y
            slope = dy / dx if dx != 0 else float('inf')

            # Interpolate between the two yellow cones and add the points to the line points
            num_points = 10  # Number of points to interpolate
            for j in range(num_points):
                t = j / float(num_points - 1)  # Interpolation parameter between 0 and 1
                x = int(pt1.x + t * dx)
                y = int(pt1.y + t * dy)
                yellow_line_points.append(Point(x=x, y=y, z=0.0))

    for i in range(len(blue_cones) - 1):
        if i + 1 < len(blue_cones):
            pt1 = Point()
            pt1.x = (blue_cones[i].xmin + blue_cones[i].xmax) / 2
            pt1.y = blue_cones[i].ymax

            pt2 = Point()
            pt2.x = (blue_cones[i+1].xmin + blue_cones[i+1].xmax) / 2
            pt2.y = blue_cones[i+1].ymax

            blue_line_points.append(pt1)
            blue_line_points.append(pt2)

            # Calculate the slope of the line connecting the two blue cones
            dx = pt2.x - pt1.x
            dy = pt2.y - pt1.y
            slope = dy / dx if dx != 0 else float('inf')

            # Interpolate between the two blue cones and add the points to the line points
            num_points = 10  # Number of points to interpolate
            for j in range(num_points):
                t = j / float(num_points - 1)  # Interpolation parameter between 0 and 1
                x = int(pt1.x + t * dx)
                y = int(pt1.y + t * dy)
                blue_line_points.append(Point(x=x, y=y, z=0.0))

        # Find intersection points between yellow and blue cone lines
        intersection_points = []
        num_points = min(len(yellow_line_points), len(blue_line_points))

        for i in range(num_points):
            yellow_point = yellow_line_points[i]
            blue_point = blue_line_points[i]

            center_x = int((yellow_point.x + blue_point.x) / 2)
            center_y = max(yellow_point.y, blue_point.y)
            intersection_points.append(Point(center_x, center_y, 0))
            
        # Calculate steer if there are enough intersection points
        if len(yellow_cones)>= 1 and len(blue_cones) >= 1:
            path_center = (yellow_cones[0].xmax + blue_cones[0].xmin) *0.5
            steer = ((-car_center_x + path_center) / car_center_x ) * 2000
            calc_steer(steer)
                             
        # Draw the path between intersection points
        for i in range(0,len(intersection_points) - 1):
            cv2.circle(image, (intersection_points[i].x, intersection_points[i].y), radius=5, color=(255, 255, 255), thickness=-1)
            cv2.line(image, (car_center_x,car_center_y), (intersection_points[0].x,intersection_points[0].y) , (255, 255, 255), 10)
        
        
        
        
def calc_XYZ(cones, depth_image):
    depth_pixel_array = []
    line_array = np.empty((0,3))
    for i in range(len(cones)):
        if cones[0].Class == "yellow_cone":
            xp = cones[i].xmax
            yp = cones[i].ymax
        else:
            xp = cones[i].xmin
            yp = cones[i].ymax           
        line_array = np.append(line_array, np.array([[xp,yp,1]]),axis=0)
        # depth 이미지에서 콘의 거리[m] 값 추출
        index_depth = depth_image[cones[i].ymin:cones[i].ymax, cones[i].xmin:cones[i].xmax]
        index_depth = index_depth[np.isfinite(index_depth)]
        lange = np.min(index_depth)
        depth_pixel_array.append(lange)
        
    homo = np.linalg.inv(K) @ line_array.T
    xz = homo[0,:]*homo[0,:]; yz = homo[1,:]*homo[1,:]

    Z = np.array(depth_pixel_array)/np.sqrt(xz+yz+np.ones(xz.shape[0]))
    X = homo[0,:]*Z
    Y = homo[1,:]*Z
    XYZ = np.vstack(( np.vstack((X,Y)), Z)).T

    return XYZ, line_array

        
        
        
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
    


def image_callback(image_msg):
    global image
    # Convert the compressed image message to OpenCV format
    image = cv_bridge.compressed_imgmsg_to_cv2(image_msg)

    # Check if the image dimensions are valid
    if image.shape[0] == 0 or image.shape[1] == 0:
        return

XYZ_yellow = np.array([])
XYZ_blue = np.array([])
def depth_callback(msg):
    global cv_bridge, XYZ_yellow, XYZ_blue
    depth_image = cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    #print(type(image))
    #print(image.shape) # (540, 960)
    
    try:
        XYZ_yellow, center_yellow = calc_XYZ(yellow_cones, depth_image)
    except:
        pass
    try:
        XYZ_blue, center_blue = calc_XYZ(blue_cones, depth_image)
    except:
        pass    
   

if __name__ == '__main__':
    rospy.init_node('yolo_sub')
    
    # Create a publisher for the marker
    marker_pub = rospy.Publisher('traffic_cones_marker', Marker, queue_size=10)
    steer_pub = rospy.Publisher('erp42_serial', ESerial, queue_size=10)
    # Subscribe to the BoundingBoxes topic
    box_topic = rospy.get_param("~input_box_topic", "/yolov5/detections")
    sub_boxes = rospy.Subscriber(box_topic, BoundingBoxes, bounding_boxes_callback)
    # Subscribe to the compressed image topic
    image_topic = rospy.get_param("~input_image_topic", 'usb_cam/image/raw')
    sub_image = rospy.Subscriber(image_topic, CompressedImage, image_callback)
    # Subscribe to the depth image topic
    depth_image_topic = rospy.get_param("~depth_image_topic", 'usb_cam/image/raw')
    deth_image = rospy.Subscriber(depth_image_topic, Image, depth_callback)    
    
    rate = rospy.Rate(100)
    rospy.wait_for_message(box_topic, BoundingBoxes)
    
    while not rospy.is_shutdown():
        draw_cones(image, yellow_cones)
        draw_cones(image, blue_cones)
        draw_path(image, yellow_cones, blue_cones)
        cv2.imshow('Image with Lines', image)
        
        if XYZ_yellow.shape[0] != 0 and XYZ_blue.shape[0] != 0:
            line = ax.plot(XYZ_yellow[:,0],XYZ_yellow[:,2],'oy')
            line = ax.plot(XYZ_blue[:,0],XYZ_blue[:,2],'ob')

            figure.canvas.draw()
            figure.canvas.flush_events()
            plt.cla()
            
        cv2.waitKey(1)    
        rate.sleep()





