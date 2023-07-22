#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage
from detection_msgs.msg import BoundingBox, BoundingBoxes
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Int64
from erp42_serial.msg import ESerial
import cv2
from cv_bridge import CvBridge

rospy.init_node('yolo_sub')

class MovingAverage:
    def __init__(self,n):
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
        for i,x in enumerate(self.data):
            s += x * self.weights[i]
        return float(s) / sum(self.weights[:len(self.data)])

mov_avg = MovingAverage(10)
center_x_avg = MovingAverage(10)
center_y_avg = MovingAverage(10)

# Create a publisher for the marker
marker_pub = rospy.Publisher('traffic_cones_marker', Marker, queue_size=10)
steer_pub = rospy.Publisher('erp42_serial', ESerial, queue_size = 10)
# Create an instance of CvBridge
cv_bridge = CvBridge()

# Initialize the lists for left and right side cones
left_cones = []
right_cones = []

# Initialize the image width
image_width = 0
steer = 0
center_x = 0
car_center_w = 0


def calc_steer(steer):
    erp_serial = ESerial()
    mov_avg.add_sample(steer)
    avg_steer = int(mov_avg.get_wmm())
    erp_serial.steer = avg_steer
    erp_serial.speed = 15
    steer_pub.publish(erp_serial)
    
    print(avg_steer    , end='\r')

def bounding_boxes_callback(msg):
    # Clear the previous cone lists
    left_cones.clear()
    right_cones.clear()

    # Sort the bounding boxes by ycenter in descending order
    sorted_boxes = sorted(msg.bounding_boxes, key=lambda box: box.ymax, reverse=True)

    # Divide the cones into left and right sides based on xcenter
    for bounding_box in sorted_boxes:
        center_point = Point()
        center_point.x = int((bounding_box.xmin +bounding_box.xmax)*0.5)
        center_point.y = bounding_box.ymax
        center_point.z = 0.0

        if center_point.x < image_width / 2:
            left_cones.append(center_point)
        else:
            right_cones.append(center_point)



def image_callback(image_msg):
    
    # Convert the compressed image message to OpenCV format
    image = cv_bridge.compressed_imgmsg_to_cv2(image_msg)

    # Check if the image dimensions are valid
    if image.shape[0] == 0 or image.shape[1] == 0:
        return

    # Update the image width
    global image_width
    global center_x
    global car_center_w
    global steer
    image_width = image.shape[1]
    
    car_center_h = int(image.shape[0])
    car_center_w = int(image.shape[1] * 0.5)
    car_center = (car_center_w, car_center_h)
    # Calculate the line thickness based on the image height
    image_height = image.shape[0]
    line_thickness = max(1, int(image_height / 200))  # Adjust the division value as needed

    # Draw lines for left side cones
    for i in range(len(left_cones) - 1):
        pt1 = (int(left_cones[i].x), int(left_cones[i].y))
        pt2 = (int(left_cones[i + 1].x), int(left_cones[i + 1].y))
        cv2.line(image, pt1, pt2, (0, 255, 0), line_thickness)

    # Draw lines for right side cones
    for i in range(len(right_cones) - 1):
        pt1 = (int(right_cones[i].x), int(right_cones[i].y))
        pt2 = (int(right_cones[i + 1].x), int(right_cones[i + 1].y))
        cv2.line(image, pt1, pt2, (0, 0, 255), line_thickness)

    for center_point in left_cones:
        cv2.circle(image, (int(center_point.x), int(center_point.y)), 3, (0, 255, 0), -1)
    for center_point in right_cones:
        cv2.circle(image, (int(center_point.x), int(center_point.y)), 3, (0, 0, 255), -1)

    if len(left_cones) > 0 and len(right_cones) > 0:
        center_x = int((left_cones[0].x + right_cones[0].x) / 2)
        center_y = int((left_cones[0].y + right_cones[0].y) / 2)
        
        center_x_avg.add_sample(center_x)
        center_y_avg.add_sample(center_y)
        
        avg_center_x = int(center_x_avg.get_wmm())
        avg_center_y = int(center_y_avg.get_wmm())
        
        center_point = (avg_center_x, avg_center_y)
        cv2.circle(image, center_point, 3, (255, 255, 255), -1)
        cv2.line(image, car_center , center_point, (255, 255, 255), line_thickness)
        steer = ((center_x - car_center_w) / (car_center_w * 0.5)) * 2000
    
    calc_steer(steer)

    # Display the image with lines
    cv2.imshow('Image with Lines', image)
    cv2.waitKey(1)        




# Subscribe to the BoundingBoxes_c topic
sub_boxes = rospy.Subscriber('/yolov5/detections_c', BoundingBoxes, bounding_boxes_callback)

# Subscribe to the compressed image topic
#sub_image = rospy.Subscriber('/usb_cam/image_raw/compressed', CompressedImage, image_callback)
sub_image = rospy.Subscriber('/zed/zed_node/rgb_raw/image_raw_color/compressed', CompressedImage, image_callback)

rospy.spin()

