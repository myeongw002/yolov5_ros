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
    




def cal_XYZ(cones, depth_image):
    depth_pixel_array = []
    center_array = np.empty((0,3))
    for i in range(len(cones)):
        xp = int((cones[i].xmax+cones[i].xmin)/2)
        yp = int((cones[i].ymax+cones[i].ymin)/2)
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
def callbackdepth(msg):
    global cv_bridge, XYZ_yellow, XYZ_blue
    depth_image = cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    #print(type(image))
    #print(image.shape) # (540, 960)
    
    try:
        XYZ_yellow, center_yellow = cal_XYZ(yellow_cones, depth_image)
    except:
        pass
    try:
        XYZ_blue, center_blue = cal_XYZ(blue_cones, depth_image)
    except:
        pass


    # depth_image = cv2.cvtColor(depth_image,cv2.COLOR_GRAY2BGR)
    # for i in range(center_yellow.shape[0]):
    #     depth_image = cv2.circle(depth_image, (int(center_yellow[i,0]),int(center_yellow[i,1])),
    #                              5,(0,124,255),-1)
    # for i in range(center_blue.shape[0]):
    #     depth_image = cv2.circle(depth_image, (int(center_blue[i,0]),int(center_blue[i,1])),
    #                              5,(255,0,0),-1)
    # cv2.imshow("depth",depth_image)
    # cv2.waitKey(1)
    # print("-----")
    


# Subscribe to the compressed image topic
# image_topic = rospy.get_param("~input_image_topic", "/usb_cam/image_raw/compressed")
# sub_image = rospy.Subscriber(image_topic, CompressedImage, image_callback)

if __name__ == '__main__':
    rospy.init_node('yolo_sub')
    rospy.wait_for_message("/yolov5/detections_c", BoundingBoxes)
    # Subscribe to the BoundingBoxes_c topic
    sub_boxes = rospy.Subscriber('/yolov5/detections_c', BoundingBoxes, bounding_boxes_callback)
    rospy.Subscriber("/zed/zed_node/depth/depth_registered", Image, callbackdepth)
    # plt.show()
    # rospy.spin()
    
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        # draw_cones(image, yellow_cones)
        # draw_cones(image, blue_cones)
        # draw_path(image, yellow_cones, blue_cones)
        # cv2.imshow('Image with Lines', image)
        # cv2.waitKey(1)

        # print("XYZ_yellow: ",XYZ_yellow.shape[0])
        # print("XYZ_yellow: ",XYZ_yellow.shape[0])
        if XYZ_yellow.shape[0] != 0 and XYZ_blue.shape[0] != 0:
            line = ax.plot(XYZ_yellow[:,0],XYZ_yellow[:,2],'oy')
            line = ax.plot(XYZ_blue[:,0],XYZ_blue[:,2],'ob')

            figure.canvas.draw()
            figure.canvas.flush_events()
            plt.cla()
        rate.sleep()
    
        





