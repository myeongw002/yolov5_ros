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
from scipy.interpolate import splprep, splev, interp1d
import math
import matplotlib.pyplot as plt
from visualization_msgs.msg import Marker



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
        self.ros_topic_func()
        
        
    def ros_topic_func(self):
        self.yellow_marker_publisher = rospy.Publisher("/path/yellow_marker", Marker, queue_size=10)
        self.blue_marker_publisher = rospy.Publisher("/path/blue_marker", Marker, queue_size=10)
        self.center_marker_publisher = rospy.Publisher("/path/center_paths_marker", Marker, queue_size=10)
        # Subscribe to the BoundingBoxes topic
        box_topic = rospy.get_param("~input_box_topic", "/yolov5/detections")
        self.sub_boxes = rospy.Subscriber(box_topic, BoundingBoxes, self.bounding_boxes_callback)
        # Subscribe to the depth image topic
        depth_image_topic = rospy.get_param("~depth_image_topic", 'usb_cam/image/raw')
        self.deth_image_sub = rospy.Subscriber(depth_image_topic, Image, self.depth_callback)
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
        self.yellow_cones.sort(key=lambda box: (box.ymax, box.xmax), reverse=True)
        self.blue_cones.sort(key=lambda box: (box.ymax, box.xmin), reverse=True)
        self.yellow_cones = self.yellow_cones[:4]
        self.blue_cones = self.blue_cones[:4]        



    def interpolate_path(self,XYZ_coordinates):
        #print("XYZ_coordinates shape:", XYZ_coordinates.shape)
        m = XYZ_coordinates.shape[0]
        if m == 1:
            return  XYZ_coordinates
        elif m < 1:
            return np.array([])     # Not enough data points for interpolation
        if np.any(np.isnan(XYZ_coordinates)) or np.any(np.isinf(XYZ_coordinates)):
            return  # Contains NaN or Inf, cannot interpolate

        # Choose the degree of the spline based on the number of data points
        k = max(1, min(2, m - 1))  # Use k=2 for quadratic interpolation if enough data points, else use the highest degree possible
        
        # Perform spline interpolation on the XZ coordinates
        
        tck, u = splprep([XYZ_coordinates[:, 0], XYZ_coordinates[:, 1], XYZ_coordinates[:, 2]], k=k, s=0)

        # Evaluate the interpolated path at desired intervals (e.g., 100 points along the path)
        u_new = np.linspace(0, 1, 5)
        interpolated_points = np.array(splev(u_new, tck)).T
        
        # Plot the interpolated path on the XZ plane
        self.ax.plot(interpolated_points[:, 0], interpolated_points[:, 2], '-r')
        
        return interpolated_points

        


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

        try:
            self.XYZ_yellow, center_yellow = self.cal_XYZ(self.yellow_cones, self.depth_image)
           
        except:
            pass
        try:
            self.XYZ_blue, center_blue = self.cal_XYZ(self.blue_cones, self.depth_image)
        except:
            pass
        

    def generate_virtual_line(self,actual_points, distance=1.0):
        """
        Generate a virtual line parallel to the actual_points and at a given distance.
        Assumes the cones are always to the left of the centerline.
        """
        virtual_points = np.copy(actual_points)

        if virtual_points.ndim == 2 :
            virtual_points[:, 0] += distance  # adjust the X-coordinates by the given distance
            virtual_points[:, 2] -= abs(distance)
            
        else:
            pass
              
        return virtual_points
    
    def calculate_centerline(self, interpolated_yellow, interpolated_blue, bias_yellow=0.5, bias_blue=0.5):
        
        centerline = bias_yellow * interpolated_yellow + bias_blue * interpolated_blue
        centerline /= (bias_yellow + bias_blue)
        
        return centerline
    
    
    def main(self):
        steer_avg = MovingAverage(5)
        distance = math.sqrt(2.2)
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            
            try:
                interpolated_yellow = self.interpolate_path(np.vstack((self.XYZ_yellow[:, 0],self.XYZ_yellow[:, 1], self.XYZ_yellow[:, 2])).T)
                interpolated_blue = self.interpolate_path(np.vstack((self.XYZ_blue[:, 0],self.XYZ_blue[:, 1], self.XYZ_blue[:, 2])).T)
                
                #print('yellow',interpolated_yellow.shape[0])
                #print('blue',interpolated_blue.shape[0])
                if interpolated_yellow.shape[0] != 0 and interpolated_blue.shape[0] != 0:
                    self.line = self.ax.plot(self.XYZ_blue[:,0],self.XYZ_blue[:,2],'ob')        
                    self.line = self.ax.plot(self.XYZ_yellow[:,0],self.XYZ_yellow[:,2],'oy')  
                    centerline = self.calculate_centerline(interpolated_yellow, interpolated_blue, 0.6, 0.4)  
                       
                elif interpolated_yellow.shape[0] == 0 and interpolated_blue.shape[0] != 0:                  
                    self.line = self.ax.plot(self.XYZ_blue[:,0],self.XYZ_blue[:,2],'ob')    
                    virtual_yellow = self.generate_virtual_line(interpolated_blue, -distance)
                    self.line = self.ax.plot(virtual_yellow[:,0],virtual_yellow[:,2],'oy')
                    interpolated_yellow = self.interpolate_path(virtual_yellow)
                    centerline = self.calculate_centerline(interpolated_yellow, interpolated_blue)
                    
                elif interpolated_yellow.shape[0] != 0  and interpolated_blue.shape[0] == 0:            
                    self.line = self.ax.plot(self.XYZ_yellow[:,0],self.XYZ_yellow[:,2],'oy')   
                    virtual_blue = self.generate_virtual_line(interpolated_yellow, distance)
                    self.line = self.ax.plot(virtual_blue[:,0],virtual_blue[:,2],'ob')
                    interpolated_blue = self.interpolate_path(virtual_blue)
                    centerline = self.calculate_centerline(interpolated_yellow, interpolated_blue)
                    
                else:   
                    continue
       
                self.line = self.ax.plot(centerline[:,0],centerline[:,2],'og')
                # Plot the centerline on the XZ plane
                self.ax.plot(centerline[:, 0], centerline[:, 2], '-g')
                yellow_maker = self.publish_marker(interpolated_yellow, (1.0, 1.0, 0.0))  
                # Red for interpolated yellow path
                blue_maker = self.publish_marker(interpolated_blue, (0.0, 0.0, 1.0))   
                # Blue for interpolated blue path
                center_marker = self.publish_marker(centerline, (0.0, 1.0, 0.0))          
                # Green for centerline                     
                self.yellow_marker_publisher.publish(yellow_maker)          
                self.blue_marker_publisher.publish(blue_maker)
                self.center_marker_publisher.publish(center_marker)
                
                x = centerline[0][2]
                y = centerline[0][0]
                #print(x,y)
                
                angle_rad = np.arctan2(y, x)
                angle_deg = np.degrees(angle_rad)
                steer_avg.calc_steer(angle_deg) 
                                 
                self.figure.canvas.draw()
                self.figure.canvas.flush_events()
                self.plt.cla()
                
            except Exception as e:
                #pass
                print(e)
                        
            rate.sleep()        
        

class MovingAverage:
    def __init__(self, n):
        self.samples = n
        self.data = []
        self.weights = list(range(1, n+1))
        self.state_pub = rospy.Publisher('set_state',setState , queue_size=10)
        self.max_speed = int(rospy.get_param('~max_speed',5))
        self.L = 1.04
        self.set_state = setState()
        
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

    def calculate_curvature(self, steer_angle):
        # Convert steering angle from degrees to radians
        steer_angle_rad = steer_angle * (math.pi / 180)
        
        curvature = 2 * math.sin(steer_angle_rad) / self.L

        return abs(curvature)

    def calculate_speed(self, curvature,Kf=0.5538):
        speed = self.max_speed * (1 - curvature * Kf)
        #print(speed)
        # Ensure speed is positive and doesn't exceed v_max
        speed = max(1, min(self.max_speed, speed))
        return speed


    def calc_steer(self,steer):
        
        #print(steer)
        self.add_sample(steer)
        self.set_state.set_gear = 0
        self.set_state.set_degree = self.get_wmm()
        #curvature = self.calculate_curvature(self.set_state.set_degree)        
        self.set_state.set_velocity = self.max_speed #self.calculate_speed(curvature, 0.3)
        
        self.state_pub.publish(self.set_state)        
        print(f'speed:{self.set_state.set_velocity}, steer:{steer}')



if __name__ == '__main__':
    rospy.init_node('yolo_sub')
    cone_detector = Cone_Detector()
    cone_detector.main()
        
            
