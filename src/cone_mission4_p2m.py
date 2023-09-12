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
        #self.deth_image_sub = rospy.Subscriber(depth_image_topic, Image, self.depth_callback, queue_size=1)
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
        
        
        try:
            self.yellow_cones.sort(key=lambda box: box.ymax, reverse=True)
            self.yellow_cones.sort(key=lambda box: abs(self.yellow_cones[0].xmax-box.xmax), reverse=False)
                     
        except:
            pass
            
        try:
            self.XYZ_yellow = self.pixel2meter(self.yellow_cones, 1.03)
        except:
            pass
            
        try:
            self.blue_cones.sort(key=lambda box: box.ymax, reverse=True)
            self.blue_cones.sort(key=lambda box: abs(self.blue_cones[0].xmax-box.xmax), reverse=False)
        except:
            pass
        try:
            self.XYZ_blue = self.pixel2meter(self.blue_cones, 1.03)
        except:
            pass   
        #print('detected yellow',len(self.yellow_cones))
        #print('detected blue',len(self.blue_cones))




    def interpolate_path(self,XYZ_coordinates):
        #print("XYZ_coordinates shape:", XYZ_coordinates.shape)
        m = XYZ_coordinates.shape[0]
        if m == 1:
            return  XYZ_coordinates
        elif m < 1:
            return np.array([])     # Not enough data points for interpolation
        if np.any(np.isnan(XYZ_coordinates)) or np.any(np.isinf(XYZ_coordinates)):
            print(XYZ_coordinates)
            return  # Contains NaN or Inf, cannot interpolate

        # Choose the degree of the spline based on the number of data points
        k = max(1, min(2, m - 1))  # Use k=2 for quadratic interpolation if enough data points, else use the highest degree possible
        
        # Perform spline interpolation on the XZ coordinates
        
        tck, u = splprep([XYZ_coordinates[:, 0], XYZ_coordinates[:, 1], XYZ_coordinates[:, 2]], k=k, s=0)

        # Evaluate the interpolated path at desired intervals (e.g., 100 points along the path)
        u_new = np.linspace(0, 1, 10)
        interpolated_points = np.array(splev(u_new, tck)).T
        
        # Plot the interpolated path on the XZ plane
        self.ax.plot(interpolated_points[:, 0], interpolated_points[:, 2], '-r')
        
        return interpolated_points

            
    def poly_lane(self, XYZ, degree=2, num_points=5):
        """
        Applies polynomial regression on the XYZ data and generates lane points using the polynomial.

        Parameters:
        - XYZ: Input data points.
        - degree: Degree of the polynomial.
        - num_points: Number of points to generate for the lane.

        Returns:
        - XYZ_poly: Generated [X, Y, Z] values using the polynomial.
        """
        if XYZ.shape[0] < degree+1 :
            if XYZ.shape[0] == 1:
                return XYZ
            degree = 1
        X = XYZ[:, 0]
        Z = XYZ[:, 2]

        # Polynomial fit
        coefficients = np.polyfit(X, Z, degree)

        # Generate lane points
        X_range = np.linspace(min(X), max(X), num_points)
        Z_pred = np.polyval(coefficients, X_range)
        Y = np.zeros_like(X_range)  # Assuming Y values are not being used or are zero.
        XYZ_poly = np.column_stack((X_range, Y, Z_pred))
        self.ax.plot(XYZ_poly[:, 0], XYZ_poly[:, 2], '-r')
        return XYZ_poly
        
        
        
        
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
        Y = np.ones(Z.shape[0]) * Y
        XYZ = np.column_stack((X, Y, Z))
        return XYZ



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

    '''
    def depth_callback(self,msg):
        self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.img_center = self.depth_image.shape[1] / 2
        
        try:
            self.XYZ_yellow, yellow_center = self.cal_XYZ(self.yellow_cones, self.depth_image)

            #print('len', len(self.yellow_cones))
            #print('rows', self.XYZ_yellow.shape[0])
            #print('calc yellow',len(yellow_center))
        except:
            pass
            
        try:
            self.XYZ_blue, blue_center = self.cal_XYZ(self.blue_cones, self.depth_image)
            #print('calc blue',len(blue_center))
        except:
            pass
      '''
      
      
    def lane_regression(self, XYZ, degree=2):
        # Check if the number of points is below the threshold
        if XYZ.shape[0] < degree + 1:
            self.ax.plot(XYZ[:, 0], XYZ[:, 2], '-r')
            return XYZ
        X = XYZ[:, 0].reshape(-1, 1)
        Z = XYZ[:, 2]
        
        # Transform data to allow polynomial regression
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        
        # Apply regression model
        model = LinearRegression().fit(X_poly, Z)
        
        # Predict Z values
        Z_pred = model.predict(X_poly)
        XYZ_regression = np.column_stack((X, XYZ[:, 1], Z_pred))
        self.ax.plot(XYZ_regression[:, 0], XYZ_regression[:, 2], '-r')
        # Return the predicted XYZ coordinates
        return XYZ_regression
      




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
    
    
    
    
    def calculate_centerline(self, yellow_lane, blue_lane, bias_yellow=0.5):
        # 두 레인의 길이를 가져옵니다.
        len_yellow = len(yellow_lane)
        len_blue = len(blue_lane)

        # 더 작은 길이를 기준으로 중앙선을 계산합니다.
        min_len = min(len_yellow, len_blue)

        # 두 레인의 길이를 최소 길이로 조정합니다.
        yellow_lane = yellow_lane[:min_len]
        blue_lane = blue_lane[:min_len]

        # 중앙선을 계산합니다.
        centerline = (bias_yellow * yellow_lane) + ((1-bias_yellow) * blue_lane)

        return centerline
    
    
    
    def main(self):
        steer_avg = MovingAverage(5)
        distance = float(rospy.get_param("~distance", 2.2))
        distance = math.sqrt(distance)
        rate = rospy.Rate(15)
        while not rospy.is_shutdown():
            try:
                if len(self.XYZ_yellow) > 0 and len(self.XYZ_blue) > 0:
                    yellow_first_cone = self.XYZ_yellow[0]
                    blue_first_cone = self.XYZ_blue[0]

                    distance = np.linalg.norm(yellow_first_cone - blue_first_cone)

                    if distance > 5:
                        print("The first yellow and blue cones are more than 5 meters apart!")
                        if np.linalg.norm(self.XYZ_yellow[0]) < np.linalg.norm(self.XYZ_blue[0]):
                            self.XYZ_blue = self.generate_virtual_line(self.XYZ_yellow, distance)
                        else:
                            self.XYZ_yellow = self.generate_virtual_line(self.XYZ_blue, -distance)    
                if self.XYZ_yellow.shape[0] != 0 and self.XYZ_blue.shape[0] != 0:
                    self.line =self. ax.plot(self.XYZ_yellow[:,0], self.XYZ_yellow[:,2],'oy')
                    self.line = self.ax.plot(self.XYZ_blue[:,0], self.XYZ_blue[:,2],'ob')
                    yellow_lane = self.interpolate_path(self.XYZ_yellow)
                    blue_lane = self.interpolate_path(self.XYZ_blue)
                    centerline = self.calculate_centerline(yellow_lane, blue_lane,0.6)
                    
                elif self.XYZ_yellow.shape[0] == 0 and self.XYZ_blue.shape[0] != 0:
                    self.line = self.ax.plot(self.XYZ_blue[:,0], self.XYZ_blue[:,2],'ob')
                    virtual_yellow = self.generate_virtual_line(self.XYZ_blue, -distance)
                    self.line =self.ax.plot(virtual_yellow[:,0], virtual_yellow[:,2],'oy')
                    yellow_lane = self.interpolate_path(virtual_yellow)
                    blue_lane = self.interpolate_path(self.XYZ_blue)
                    centerline = self.calculate_centerline(yellow_lane, blue_lane)
                    
                elif self.XYZ_yellow.shape[0] != 0 and self.XYZ_blue.shape[0] == 0:
                    self.line = self.ax.plot(self.XYZ_yellow[:,0], self.XYZ_yellow[:,2],'oy')
                    virtual_blue = self.generate_virtual_line(self.XYZ_yellow, distance)
                    self.line = self.ax.plot(virtual_blue[:,0], virtual_blue[:,2],'ob')
                    yellow_lane = self.interpolate_path(self.XYZ_yellow)
                    blue_lane = self.interpolate_path(virtual_blue)
                    centerline = self.calculate_centerline(yellow_lane, blue_lane)
                                    
                else:
                    yellow_lane = np.array([[1.0, 1.0, 3.0]])
                    #print(interpolated_yellow)
                    self.line = self.ax.plot(yellow_lane[:,0], yellow_lane[:,2],'oy')
                    blue_lane = self.generate_virtual_line(yellow_lane, distance)
                    self.line = self.ax.plot(blue_lane[:,0], blue_lane[:,2],'ob')
                    centerline = self.calculate_centerline(yellow_lane, blue_lane)
       
      
                
                self.line = self.ax.plot(centerline[:,0],centerline[:,2],'og')
                # Plot the centerline on the XZ plane
                self.ax.plot(centerline[:, 0], centerline[:, 2], '-g')
                yellow_maker = self.publish_marker(yellow_lane, (1.0, 1.0, 0.0))  
                # Red for interpolated yellow path
                blue_maker = self.publish_marker(blue_lane, (0.0, 0.0, 1.0))   
                # Blue for interpolated blue path
                center_marker = self.publish_marker(centerline, (0.0, 1.0, 0.0))          
                # Green for centerline                     
                self.yellow_marker_publisher.publish(yellow_maker)          
                self.blue_marker_publisher.publish(blue_maker)
                self.center_marker_publisher.publish(center_marker)
                #print(centerline)
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
                tb = traceback.format_exc()  # Get the traceback information
                error_line = tb.split("\n")[-3]  # Extract the error line from the traceback
                print(f"Error: {e} on {error_line}")
                        
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
        rospy.loginfo(f'speed:{self.set_state.set_velocity}, steer:{steer}')



if __name__ == '__main__':
    rospy.init_node('yolo_sub')
    cone_detector = Cone_Detector()
    cone_detector.main()
        
            
