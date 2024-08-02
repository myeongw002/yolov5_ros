#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from detection_msgs.msg import BoundingBox, BoundingBoxes
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import cv2
from cv_bridge import CvBridge
import numpy as np
from scipy.spatial import Delaunay, ConvexHull, KDTree
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import traceback

class Cone_Detector:

    def __init__(self):
        self.K = np.array([1053.875392,   -0.958657,  944.171673,
            0.0     , 1054.496092  ,  568.170328,
            0.0     ,    0.0     ,    1.0     ]).reshape((3,3))
        self.plt = plt
        self.plt.ion()
        self.figure, self.ax = self.plt.subplots(figsize=(8,6))
        self.point = np.array([[0,0]])
        self.line, = self.ax.plot(self.point[:,0],self.point[:,1],'og')
        self.cv_bridge = CvBridge()
        self.yellow_cones = []
        self.blue_cones = []
        self.XYZ_yellow = np.array([])
        self.XYZ_blue = np.array([])
        self.img_center = None
        self.ros_topic_func()
        self.goal_point = Point()
        self.prev_yellow = []
        self.prev_blue = []
        
    def ros_topic_func(self):
        self.yellow_marker_publisher = rospy.Publisher("/path/yellow_marker", Marker, queue_size=10)
        self.blue_marker_publisher = rospy.Publisher("/path/blue_marker", Marker, queue_size=10)
        self.center_marker_publisher = rospy.Publisher("/path/center_paths_marker", Marker, queue_size=10)
        self.goal_point_pub = rospy.Publisher('/goal_point', Point, queue_size= 1)
        box_topic = rospy.get_param("~input_box_topic", "/yolov5/detections")
        self.sub_boxes = rospy.Subscriber(box_topic, BoundingBoxes, self.bounding_boxes_callback, queue_size=1)
        depth_image_topic = rospy.get_param("~depth_image_topic", 'usb_cam/image/raw')
        rospy.wait_for_message(box_topic, BoundingBoxes)
        
    def publish_marker(self, points, color):
        marker_msg = Marker()
        marker_msg.header.frame_id = "base_link"
        marker_msg.header.stamp = rospy.Time.now()
        marker_msg.ns = "interpolated_paths"
        marker_msg.id = 0
        marker_msg.type = Marker.LINE_STRIP
        marker_msg.action = Marker.ADD
        marker_msg.pose.orientation.w = 1.0
        marker_msg.scale.x = 0.1
        marker_msg.color.a = 1.0
        marker_msg.color.r = color[0]
        marker_msg.color.g = color[1]
        marker_msg.color.b = color[2]
        for point in points:
            p = Point()
            p.x = point[2]
            p.y = -point[0]
            p.z = 0
            marker_msg.points.append(p)
        return marker_msg


    def bounding_boxes_callback(self, msg):
        self.XYZ_yellow = self.process_cones(msg.bounding_boxes, "yellow_cone")
        self.XYZ_blue = self.process_cones(msg.bounding_boxes, "blue_cone")

    def process_cones(self, bounding_boxes, label):
        filtered_cones = self.filter_and_sort_cones(bounding_boxes, label)
        if filtered_cones:
            try:
                return self.pixel2meter(filtered_cones, 1.03)
            except Exception as e:
                rospy.logerr(f"Failed to convert {label} cones: {e}")
        return np.array([])

    def filter_and_sort_cones(self, bounding_boxes, label):
        filtered_cones = [box for box in bounding_boxes if box.Class == label]
        if not filtered_cones:
            return []

        filtered_cones.sort(key=lambda box: box.ymax, reverse=True)
        if len(filtered_cones) > 1:
            ref_xmax = filtered_cones[0].xmax
            filtered_cones.sort(key=lambda box: abs(ref_xmax - box.xmax), reverse=False)
        return filtered_cones[:4]




    def interpolate_path(self, XYZ_coordinates):
        #print("XYZ_coordinates shape:", XYZ_coordinates.shape)
        m = XYZ_coordinates.shape[0]
        if m == 1:
            return XYZ_coordinates
        elif m < 1:
            return np.array([])     # Not enough data points for interpolation
        if np.any(np.isnan(XYZ_coordinates)) or np.any(np.isinf(XYZ_coordinates)):
            print(XYZ_coordinates)
            return  # Contains NaN or Inf, cannot interpolate

        # Choose the degree of the spline based on the number of data points
        k = max(1, min(2, m - 1))  # Use k=2 for quadratic interpolation if enough data points, else use the highest degree possible
        
        # Perform spline interpolation
        if XYZ_coordinates.shape[1] == 3:
            tck, u = splprep([XYZ_coordinates[:, 0], XYZ_coordinates[:, 1], XYZ_coordinates[:, 2]], k=k, s=0)
            # Evaluate the interpolated path at desired intervals (e.g., 100 points along the path)
            u_new = np.linspace(0, 1, 10)
            interpolated_points = np.array(splev(u_new, tck)).T
            
        elif XYZ_coordinates.shape[1] == 2:
            tck, u = splprep([XYZ_coordinates[:, 0], XYZ_coordinates[:, 1]], k=k, s=0)
            # Evaluate the interpolated path at desired intervals (e.g., 100 points along the path)
            u_new = np.linspace(0, 1, 10)
            interpolated_points = np.array(splev(u_new, tck)).T
                    
        return interpolated_points



    def pixel2meter(self, cones, Y):
        center_array = np.empty((0, 3))
        for i in range(len(cones)):
            if cones[i].Class == "yellow_cone":
                xp = cones[i].xmax
                yp = cones[i].ymax
            else:
                xp = cones[i].xmin
                yp = cones[i].ymax
            center_array = np.append(center_array, np.array([[xp, yp, 1]]), axis=0)
        Z = self.K[1, 1] * Y / (center_array[:, 1] - self.K[1, 2])
        X = (center_array[:, 0] - self.K[0, 2]) / (self.K[0, 0]) * Z
        Y = np.ones(Z.shape[0]) * Y
        XYZ = np.column_stack((X, Y, Z))
        return XYZ

    def generate_virtual_line(self, actual_points, distance=1.0):
        virtual_points = np.copy(actual_points)
        if virtual_points.ndim == 2:
            virtual_points[:, 0] += distance
        else:
            pass
        return virtual_points

    def perform_delaunay_triangulation(self, points):
        if len(points) >= 3:
            tri = Delaunay(points)
            return tri
        return None

    def filter_exterior_triangles(self, points, tri, labels):
        hull = ConvexHull(points)
        hull_indices = hull.vertices
        mask = np.any(np.isin(tri.simplices, hull_indices), axis=1)

        same_label_mask = np.array([labels[triangle[0]] == labels[triangle[1]] == labels[triangle[2]] for triangle in tri.simplices])
        exterior_triangles = tri.simplices[mask & same_label_mask]

        interior_triangles = tri.simplices[~mask | ~same_label_mask]
        return interior_triangles, exterior_triangles


    def plot_cones(self, points, color):
        self.ax.plot(points[:, 0], points[:, 2], 'o', color=color)

    def plot_triangulation(self, points, tri):
        self.ax.triplot(points[:, 0], points[:, 1], tri.simplices, color='blue')

    def plot_midpoints(self, points, tri, labels):
        midpoints = []
        for triangle in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    p1, p2 = triangle[i], triangle[j]
                    if labels[p1] != labels[p2]:
                        midpoint = (points[p1] + points[p2]) / 2
                        midpoints.append(midpoint)
        
        midpoints = np.array(midpoints)
        midpoints = np.unique(midpoints, axis=0)
        

        if len(midpoints) > 1:
            # y축 오름차순으로 정렬
            midpoints = midpoints[np.argsort(midpoints[:, 1])]

            # 탐색배열과 결과 배열 생성 및 초기화
            remaining_midpoints = list(midpoints)
            result_midpoints = [remaining_midpoints.pop(0)]

            # 중점 연결
            while remaining_midpoints:
                current_point = result_midpoints[-1]
                distances = np.linalg.norm(remaining_midpoints - current_point, axis=1)
                nearest_index = np.argmin(distances)
                result_midpoints.append(remaining_midpoints.pop(nearest_index))

            # 결과 배열에 있는 점들끼리 선으로 연결
            result_midpoints = np.array(result_midpoints)
            result_midpoints = self.interpolate_path(result_midpoints)
            self.ax.plot(result_midpoints[:, 0], result_midpoints[:, 1], 'go-', markersize=5)

        return midpoints



    def main(self):
        virtual_distance = float(rospy.get_param("~distance", 2.2))
        rate = rospy.Rate(15)
        while not rospy.is_shutdown():
            try:
                self.ax.cla()
                if len(self.XYZ_yellow) > 0 and len(self.XYZ_blue) > 0:
                    yellow_first_cone = self.XYZ_yellow[0]
                    blue_first_cone = self.XYZ_blue[0]
                    distance = np.linalg.norm(yellow_first_cone - blue_first_cone)
                    if distance > 5:
                        print("Too far from first cone")
                        if np.linalg.norm(self.XYZ_yellow[0]) < np.linalg.norm(self.XYZ_blue[0]):
                            self.XYZ_blue = np.array([])
                        else:
                            self.XYZ_yellow = np.array([])

                # Remove cones farther than 5m from the first cone
                def filter_cones_by_distance(cones, first_cone, max_distance=5):
                    filtered_cones = []
                    for cone in cones:
                        if np.linalg.norm(cone - first_cone) <= max_distance:
                            filtered_cones.append(cone)
                    return np.array(filtered_cones)

                if len(self.XYZ_yellow) > 0:
                    self.XYZ_yellow = filter_cones_by_distance(self.XYZ_yellow, self.XYZ_yellow[0])

                if len(self.XYZ_blue) > 0:
                    self.XYZ_blue = filter_cones_by_distance(self.XYZ_blue, self.XYZ_blue[0])

                yellow_bias = rospy.get_param('~yellow_bias', 0.6)
                if self.XYZ_yellow.shape[0] != 0 and self.XYZ_blue.shape[0] != 0:
                    if self.XYZ_blue.shape[0] <= 2 and self.XYZ_yellow.shape[0] >= 2:
                        yellow_bias = yellow_bias + 0.1
                    elif self.XYZ_yellow.shape[0] <= 2 and self.XYZ_blue.shape[0] >= 2:
                        yellow_bias = 1 - (yellow_bias + 0.1)
                        
                    points = np.vstack((self.XYZ_yellow, self.XYZ_blue))
                    labels = np.array([0] * len(self.XYZ_yellow) + [1] * len(self.XYZ_blue))  # Set labels for the points
                    self.plot_cones(self.XYZ_yellow, 'yellow')
                    self.plot_cones(self.XYZ_blue, 'blue')
                elif self.XYZ_yellow.shape[0] == 0 and self.XYZ_blue.shape[0] != 0:
                    virtual_yellow = self.generate_virtual_line(self.XYZ_blue, -virtual_distance)
                    points = np.vstack((virtual_yellow, self.XYZ_blue))
                    labels = np.array([0] * len(virtual_yellow) + [1] * len(self.XYZ_blue))  # Set labels for the points
                    self.plot_cones(virtual_yellow, 'yellow')
                    self.plot_cones(self.XYZ_blue, 'blue')
                elif self.XYZ_yellow.shape[0] != 0 and self.XYZ_blue.shape[0] == 0:
                    virtual_blue = self.generate_virtual_line(self.XYZ_yellow, virtual_distance)
                    points = np.vstack((self.XYZ_yellow, virtual_blue))
                    labels = np.array([0] * len(self.XYZ_yellow) + [1] * len(virtual_blue))  # Set labels for the points
                    self.plot_cones(self.XYZ_yellow, 'yellow')
                    self.plot_cones(virtual_blue, 'blue')
                else:
                    points = np.array([[0, 0, 0]])
                    labels = np.array([0])  # Default label for a single point
                
                # Delaunay triangulation on all points
                if points.shape[0] > 3:
                    tri = self.perform_delaunay_triangulation(points[:, [0, 2]])
                    if tri is not None:
                        if len(labels) != points.shape[0]:
                            print("Error: Mismatch between labels and points length")
                            continue
                        interior_triangles, exterior_triangles = self.filter_exterior_triangles(points[:, [0, 2]], tri, labels)

                        #self.plot_triangulation(points[:, [0, 2]], tri)
                        self.plot_midpoints(points[:, [0, 2]], tri, labels)
                        self.ax.triplot(points[:, 0], points[:, 2], interior_triangles, color='red')  # Plot interior triangles only

                self.figure.canvas.draw()
                self.figure.canvas.flush_events()
                
                rate.sleep()
            except Exception as e:
                tb = traceback.format_exc()
                error_line = tb.split("\n")[-3]
                print(f"Error: {e} on {error_line}")
                rate.sleep()

if __name__ == '__main__':
    rospy.init_node('yolo_sub')
    cone_detector = Cone_Detector()
    cone_detector.main()
