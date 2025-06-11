import time

import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy import savetxt
from numpy.linalg import inv
#from matplotlib.animation import FuncAnimation

from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans


class Image2Waypoints2():
    def __init__(self):
        "some variables"

    def fit_dual_quadratics_on_raw_image(im_bw, debug_visualize=True):
        """
        Fit two quadratic curves (left and right lane boundaries) from a binary top-view image.

        Returns:
            left_path   : np.ndarray of shape (N, 2) -- fitted left lane boundary pixels (x, y)
            right_path  : np.ndarray of shape (N, 2) -- fitted right lane boundary pixels (x, y)
            center_path : np.ndarray of shape (N, 2) -- average path between left and right
            vis_image   : image with the fitted curves drawn (optional)
        """
        H, W = im_bw.shape

        # Step 1: Invert and threshold for blob detection
        _, binary = cv2.threshold(im_bw, 125, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 2: Get centroids of blobs
        centroids = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
        centroids = np.array(centroids)

        if len(centroids) < 6:
            print("[Fit] Not enough blobs to form both boundaries.")
            return None, None, None, None

        # Step 3: Cluster into 2 groups (left/right)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(centroids)
        labels = kmeans.labels_

        cluster_0 = centroids[labels == 0]
        cluster_1 = centroids[labels == 1]

        # Label clusters as left/right based on mean x
        if np.mean(cluster_0[:, 0]) < np.mean(cluster_1[:, 0]):
            left_centroids = cluster_0
            right_centroids = cluster_1
        else:
            left_centroids = cluster_1
            right_centroids = cluster_0

        # Step 4: Fit quadratic for each group (x = f(y))
        def fit_path(pts):
            y = pts[:, 1]
            x = pts[:, 0]
            coeffs = np.polyfit(y, x, deg=2)
            poly = np.poly1d(coeffs)
            y_fit = np.linspace(0, H - 1, 100)
            x_fit = poly(y_fit)
            return np.column_stack((x_fit, y_fit))

        left_path = fit_path(left_centroids)
        right_path = fit_path(right_centroids)

        # Step 5: Compute center path (average of left and right)
        center_path = (left_path + right_path) / 2

        # Step 6: Visualization
        vis_img = cv2.cvtColor(im_bw, cv2.COLOR_GRAY2BGR)
        if debug_visualize:
            for x, y in left_path.astype(int):
                if 0 <= x < W and 0 <= y < H:
                    vis_img[int(y), int(x)] = (0, 255, 0)  # Green: left
            for x, y in right_path.astype(int):
                if 0 <= x < W and 0 <= y < H:
                    vis_img[int(y), int(x)] = (0, 0, 255)  # Red: right
            for x, y in center_path.astype(int):
                if 0 <= x < W and 0 <= y < H:
                    vis_img[int(y), int(x)] = (255, 255, 0)  # Cyan: centerline
            for cx, cy in centroids:
                cv2.circle(vis_img, (int(cx), int(cy)), 2, (255, 0, 0), -1)

        return left_path, right_path, center_path, vis_img
    

    

    def convert_pixel_path_to_waypoints(center_path_px, H_img_to_world):
        """
        Convert a centerline path from image pixels to metric waypoints using homography.

        Parameters:
            center_path_px    : np.ndarray of shape (N, 2) - image pixel path (x, y)
            H_img_to_world    : np.ndarray of shape (3, 3) - homography matrix (image to robot/world frame)

        Returns:
            waypoints_m       : np.ndarray of shape (N, 2) - metric coordinates (X, Y) in meters
        """
        '''
        N = center_path_px.shape[0]
        pts_homog = np.hstack([center_path_px, np.ones((N, 1))])  # (N, 3)
        pts_transformed = (H_img_to_world @ pts_homog.T).T        # Apply H: (N, 3)

        # Normalize homogeneous coordinates
        pts_transformed /= pts_transformed[:, 2:3]
        waypoints_m = pts_transformed[:, :2]

        return waypoints_m
        '''

        pts = center_path_px.reshape(-1, 1, 2).astype(np.float32)
        waypoints_m = cv2.perspectiveTransform(pts, H_img_to_world)
        return waypoints_m.reshape(-1, 2)

        
    
    

    def plot_waypoints(waypoints, title="Waypoints in Robot Frame"):
        """
        Visualizes waypoints in the robot's body frame (X: lateral, Y: forward).

        Parameters:
            waypoints : np.ndarray of shape (N, 2) - metric waypoints (X, Y)
            title     : str - title of the plot
        """
        x = waypoints[:, 0]
        y = waypoints[:, 1]

        plt.figure(figsize=(8, 6))
        plt.plot(x, y, marker='o', linestyle='-', color='blue', label='Waypoint Path')
        plt.scatter(x[0], y[0], color='green', label='Start')
        plt.scatter(x[-1], y[-1], color='red', label='End')
        plt.xlabel("X (meters) [Lateral]")
        plt.ylabel("Y (meters) [Forward]")
        plt.title(title)
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.show()


