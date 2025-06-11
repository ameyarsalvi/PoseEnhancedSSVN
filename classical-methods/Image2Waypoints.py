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



class Image2Waypoints():
    def __init__(self):
        "some variables"

        #Provide homography matrix or use the following

    def getHomography():
        ### calculate homography
        u1 = 176
        v1 = 122
        u2 = 234
        v2 = 58
        u3 = 452
        v3 = 65
        u4 = 502
        v4 = 161


        img_pts = np.array([
                [u1, v1],
                [u2, v2],
                [u3, v3],
                [u4, v4]
            ], dtype=np.float32)

            # Define corresponding points in top-view world coordinates (assuming planar surface)
    
        l = (3.5/640)
        m = (5/192)
        # (2) Define corresponding real-world points in meters
        # X: left-right (3.5m), Y: forward-backward (5.0m)
        world_pts_m = np.array([
            [l*176, m*122],   # corresponding to [176, 122]
            [l*234, m*58],   # corresponding to [234, 58]
            [l*452, m*65],   # corresponding to [452, 65]
            [l*502, m*161]    # corresponding to [502, 161]
        ], dtype=np.float32)

        # (3) Choose a scale — 100 pixels per meter
        scale = 100
        world_pts_px = world_pts_m * scale  # convert world coords to pixels

        # (4) Compute homography
        homo, _ = cv2.findHomography(img_pts, world_pts_px)

        return homo
    
    def refine_binary_with_fitted_curve(im_bw):
        """
        Detects all blob pixels, fits a smooth curve through them,
        and returns a refined binary image with only the curve drawn in white.

        Args:
            im_bw (np.ndarray): Original binary image (uint8, shape HxW)

        Returns:
            np.ndarray: Refined binary image with smooth path curve drawn
        """
        # Step 1: Invert image to detect dark blobs
        _, binary = cv2.threshold(im_bw, 125, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Step 2: Collect all points from all contours
        all_points = []
        for cnt in contours:
            if len(cnt) >= 5:  # skip very small/noisy contours
                all_points.extend(cnt.reshape(-1, 2))  # (N, 2)

        if len(all_points) < 5:
            print("[Refine] Not enough points to fit a curve.")
            return im_bw

        points = np.array(all_points)
        # Sort by y (from bottom to top of image)
        points = points[np.argsort(points[:, 1])]

        # Step 3: Fit smooth polynomial (e.g., degree-2 or 3)
        X = points[:, 1].reshape(-1, 1)  # y as independent variable
        y = points[:, 0]                 # x as dependent variable

        poly_model = make_pipeline(PolynomialFeatures(degree=2), RANSACRegressor())
        poly_model.fit(X, y)
        x_fit = poly_model.predict(X)

        # Step 4: Generate curve points
        fitted_curve = np.column_stack((x_fit.astype(np.int32), X.flatten().astype(np.int32)))

        # Step 5: Create new blank image and draw curve
        refined = np.zeros_like(im_bw)
        for pt in fitted_curve:
            x, y = pt
            if 0 <= x < refined.shape[1] and 0 <= y < refined.shape[0]:
                cv2.circle(refined, (x, y), 2, 255, -1)

        return refined
    
    def fit_quadratic_on_raw_image(im_bw):
        """
        Fit a quadratic curve to:
        - Bottom row center
        - Mean blob centroid from middle third
        - Mean blob centroid from top third
        """
        H, W = im_bw.shape

        # Step 1: Invert for blob detection
        _, binary = cv2.threshold(im_bw, 125, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 2: Get all blob centroids
        centroids = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))

        centroids = np.array(centroids)
        if len(centroids) < 2:
            print("[Fit] Not enough blobs found.")
            return None, None

        # Step 3: Fixed point at bottom center
        bottom_center = np.array([W // 2, H - 1])

        # Step 4: Slice remaining image into 2 bands
        band_points = [bottom_center]
        slice_height = H // 3

        for i in [1, 0]:  # Middle, then top
            y_start = i * slice_height
            y_end = (i + 1) * slice_height
            mask = (centroids[:, 1] >= y_start) & (centroids[:, 1] < y_end)
            points_in_band = centroids[mask]
            if len(points_in_band) > 0:
                mean_point = np.mean(points_in_band, axis=0)
                band_points.append(mean_point)

        if len(band_points) != 3:
            print("[Fit] Could not extract 3 usable points.")
            return None, None

        band_points = np.array(band_points)
        x = band_points[:, 0]
        y = band_points[:, 1]

        # Step 5: Fit x = a*y^2 + b*y + c
        coeffs = np.polyfit(y, x, deg=2)
        poly_func = np.poly1d(coeffs)

        y_fit = np.linspace(min(y), max(y), 100)
        x_fit = poly_func(y_fit)
        path_pixels = np.column_stack((x_fit, y_fit))

        return path_pixels, band_points
    

    

    def fit_dual_quadratics_on_raw_image(im_bw, debug_visualize=True):
        """
        Fit two quadratic curves (left and right lane boundaries) from a binary top-view image.

        Returns:
            left_path  : np.ndarray of shape (N, 2) -- fitted left lane boundary pixels (x, y)
            right_path : np.ndarray of shape (N, 2) -- fitted right lane boundary pixels (x, y)
            vis_image  : image with the fitted curves drawn (optional)
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
            return None, None, None

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
            y_fit = np.linspace(0, H-1, 100)
            x_fit = poly(y_fit)
            return np.column_stack((x_fit, y_fit))

        left_path = fit_path(left_centroids)
        right_path = fit_path(right_centroids)

        # Step 5: Visualization
        vis_img = cv2.cvtColor(im_bw, cv2.COLOR_GRAY2BGR)
        if debug_visualize:
            for x, y in left_path.astype(int):
                if 0 <= x < W and 0 <= y < H:
                    vis_img[int(y), int(x)] = (0, 255, 0)  # Green: left lane
            for x, y in right_path.astype(int):
                if 0 <= x < W and 0 <= y < H:
                    vis_img[int(y), int(x)] = (0, 0, 255)  # Red: right lane

            # Optional: draw centroids
            for cx, cy in centroids:
                cv2.circle(vis_img, (int(cx), int(cy)), 2, (255, 0, 0), -1)

        return left_path, right_path, vis_img



    def warp_and_overlay_path(original_path, warped_img, H):
        """
        Warps a list of 2D path points from original image using homography H
        and plots them on the warped image.

        Args:
            original_path (np.ndarray): Nx2 array of [x, y] path in original image.
            warped_img (np.ndarray): Warped binary image (grayscale or color).
            H (np.ndarray): 3x3 homography matrix.
        """
        if original_path is None or len(original_path) == 0:
            print("[WarpOverlay] No path to warp.")
            return

        # Convert to homogeneous coordinates
        ones = np.ones((original_path.shape[0], 1))
        path_homogeneous = np.hstack([original_path, ones])  # (N x 3)

        # Apply homography
        warped_points_h = (H @ path_homogeneous.T).T  # (N x 3)
        warped_points = warped_points_h[:, :2] / warped_points_h[:, 2].reshape(-1, 1)

        # Draw on warped image
        for x, y in warped_points.astype(int):
            if 0 <= x < warped_img.shape[1] and 0 <= y < warped_img.shape[0]:
                cv2.circle(warped_img, (x, y), 2, (127), -1)

        cv2.imshow("Warped Image with Projected Path", warped_img)
        cv2.waitKey(1)

    def warped_path_to_meters(warped_path_px, scale=100):
        """
        Converts pixel coordinates in warped image to real-world meters.

        Args:
            warped_path_px (np.ndarray): Nx2 array of [x, y] points in pixels.
            scale (float): Pixels per meter (default: 100).

        Returns:
            np.ndarray: Nx2 array of [x, y] in meters.
        """
        if warped_path_px is None or len(warped_path_px) == 0:
            print("[WarpedToMeters] Invalid input path.")
            return None

        return warped_path_px / scale

    def convert_pixel_path_to_waypoints_in_meters(pixel_path, warped_shape, scale=100):
        """
        Converts pixel coordinates of the path into meters, with the origin at the bottom center of the warped image.

        Args:
            pixel_path (np.ndarray): Nx2 array of pixel (x, y) coordinates in the warped image.
            warped_shape (tuple): Shape of the warped image (height, width).
            scale (float): Pixels per meter.

        Returns:
            np.ndarray: Nx2 array of [X, Y] waypoints in meters relative to robot's frame.
        """
        height, width = warped_shape

        origin_x_px = width // 2
        origin_y_px = height  # bottom row

        shifted_px = pixel_path - np.array([origin_x_px, origin_y_px])
        path_meters = shifted_px / scale

        # Flip y-axis so forward motion is +Y in meters
        path_meters[:, 1] *= -1

        return path_meters
    
    
