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
    
    
