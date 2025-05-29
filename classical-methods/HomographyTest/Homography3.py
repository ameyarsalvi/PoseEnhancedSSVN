# Perspective transformation
# Discrete points to curve >> To lane center path which will be the reference path
# MPC and pure-pursuit based on the refernce path

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


from coppeliasim_zmqremoteapi_client import RemoteAPIClient


## Variable initialization
sim_time = []


print('Program started')


client = RemoteAPIClient('localhost',23004)
sim = client.getObject('sim')

visionSensorHandle = sim.getObject('/Vision_sensor')
fl_w = sim.getObject('/flw')
fr_w = sim.getObject('/frw')
rr_w = sim.getObject('/rrw')
rl_w = sim.getObject('/rlw')
IMU = sim.getObject('/Accelerometer_forceSensor')
#COM = sim.getObject('/Husky/ReferenceFrame')
COM = sim.getObject('/Husky/Accelerometer/Accelerometer_mass')
Husky_ref = sim.getObject('/Husky')
#H1 = sim.getObject('/ConeH1')
#H2 = sim.getObject('/ConeH2')
#H3 = sim.getObject('/ConeH3')
#H4 = sim.getObject('/ConeH4')
#InertialFrame = sim.getObject('/InertialFrame')

defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
sim.setInt32Param(sim.intparam_idle_fps, 0)

# Run a simulation in stepping mode:
client.setStepping(True)
sim.startSimulation()

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

    x1 = 10.3349
    y1 = 22.8218
    x2 = 9.4099
    y2 = 22.1968
    x3 = 9.0394
    y3 = 23.0468
    x4 = 10.1599
    y4 = 23.6718


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


def control_wheels(im_bw):

    im_bw_ = cv2.bitwise_not(im_bw)

    # calculate moments of binary image
    M = cv2.moments(im_bw_)
 
    # calculate x,y coordinate of center
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    error = 128 - cX
    p_gain = -0.001*0

    V = 0.1
    omega = p_gain*error 

    A = np.array([[0.081675,0.081675],[-0.1081,0.1081]]) 
    velocity = np.array([[V],[omega]])
    phi_dots = np.matmul(inv(A),velocity)

    phi_dots = phi_dots.astype(float)
    Left = phi_dots[0].item()
    Right = phi_dots[1].item()
    


    sim.setJointTargetVelocity(fl_w, Left)
    sim.setJointTargetVelocity(fr_w, Right)
    sim.setJointTargetVelocity(rl_w, Left)
    sim.setJointTargetVelocity(rr_w, Right)

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

  

while (t:= sim.getSimulationTime()) < 600:
    #print(t)
    
    # IMAGE PROCESSING CODE ################

    img, resX, resY = sim.getVisionSensorCharImage(visionSensorHandle)
    img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
            # In CoppeliaSim images are left to right (x-axis), and bottom to top (y-axis)
            # (consistent with the axes of vision sensors, pointing Z outwards, Y up)
            # and color format is RGB triplets, whereas OpenCV uses BGR:
    img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0) #Convert to Grayscale

            # Current image
    cropped_image = img[288:480, 0:640] # Crop image to only to relevant path data (Done heuristically)
    im_bw = cv2.threshold(cropped_image, 125, 255, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
    #im_bw = cv2.bitwise_not(im_bw)
    cv2.imshow("Image", im_bw)

    control_wheels(im_bw)

    H = getHomography()

    print("Homography Matrix is", H)

    scale = 100

    # (5) Create warp canvas of size (3.5m x 5.0m) in pixels
    canvas_width = int(3.5 * scale)
    canvas_height = int(5.0 * scale)

    # (6) Warp
    warped = cv2.warpPerspective(im_bw, H, (canvas_width, canvas_height))
    warped = cv2.bitwise_not(warped)

    # (7) Show result
    cv2.imshow("Warped Top-Down View", warped)

    
    # --- ⬇️ Replace all centroid-based centerline code with this block ---


    # Apply function
    fitted_path, band_points = fit_quadratic_on_raw_image(im_bw)

    if fitted_path is not None:
        # Draw path
        for x, y in fitted_path.astype(int):
            if 0 <= x < im_bw.shape[1] and 0 <= y < im_bw.shape[0]:
                cv2.circle(im_bw, (x, y), 2, (200), -1)

        # Mark band centroids
        for cx, cy in band_points.astype(int):
            cv2.circle(im_bw, (cx, cy), 5, (127), -1)

        cv2.imshow("Quadratic Fit on Raw Image", im_bw)
        #cv2.waitKey(1)

    
    # Warp and overlay on top-down view
    if fitted_path is not None:
        warp_and_overlay_path(fitted_path, warped, H)

        # 3. Warp coordinates
        ones = np.ones((fitted_path.shape[0], 1))
        path_homog = np.hstack([fitted_path, ones])
        warped_path_h = (H @ path_homog.T).T
        warped_path_px = warped_path_h[:, :2] / warped_path_h[:, 2].reshape(-1, 1)

        # 4. Convert to meters
        waypoints_meters = warped_path_to_meters(warped_path_px, scale=100)
        print("Waypoints in meters:\n", waypoints_meters)
    

    # --- Step 2: Get the warped image dimensions ---
    height, width = warped.shape  # Or: warped.shape[:2]

    # --- Step 3: Convert to robot-relative coordinates in meters ---
    waypoints_meters = convert_pixel_path_to_waypoints_in_meters(
        pixel_path=fitted_path,
        warped_shape=(height, width),
        scale=100  # use the same scale as used in warp
    )


    # --- Step 4: (Optional) Visualize waypoints in robot frame ---
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(waypoints_meters[:, 0], waypoints_meters[:, 1], 'bo-', label="Waypoints (m)")
    plt.xlabel("Lateral offset (m)")
    plt.ylabel("Forward distance (m)")
    plt.title("Waypoints in Robot Frame")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()




    sim_time.append(t)
    print(t)

    client.step()  # triggers next simulation step

sim.stopSimulation()






