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

def fit_ransac_curve(points, poly_order=2):
        if len(points) < 3:
            return None

        X = points[:, 1].reshape(-1, 1)  # y as independent
        y = points[:, 0]                # x as dependent

        model = make_pipeline(PolynomialFeatures(poly_order), RANSACRegressor())
        model.fit(X, y)

        y_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        x_fit = model.predict(y_fit)

        return np.column_stack((x_fit, y_fit.flatten()))

from sklearn.linear_model import LinearRegression

def fit_linear_curve(points, poly_order=2):
    if len(points) < poly_order + 1:
        print(f"Too few points ({len(points)}) to fit polynomial of order {poly_order}.")
        return None

    X = points[:, 1].reshape(-1, 1)  # y as independent
    y = points[:, 0]                # x as dependent

    model = make_pipeline(PolynomialFeatures(poly_order), LinearRegression())

    try:
        model.fit(X, y)
    except Exception as e:
        print(f"Linear fit failed: {e}")
        return None

    y_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    x_fit = model.predict(y_fit)

    return np.column_stack((x_fit, y_fit.flatten()))


from scipy.optimize import curve_fit

def fit_nonlinear_curve(points, model_func, p0=None):
    if len(points) < 4:
        print("Too few points for non-linear regression.")
        return None

    y_data = points[:, 1]
    x_data = points[:, 0]

    try:
        popt, _ = curve_fit(model_func, y_data, x_data, p0=p0)
        y_fit = np.linspace(y_data.min(), y_data.max(), 100)
        x_fit = model_func(y_fit, *popt)
        return np.column_stack((x_fit, y_fit))
    except RuntimeError as e:
        print(f"Curve fitting failed: {e}")
        return None
    

import numpy as np
from scipy.interpolate import splprep, splev

def fit_smoothed_spline(points, smoothing=20.0, num_points=100):
    """
    Fits a smooth centerline to 2D points using B-spline with smoothing.

    Args:
        points (np.ndarray): Nx2 array of [x, y] points (e.g., centroids).
        smoothing (float): Smoothing factor (larger = smoother).
        num_points (int): Number of output points along the fitted curve.

    Returns:
        np.ndarray: Smoothed path (num_points x 2) of [x, y] points.
    """
    if len(points) < 4:
        print("Too few points to fit spline.")
        return None

    # Sort by y (forward direction) for more stability
    points = points[np.argsort(points[:, 1])]
    x, y = points[:, 0], points[:, 1]

    try:
        # Fit parametric spline to [x(t), y(t)] with smoothing
        tck, _ = splprep([x, y], s=smoothing)
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        return np.column_stack((x_new, y_new))
    except Exception as e:
        print(f"Spline fit failed: {e}")
        return None


from scipy.signal import savgol_filter

def smooth_centerline(center_path, window_length=11, polyorder=3):
    """
    Applies Savitzky-Golay filter to smooth a centerline path.
    Args:
        center_path (np.ndarray): Nx2 array of [x, y] path.
        window_length (int): Must be odd, controls smoothing strength.
        polyorder (int): Polynomial order for local fitting.
    Returns:
        np.ndarray: Smoothed centerline.
    """
    x_smooth = savgol_filter(center_path[:, 0], window_length, polyorder)
    y_smooth = savgol_filter(center_path[:, 1], window_length, polyorder)
    return np.column_stack((x_smooth, y_smooth))

    

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
    cv2.imshow("Image", im_bw)

    '''
    # display the image
    #cv2.imshow("Image", im_bw)


    # Invert image: cones are black (0), background is white (255)
    _, binary = cv2.threshold(cropped_image, 125, 255, cv2.THRESH_BINARY_INV)

    # Find contours (external only)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create output image for visualization
    output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))
            cv2.circle(output, (cX, cY), 5, (0, 0, 255), -1)

    # Optional: print cone coordinates
    print("Centroid Coordinates:", centroids)

    # Show result
    plt.imshow(output)
    plt.title("Detected Cone Centroids")
    plt.axis('off')
    plt.show()


    #cv2.waitKey(1)

    H1_pose = sim.getObjectPose(H1, sim.handle_world)
    print('H1 is', H1_pose)
    H2_pose = sim.getObjectPose(H2, sim.handle_world)
    print('H2 is', H2_pose)
    H3_pose = sim.getObjectPose(H3, sim.handle_world)
    print('H3 is', H3_pose)
    H4_pose = sim.getObjectPose(H4, sim.handle_world)
    print('H4 is', H4_pose)

'''
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
    '''
    world_pts = np.array([
            [x1, y1],
            [x2, y2],
            [x3, y3],
            [x4, y4]
        ], dtype=np.float32)
    '''
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
    H, _ = cv2.findHomography(img_pts, world_pts_px)

    print("Homography Matrix is", H)

    # (5) Create warp canvas of size (3.5m x 5.0m) in pixels
    canvas_width = int(3.5 * scale)
    canvas_height = int(5.0 * scale)

    # (6) Warp
    warped = cv2.warpPerspective(im_bw, H, (canvas_width, canvas_height))
    warped = cv2.bitwise_not(warped)

    # (7) Show result
    cv2.imshow("Warped Top-Down View", warped)
    #cv2.waitKey(1)

    # Find contours in the warped binary image
    contours, _ = cv2.findContours(warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))

    # Convert to numpy array
    if len(centroids) < 2:
        print("Too few blobs to fit RANSAC.")
        client.step()
        continue

    points = np.array(centroids)

    # Optional: skip if too few points
    if len(points) < 4:
        print("Too few centroids")
        client.step()
        continue

    mid_x = warped.shape[1] // 2  # vertical image center

    left_points = points[points[:, 0] < mid_x]
    right_points = points[points[:, 0] >= mid_x]

    # Polynomial order for the curve
    poly_order = 2

    #path_left = fit_ransac_curve(left_points, poly_order=2)
    #path_right = fit_ransac_curve(right_points, poly_order=2)
    '''
    path_left = fit_linear_curve(left_points, poly_order=2)
    path_right = fit_linear_curve(right_points, poly_order=2)


    if path_left is not None and path_right is not None:
        min_len = min(len(path_left), len(path_right))
        center_path = (path_left[:min_len] + path_right[:min_len]) / 2
    else:
        center_path = path_left if path_left is not None else path_right

    
    for x, y in path_left.astype(int):
        cv2.circle(warped, (x, y), 2, (255), -1)

    for x, y in path_right.astype(int):
        cv2.circle(warped, (x, y), 2, (127), -1)

    for x, y in center_path.astype(int):
        cv2.circle(warped, (x, y), 2, (200), -1)
    '''
    

    # Sort centroids and fit to clothoid-style curve or spline
    centroids_array = np.array(centroids)

    # Option 1: Nonlinear regression (e.g., clothoid approx)
    # path_center = fit_nonlinear_curve(centroids_array, clothoid_approx)

    # Option 2: CubicSpline
    path_center = fit_smoothed_spline(points, smoothing=30.0)

    if path_center is not None:
        for x, y in path_center.astype(int):
            cv2.circle(warped, (x, y), 2, (200), -1)
    
    
    center_path_meters = path_center / 100
    print(center_path_meters)

    import matplotlib.pyplot as plt
    import numpy as np

    # Assume you already have:
    # center_path_meters = path_center / 100

    # Extract x and y
    x = center_path_meters[:, 0]
    y = center_path_meters[:, 1]

    # Plot
    plt.figure(figsize=(6, 8))
    plt.plot(x, y, 'b-', linewidth=2, label="Centerline (meters)")
    plt.scatter(x, y, color='red', s=10, label="Waypoints")
    plt.gca().invert_yaxis()  # Optional: matches image view if needed
    plt.xlabel("Lateral position (m)")
    plt.ylabel("Forward position (m)")
    plt.title("Fitted Centerline in Meters")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.show()


    smoothed_path = smooth_centerline(center_path_meters, window_length=15, polyorder=3)

    # Plot
    plt.figure()
    plt.plot(center_path_meters[:, 0], center_path_meters[:, 1], 'r--', label='Raw center path')
    plt.plot(smoothed_path[:, 0], smoothed_path[:, 1], 'b-', linewidth=2, label='Smoothed centerline')
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True)
    plt.title("Smoothed Centerline")
    plt.legend()
    plt.tight_layout()
    plt.show()




    cv2.imshow("Fitted Lane Curves", warped)
    cv2.waitKey(1)





    '''

    # Independent: forward direction (y), Dependent: lateral offset (x)
    X = points[:, 1].reshape(-1, 1)  # y (vertical direction in image)
    y = points[:, 0]  # x (horizontal direction in image)

    # Fit RANSAC model
    model = make_pipeline(PolynomialFeatures(poly_order), RANSACRegressor())
    model.fit(X, y)

    # Predict over a range
    y_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    x_fit = model.predict(y_fit)

    # Combine into fitted path
    fitted_path = np.column_stack((x_fit, y_fit))

    for x, y in fitted_path.astype(int):
        cv2.circle(warped, (x, y), 2, (127), -1)

    cv2.imshow("RANSAC Fit", warped)

    '''
    
    #cv2.waitKey(1)
    sim_time.append(t)
    print(t)

    client.step()  # triggers next simulation step

sim.stopSimulation()






