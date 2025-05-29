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


def fit_linear_path_from_warped(warped_img, scale=100):
    """
    Fits a linear centerline path directly from the warped binary image.
    Steps:
    1. Slice image row-wise.
    2. Compute blob centroids in each row (if present).
    3. Fit x = a*y + b.
    4. Return smoothed path in meters.
    """
    H, W = warped_img.shape
    num_rows = 30
    row_step = H // num_rows

    centroids = []

    for i in range(num_rows):
        row_start = i * row_step
        row_end = row_start + row_step

        row_slice = warped_img[row_start:row_end, :]
        moments = cv2.moments(row_slice)

        if moments["m00"] != 0:
            cX = int(moments["m10"] / moments["m00"])
            cY = row_start + int(moments["m01"] / moments["m00"])
            centroids.append((cX, cY))

    if len(centroids) < 4:
        print("[LinearFit] Too few valid centroids.")
        return None, None

    points = np.array(centroids)

    # Fit x = a*y + b
    x = points[:, 0]
    y = points[:, 1]
    coeffs = np.polyfit(y, x, deg=1)
    poly_func = np.poly1d(coeffs)

    # Evaluate over vertical extent
    y_vals = np.linspace(min(y), max(y), 100)
    x_vals = poly_func(y_vals)

    # Build pixel path
    pixel_path = np.column_stack((x_vals, y_vals))

    # Convert to meters
    path_meters = pixel_path / scale
    return path_meters, pixel_path


def fit_dual_linear_paths_from_warped(warped_img, scale=100):
    """
    Splits the warped binary image into left and right halves,
    fits a linear path to each side using row-wise centroid scan.
    Returns:
        - left_path_meters, right_path_meters
        - left_pixel_path, right_pixel_path
    """
    H, W = warped_img.shape
    mid_x = W // 2
    num_rows = 30
    row_step = H // num_rows

    left_centroids, right_centroids = [], []

    for i in range(num_rows):
        row_start = i * row_step
        row_end = row_start + row_step

        row_slice = warped_img[row_start:row_end, :]

        # Split row into left and right
        left_row = row_slice[:, :mid_x]
        right_row = row_slice[:, mid_x:]

        # LEFT
        M_left = cv2.moments(left_row)
        if M_left["m00"] != 0:
            cX = int(M_left["m10"] / M_left["m00"])
            cY = row_start + int(M_left["m01"] / M_left["m00"])
            left_centroids.append((cX, cY))

        # RIGHT
        M_right = cv2.moments(right_row)
        if M_right["m00"] != 0:
            cX = int(M_right["m10"] / M_right["m00"]) + mid_x  # shift X back
            cY = row_start + int(M_right["m01"] / M_right["m00"])
            right_centroids.append((cX, cY))

    paths = {}
    for side, centroids in zip(['left', 'right'], [left_centroids, right_centroids]):
        if len(centroids) < 4:
            print(f"[{side.upper()} LINEAR FIT] Too few centroids.")
            paths[side] = (None, None)
            continue

        pts = np.array(centroids)
        x, y = pts[:, 0], pts[:, 1]
        coeffs = np.polyfit(y, x, deg=1)
        poly_func = np.poly1d(coeffs)

        y_vals = np.linspace(min(y), max(y), 100)
        x_vals = poly_func(y_vals)
        pixel_path = np.column_stack((x_vals, y_vals))
        path_meters = pixel_path / scale

        paths[side] = (path_meters, pixel_path)

    return paths['left'][0], paths['right'][0], paths['left'][1], paths['right'][1]




def get_smoothed_centerline_from_warped(warped_img, n_slices=4, smoothing=10.0, scale=100, window_length=15, polyorder=1):
    import numpy as np
    import cv2
    from scipy.interpolate import splprep, splev
    from scipy.signal import savgol_filter

    height, width = warped_img.shape[:2]
    slice_height = height // n_slices

    centroids = []

    for i in range(n_slices):
        y_start = i * slice_height
        y_end = y_start + slice_height
        slice_img = warped_img[y_start:y_end, :]

        M = cv2.moments(slice_img, binaryImage=True)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            global_cY = y_start + cY
            centroids.append((cX, global_cY))

    if len(centroids) < 4:
        print("Too few lane centroids for fitting.")
        return None

    points = np.array(centroids)
    points = points[np.argsort(points[:, 1])]
    x, y = points[:, 0], points[:, 1]

    try:
        tck, _ = splprep([x, y], s=100)
        u_new = np.linspace(0, 1, 100)
        x_new, y_new = splev(u_new, tck)
        spline_path = np.column_stack((x_new, y_new))
    except Exception as e:
        print(f"Spline fit failed: {e}")
        return None

    x_smooth = savgol_filter(spline_path[:, 0], window_length, polyorder)
    y_smooth = savgol_filter(spline_path[:, 1], window_length, polyorder)

    smoothed_path = np.column_stack((x_smooth, y_smooth))
    path_meters = smoothed_path / scale

    return path_meters


def fit_quadratic_path(path_meters):
    """
    Fits x = a*y^2 + b*y + c through centerline path.
    Returns a callable polynomial function and fit coefficients.
    """
    x = path_meters[:, 0]
    y = path_meters[:, 1]

    coeffs = np.polyfit(y, x, deg=2)  # x = a*y^2 + b*y + c
    poly_func = np.poly1d(coeffs)
    return poly_func, coeffs


def fit_linear_path(path_meters):
    """
    Fits x = a*y + b through the centerline path (x vs y).
    Returns a callable polynomial and coefficients.
    """
    x = path_meters[:, 0]
    y = path_meters[:, 1]

    coeffs = np.polyfit(y, x, deg=1)  # linear fit
    poly_func = np.poly1d(coeffs)
    return poly_func, coeffs

def fit_quadratic_curve_split_top_bottom(warped, scale=100):
    """
    Splits warped image horizontally, finds centroids in top/bottom halves,
    adds a center-bottom anchor, fits a quadratic curve (x = f(y)),
    and returns the curve in both pixel and meter coordinates.
    """
    H, W = warped.shape
    mid_y = 1 - (H // 2)

    top_half = warped[:mid_y, :]
    bottom_half = warped[mid_y:, :]

    def find_centroid(binary_img, y_offset=0):
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"]) + y_offset
        return np.array([cX, cY])

    centroid_top = find_centroid(top_half, y_offset=0)
    centroid_bottom = find_centroid(bottom_half, y_offset=mid_y)

    if centroid_top is None or centroid_bottom is None:
        print("Could not find centroids in both halves.")
        return None, None

    # Fixed center-bottom anchor point
    center_bottom = np.array([W // 2, H - 1])

    # Combine 3 points
    points = np.vstack([centroid_top, centroid_bottom, center_bottom])
    y_vals = points[:, 1]
    x_vals = points[:, 0]

    # Fit quadratic: x = a*y^2 + b*y + c
    coeffs = np.polyfit(y_vals, x_vals, deg=2)
    poly_func = np.poly1d(coeffs)

    # Generate fitted path
    y_curve = np.linspace(min(y_vals), max(y_vals), 100)
    x_curve = poly_func(y_curve)
    pixel_path = np.column_stack((x_curve, y_curve))
    path_meters = pixel_path / scale

    return path_meters, pixel_path


def fit_quadratic_from_linear_samples(warped_img, scale=100):
    """
    Fits a quadratic curve using:
    - Bottom-center of image
    - 1/3 and 2/3 of fitted linear path
    Returns (path_meters, pixel_path)
    """
    H, W = warped_img.shape

    # Point 1: Bottom-center pixel
    pt1 = (W // 2, H - 1)

    # Fit a linear path
    path_meters, pixel_path = fit_linear_path_from_warped(warped_img, scale=scale)

    if pixel_path is None or len(pixel_path) < 3:
        print("[QuadraticFit] Not enough linear path points.")
        return None, None

    # Select 1/3 and 2/3 waypoints *along the path*
    pt2 = tuple(pixel_path[len(pixel_path) // 3])
    pt3 = tuple(pixel_path[2 * len(pixel_path) // 3])

    # Fit quadratic to (x, y)
    three_points = np.array([pt1, pt2, pt3])
    x = three_points[:, 0]
    y = three_points[:, 1]

    coeffs = np.polyfit(y, x, deg=2)  # x = a y² + b y + c
    poly_func = np.poly1d(coeffs)

    y_vals = np.linspace(min(y), max(y), 100)
    x_vals = poly_func(y_vals)
    pixel_path = np.column_stack((x_vals, y_vals))
    path_meters = pixel_path / scale

    return path_meters, pixel_path




    

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

    control_wheels(im_bw)


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

    
    # --- ⬇️ Replace all centroid-based centerline code with this block ---


    # --- Fit linear path ---
    path_meters, pixel_path = fit_quadratic_from_linear_samples(warped, scale=100)

    if pixel_path is not None:
        for x, y in pixel_path.astype(int):
            if 0 <= x < warped.shape[1] and 0 <= y < warped.shape[0]:
                cv2.circle(warped, (x, y), 2, (200), -1)

        cv2.imshow("Quadratic Path from 3 Points", warped)
        cv2.waitKey(1)
        '''
        # Optional: plot in meters
        x_m, y_m = path_meters[:, 0], path_meters[:, 1]
        plt.figure()
        plt.plot(x_m, y_m, 'b-', linewidth=2)
        plt.scatter(x_m, y_m, color='red')
        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.grid(True)
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("3-Point Quadratic Fit")
        plt.tight_layout()
        plt.show()

        '''
       



    '''
    # Compute lane centerline from warped binary image
    smoothed_centerline_meters = get_smoothed_centerline_from_warped(warped)
    print(smoothed_centerline_meters)

    
    if smoothed_centerline_meters is not None:
        x = smoothed_centerline_meters[:, 0]
        y = smoothed_centerline_meters[:, 1]

        # Plot in matplotlib
        plt.figure(figsize=(6, 8))
        plt.plot(x, y, 'b-', linewidth=2, label="Centerline (meters)")
        plt.scatter(x, y, color='red', s=10, label="Waypoints")
        plt.gca().invert_yaxis()
        plt.xlabel("Lateral position (m)")
        plt.ylabel("Forward position (m)")
        plt.title("Fitted Centerline in Meters")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Visualize centerline in warped image (in pixels)
        #path_pixels = (smoothed_centerline_meters * 100).astype(int)
        #for x, y in path_pixels:
        #    cv2.circle(warped, (x, y), 2, (200), -1)

    #cv2.imshow("Fitted Centerline", warped)
    
    
    
    cv2.waitKey(1)

    x = smoothed_centerline_meters[:, 0]
    y = smoothed_centerline_meters[:, 1]

    # Plot in matplotlib
    plt.figure(figsize=(6, 8))
    plt.plot(x, y, 'b-', linewidth=2, label="Centerline (meters)")
    plt.scatter(x, y, color='red', s=10, label="Waypoints")
    plt.gca().invert_yaxis()
    plt.xlabel("Lateral position (m)")
    plt.ylabel("Forward position (m)")
    plt.title("Fitted Centerline in Meters")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()



    smoothed_path = smoothed_centerline_meters
    
    poly_func, coeffs = fit_quadratic_path(smoothed_path)

    # Sample over the range of y
    y_vals = np.linspace(smoothed_path[:, 1].min(), smoothed_path[:, 1].max(), 100)
    x_vals = poly_func(y_vals)

    fitted_poly_path = np.column_stack((x_vals, y_vals))
    

    poly_func, coeffs = fit_linear_path(smoothed_path)

    # Sample over the y-range
    y_vals = np.linspace(smoothed_path[:, 1].min(), smoothed_path[:, 1].max(), 100)
    x_vals = poly_func(y_vals)

    fitted_poly_path = np.column_stack((x_vals, y_vals))  # still in meters


    # Plot
    plt.figure()
    plt.plot(smoothed_path[:, 0], smoothed_path[:, 1], 'r--', label='Original Path')
    plt.plot(fitted_poly_path[:, 0], fitted_poly_path[:, 1], 'b-', label='Quadratic Fit')
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid()
    plt.legend()
    plt.title("Quadratic Fit to Centerline")
    plt.tight_layout()
    #plt.show()

    # Assuming: scale = 100  # pixels per meter
    fitted_poly_pixels = (fitted_poly_path * scale).astype(int)

    # Draw on warped image
    for x, y in fitted_poly_pixels:
        if 0 <= x < warped.shape[1] and 0 <= y < warped.shape[0]:  # bounds check
            cv2.circle(warped, (x, y), 2, (200), -1)  # gray intensity 200

    # Show image with overlay
    cv2.imshow("Top-Down + Fitted Poly Path", warped)
    cv2.waitKey(1)

'''

    sim_time.append(t)
    print(t)

    client.step()  # triggers next simulation step

sim.stopSimulation()






