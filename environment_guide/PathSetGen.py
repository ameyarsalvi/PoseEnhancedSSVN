import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from pyclothoids import Clothoid  # Ensure this library is installed in your environment
import os

def compute_arc_length(path):
    """
    Computes the cumulative arc length of a given 2D path.
    
    Parameters:
        path (numpy.ndarray): Nx2 array of (x, y) points.
    
    Returns:
        numpy.ndarray: Cumulative arc length.
    """
    distances = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
    return np.hstack(([0], np.cumsum(distances)))

def compute_quaternion(heading_angles):
    """
    Converts a list of heading angles (yaw) to quaternions.
    
    Parameters:
        heading_angles (numpy.ndarray): Array of yaw angles in radians.
    
    Returns:
        numpy.ndarray: Nx4 array of quaternions (q1, q2, q3, q4).
    """
    euler_angles = np.vstack((heading_angles, np.zeros_like(heading_angles), np.zeros_like(heading_angles))).T
    return R.from_euler('zyx', euler_angles).as_quat()

def generate_clothoid_paths(input_csv, output_dir, shift_amount=50, num_paths=5, num_points=1000):
    """
    Generates transformed paths with clothoid-based arc length parameterization and quaternion conversion.
    
    Parameters:
        input_csv (str): Path to the input CSV file containing the base path.
        output_dir (str): Directory to save the output CSV files.
        shift_amount (int): Number of rows to shift for each transformation.
        num_paths (int): Number of transformed paths to generate.
        num_points (int): Number of interpolated points per path.
    
    Returns:
        None
    """
    # Load the input CSV and extract the first two columns (assumed to be x_wp and y_wp)
    data = pd.read_csv(input_csv)
    base_path = data.iloc[:, :2].values  # Get the first two columns as a numpy array

    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    for i in range(num_paths):
        # Circular shift
        shifted_path = np.roll(base_path, -shift_amount * i, axis=0)
        # Flip the shifted path
        flipped_path = np.flip(shifted_path, axis=0)

        for variation, path in [("shifted", shifted_path), ("flipped", flipped_path)]:
            # Interpolate using clothoid fitting
            interpolated_path = []
            clothoid_arc_lengths = []
            for j in range(len(path) - 1):
                x0, y0 = path[j]
                x1, y1 = path[j + 1]

                # Fit a clothoid between consecutive points
                #clothoid = pyclothoids.G1Hermite()
                clothoid = Clothoid.G1Hermite(x0, y0, 0, x1, y1, 0)  # Assume 0 heading for simplicity

                # Get the length of the clothoid
                clothoid_length = clothoid.length

                # Sample points along the clothoid
                sampled_points = np.array(clothoid.SampleXY(num_points // (len(path) - 1)))  # Convert to NumPy array
                interpolated_path.extend(sampled_points[:, :2])  # Extract x, y from sampled points
                arc_lengths = np.linspace(0, clothoid.length, num_points // (len(path) - 1))
                clothoid_arc_lengths.extend(arc_lengths + (clothoid_arc_lengths[-1] if clothoid_arc_lengths else 0))

            interpolated_path = np.array(interpolated_path)
            clothoid_arc_lengths = np.array(clothoid_arc_lengths)

            # Ensure all arrays have consistent lengths
            min_length = min(
                len(interpolated_path), len(clothoid_arc_lengths)
            )

            interpolated_path = interpolated_path[:min_length]
            clothoid_arc_lengths = clothoid_arc_lengths[:min_length]

            # Compute heading angles (based on clothoid derivatives)
            delta_x = np.diff(interpolated_path[:, 0], prepend=interpolated_path[0, 0])
            delta_y = np.diff(interpolated_path[:, 1], prepend=interpolated_path[0, 1])
            heading_angles = np.arctan2(delta_y, delta_x)
            heading_angles = heading_angles[:min_length]

            # Convert heading to quaternions
            quaternions = compute_quaternion(heading_angles)

            # Compute curvature (based on clothoid properties)
            curvature = np.array([
                (1 / clothoid.Theta(s) if clothoid.Theta(s) != 0 else 0) for s in clothoid_arc_lengths
            ])
            curvature = curvature[:min_length]

            curvature_derivative = np.gradient(curvature, clothoid_arc_lengths)
            curvature_derivative = curvature_derivative[:min_length]

            # Combine results into the desired format
            output_data = np.hstack((
                interpolated_path,  # x, y
                heading_angles[:, None],  # heading (theta)
                quaternions[:min_length],  # q1, q2, q3, q4
                curvature[:, None],  # curvature (kappa)
                curvature_derivative[:, None],  # derivative of curvature (kappa')
                clothoid_arc_lengths[:, None]  # cumulative arc length (s)
            ))

            # Save to CSV
            output_filename = f"{output_dir}/path_{i+1}_{variation}.csv"
            pd.DataFrame(output_data, columns=[
                'x', 'y', 'heading', 'q1', 'q2', 'q3', 'q4', 'curvature', 'curvature_derivative', 'arc_length'
            ]).to_csv(output_filename, index=False)

    print(f"Generated {num_paths * 2} clothoid-based paths with arc length and quaternion data.")



if __name__ == "__main__":
    # Define input CSV file and output directory
    input_csv = "/home/asalvi/Downloads/path.csv"  # Replace with your actual input file path
    output_dir = "/home/asalvi/Downloads/test_paths/"  # Replace with your actual output directory path

    # Call the function
    generate_clothoid_paths(input_csv, output_dir)

