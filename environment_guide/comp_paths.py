import pandas as pd
import matplotlib.pyplot as plt

def add_headers_to_file(file_path, headers, output_path):
    """
    Add headers to a CSV file without headers.

    Parameters:
        file_path (str): Path to the input CSV file without headers.
        headers (list): List of column headers to add.
        output_path (str): Path to save the updated CSV file.

    Returns:
        None
    """
    # Load the file without headers
    data = pd.read_csv(file_path, header=None)

    # Assign headers
    data.columns = headers

    # Save the updated file
    data.to_csv(output_path, index=False)

    print(f"Headers added and saved to {output_path}")

def compare_paths(file1, file2):
    """
    Compare two paths by plotting them.

    Parameters:
        file1 (str): Path to the first CSV file (e.g., Python-generated).
        file2 (str): Path to the second CSV file (e.g., MATLAB-generated).

    Returns:
        None
    """
    # Load the CSV files
    data1 = pd.read_csv(file1)
    data2 = pd.read_csv(file2)

    # Extract x and y coordinates
    x1, y1 = data1['x'], data1['y']
    x2, y2 = data2['x'], data2['y']

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot the paths
    plt.plot(x1, y1, label='Path from Python Script', linestyle='-', linewidth=2)
    plt.plot(x2, y2, label='Path from MATLAB Script', linestyle='--', linewidth=2)

    # Add labels and legend
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Comparison of Paths')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Define file paths
    file1 = "/home/asalvi/Downloads/test_paths/path_1_shifted.csv"  # Replace with actual file path
    file2 = "/home/asalvi/code_workspace/Husky_CS_SB3/PoseEnhancedVN/train/MixPathFlip/ArcPath1.csv"  # Replace with actual file path

    # Add headers to file2
    headers = ['x', 'y', 'heading', 'q1', 'q2', 'q3', 'q4', 'curvature', 'curvature_derivative', 'arc_length']
    output_file2 = "/home/asalvi/Downloads/test_paths/updated_matlab_path.csv"  # Replace with desired output file path
    add_headers_to_file(file2, headers, output_file2)

    # Compare the paths
    compare_paths(file1, output_file2)




