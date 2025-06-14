import cv2
import numpy as np

class CalcHomography():
    
    def __init__(self):
        pass
        
        
    def calculate(self):
        # Load the image
        image = cv2.imread("/home/asalvi/Downloads/snapshot_2025_06_11_13_04_57.jpg")  # Replace "your_image.jpg" with the path to your image file

        # Define the new width and height
        new_width = 640
        new_height = 480

        # Resize the image to the specified dimensions
        resized_image = cv2.resize(image, (new_width, new_height))
        cropped_image = resized_image[288:480, 0:640]
        small_size_image = cv2.resize(cropped_image, (0,0), fx=0.5, fy=0.5)
        hsv = cv2.cvtColor(small_size_image, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([5,50,50])
        upper_orange = np.array([15,255,255])
        bw_img = cv2.inRange(hsv, lower_orange, upper_orange)

        contours, _ = cv2.findContours(bw_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 2: Get centroids of blobs
        centroids = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
        centroids = np.array(centroids)

        print(centroids)

        img_pts = np.array([
            [0, 85],
            [157, 92],
            [200, 33],
            [77, 25]
        ], dtype=np.float32)

        world_pts = np.array([
            [-0.84, 1.55],
            [0, 1.55],
            [0.26, 3.18],
            [-0.84, 3.18]
        ], dtype=np.float32)

        homo, _ = cv2.findHomography(img_pts, world_pts)

        return resized_image, bw_img, homo


if __name__ == "__main__":

    resized_image, bw_img, homo = CalcHomography().calculate()

    print(f"Homography Tranform is : {homo}")



    # Display the resized image (optional)
    #cv2.imshow("Resized Image", resized_image)
    #cv2.imshow("Cropped BW Image", bw_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()