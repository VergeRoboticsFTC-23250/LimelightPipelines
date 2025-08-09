import cv2
import numpy as np

def runPipeline(image, llrobot):
    camera_matrix = np.array([
        [1221.445, 0.0, 637.226],
        [0, 1223.398, 502.549],
        [0, 0, 1]], dtype=np.float32)
    

    dist_coeffs = np.array([.177168, -0.457341, 0.000360, 0.002753, 0.178259], dtype=np.float32)

    undistort = cv2.undistort(image, camera_matrix, dist_coeffs)
    
    # Convert image to HSV
    img_hsv = cv2.cvtColor(undistort, cv2.COLOR_BGR2HSV)

    # Color threshold using your ranges
    img_threshold = cv2.inRange(img_hsv, (2, 220, 57), (25, 255, 190))

    # Find contours on the thresholded mask
    contours, _ = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largestContour = np.array([[]])
    llpython = [0,0,0,0,0,0,0,0]

    # Frame area
    frame_area = image.shape[0] * image.shape[1]

    # Filter contours by area using pixel min and percentage max
    MIN_AREA = 500
    MAX_AREA_PERCENT = 0.2  # e.g., max 50% of frame area
    max_area_pixels = frame_area * MAX_AREA_PERCENT

    contours = [c for c in contours if MIN_AREA <= cv2.contourArea(c) <= max_area_pixels]

    if contours:
        # Pick largest contour
        largestContour = max(contours, key=cv2.contourArea)

        # Draw largest contour and bounding box
        cv2.drawContours(image, [largestContour], -1, (255, 0, 0), 2)
        x, y, w, h = cv2.boundingRect(largestContour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)

        llpython = [1, x, y, w, h, 9, 8, 7]

    return largestContour, image, llpython
