import cv2
import numpy as np

def runPipeline(image, llrobot):
    camera_matrix = np.array([
        [1221.445, 0.0, 637.226],
        [0, 1223.398, 502.549],
        [0, 0, 1]], dtype=np.float32)
    
    dist_coeffs = np.array([.177168, -0.457341, 0.000360, 0.002753, 0.178259], dtype=np.float32)

    # 1. Undistort
    undistort = cv2.undistort(image, camera_matrix, dist_coeffs)

    # 2. Sobel filter
    gray = cv2.cvtColor(undistort, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobelx = cv2.convertScaleAbs(sobelx)
    abs_sobely = cv2.convertScaleAbs(sobely)
    sobel = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)
    sobel_bgr = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)

    # 3. HSV threshold
    img_hsv = cv2.cvtColor(undistort, cv2.COLOR_BGR2HSV)
    img_threshold = cv2.inRange(img_hsv, (5, 170, 65), (26, 255, 255))

    # 4. Morphology
    kernel = np.ones((5, 5), np.uint8)
    img_threshold = cv2.morphologyEx(img_threshold, cv2.MORPH_OPEN, kernel)
    img_threshold = cv2.morphologyEx(img_threshold, cv2.MORPH_CLOSE, kernel)
    threshold_bgr = cv2.cvtColor(img_threshold, cv2.COLOR_GRAY2BGR)

    # Pick what you want to see
    debug_view = sobel_bgr  # or threshold_bgr, or undistort

    # No contours found/returned
    largestContour = np.array([[]])
    llpython = [0,0,0,0,0,0,0,0]

    return largestContour, sobel, llpython
