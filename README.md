# FootBall-Player-Detection

NO YOLO, NO TENSORFLOW, NO PYTORCH, ONLY OPENCV AND NUMPY 🖥️ Though we have become very advanced in all the approaches but its important to stick to your grounds, Pure computer vision-based image processing can give you a lot faster outputs and usable accuracy for less complex problems. 🛠️ The video demonstrates the below image processing techniques to detect players in an environment 🎥

Here's a detailed summary of what the code does and how it distinguishes between players of different teams:

- Video Capture and Processing: The code begins by importing necessary libraries and reading a video file frame by frame using OpenCV (cv::VideoCapture).

- Color Detection: It converts each frame of the video from BGR to HSV color space, which is more suitable for color detection. Defines ranges for four colors: green (for the field), blue (for France team jersey), red (for Belgium team jersey), and white (for the football). Creates masks for each color range using cv::inRange. Applies bitwise AND operations between the frame and each mask to extract regions of interest (ROIs) containing the specified colors.

- Preprocessing: Converts the resulting green ROI image to grayscale. Applies thresholding and morphological operations to obtain a binary image highlighting the players and football.

- Contour Detection: Finds contours in the binary image using cv2.findContours. Iterates through each contour and checks its dimensions to determine if it represents a player or the football.

- Player Classification: For contours representing players, it calculates the number of non-zero pixels in the blue and red color ranges within the player's ROI. If the count of blue pixels exceeds a threshold, the player is labeled as part of the France team and marked with a blue rectangle and label. If the count of red pixels exceeds a threshold, the player is labeled as part of the Belgium team and marked with a red rectangle and label.

- Football Detection: For contours representing the football, it calculates the number of non-zero pixels in the white color range within the football's ROI. If the count exceeds a threshold, the object is labeled as a football and marked with a green rectangle and label.

- Display and Output: Writes the processed frame with highlighted regions to a directory. Displays the processed frame in a window labeled "Match Detection". Allows the user to quit the video playback by pressing 'q' and releases resources after completion.

This code essentially detects players and football in a soccer match video and classifies players based on the color of their jerseys, marking them as either part of the France team or the Belgium team. The football is also detected and labeled accordingly.
