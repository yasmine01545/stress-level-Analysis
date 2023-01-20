# stress-level-Analysis
we will create a script that takes as input a file from a streaming and returns as output a graph that describes the stress level per second for each factor and an aggregation of the different factors together.
This code uses the OpenCV, dlib, and scipy libraries to track facial landmarks on a video feed and calculate various facial movements and expressions, such as blink rate, lip movement, and eyebrow movement.

# Dependencies 
 - OpenCV
 - dlib
 - scipy
 - matplotlib

# Usage
1. Download the pre-trained dlib facial landmark detector from [Here](https://link-url-here.org) and place it in the same directory as the code.
2. Run the code by typing python extractvideo.py in the command line.
3. The code will start capturing frames from your webcam and display the facial movements and expressions on the screen.

# Inputs and Outputs
 - Input: Video feed from your webcam
 - Output: Calculated facial movements and expressions, such as blink rate, lip movement, and eyebrow movement.

# How it works
The code uses the dlib library to load a pre-trained facial landmark detector and the OpenCV library to capture frames from a video feed. The code then calculates the average distance between certain sets of landmarks on the face, such as the upper and lower eyelids, and uses these distances to compute various facial movements and expressions. Euclidean distance formula is used to calculate the distance between different points on the face.

# Additional Information

The code also uses matplotlib library to plot the facial expression in the form of graph. Stress information is also tracked and stored in the form of list.

# How to contribute
If you would like to contribute to this code, please follow these guidelines:

 - Fork the repository
 - Make your changes
 - Test your changes
 - Create a pull request

