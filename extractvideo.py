import cv2
import dlib
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt

import math
def calculate_eyebrow_movement(left_eyebrow_landmarks, right_eyebrow_landmarks):
    # Compute the average distance between the left eyebrow landmarks
    left_eyebrow_distance = (euclidean(left_eyebrow_landmarks[0], left_eyebrow_landmarks[1]) + 
                         euclidean(left_eyebrow_landmarks[1], left_eyebrow_landmarks[2])) / 2
    # Compute the average distance between the right eyebrow landmarks
    right_eyebrow_distance = (euclidean(right_eyebrow_landmarks[0], right_eyebrow_landmarks[1]) + 
                          euclidean(right_eyebrow_landmarks[1], right_eyebrow_landmarks[2])) / 2

    # Compute the eyebrow movement as the ratio of the left eyebrow distance to the right eyebrow distance
    eyebrow_movement = left_eyebrow_distance / right_eyebrow_distance

    return eyebrow_movement


def calculate_lip_movement(top_lip_landmarks, bottom_lip_landmarks):
    # Compute the average distance between the top lip landmarks
    top_lip_distance = (euclidean(top_lip_landmarks[0], top_lip_landmarks[1]) + 
                         euclidean(top_lip_landmarks[1], top_lip_landmarks[2])) / 2
    # Compute the average distance between the bottom lip landmarks
    bottom_lip_distance = (euclidean(bottom_lip_landmarks[0], bottom_lip_landmarks[1]) + 
                          euclidean(bottom_lip_landmarks[1], bottom_lip_landmarks[2])) / 2

    # Compute the lip movement as the ratio of the top lip distance to the bottom lip distance
    lip_movement = top_lip_distance / bottom_lip_distance

    return lip_movement



def euclidean(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def calculate_blink_rate(left_eye_landmarks, right_eye_landmarks):
    # Compute the average distance between the upper and lower eyelids for the left eye
    left_eye_distance = (euclidean(left_eye_landmarks[1], left_eye_landmarks[5]) + 
                         euclidean(left_eye_landmarks[2], left_eye_landmarks[4])) / 2

    # Compute the average distance between the upper and lower eyelids for the right eye
    right_eye_distance = (euclidean(right_eye_landmarks[1], right_eye_landmarks[5]) + 
                          euclidean(right_eye_landmarks[2], right_eye_landmarks[4])) / 2

    # Compute the blink rate as the ratio of the average eyelid distance to the inter-pupil distance
    blink_rate = (left_eye_distance + right_eye_distance) / euclidean(left_eye_landmarks[0], right_eye_landmarks[0])

    return blink_rate

 
    # Load the pre-trained dlib facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/msi/Desktop/test/shape_predictor_68_face_landmarks.dat")


    # Open the input video
cap = cv2.VideoCapture(0)
    
    # Initialize variables to keep track of blink rate and lip movement
blink_rate = 0
lip_movement = 0
eyebrow_movement = 0
facial_expression = 0
 # Initialize lists to store stress information
stress_blink_rate= []
stress_lip_movement=[]
stress_eyebrow_movement=[]

while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Break the loop if the video has ended
        if not ret:
            break
         
      
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = detector(gray, 0)

        # Loop over the detected faces
        for face in faces:
            # Get the facial landmarks for the face
            shape = predictor(gray, face)

            # Draw the facial landmarks on the frame
            for i in range(68):
                x = shape.part(i).x
                y = shape.part(i).y
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
        # Extract blink rate
            left_eye_landmarks = shape.part(36), shape.part(37), shape.part(38), shape.part(39), shape.part(40), shape.part(41)
            right_eye_landmarks = shape.part(42), shape.part(43), shape.part(44), shape.part(45), shape.part(46), shape.part(47)


        
            blink_rate += calculate_blink_rate(left_eye_landmarks, right_eye_landmarks)
        
            stress_blink_rate.append(blink_rate)
        # Extract lip movement
            top_lip_landmarks = shape.part(50), shape.part(51), shape.part(52), shape.part(53)
     
            bottom_lip_landmarks =  shape.part(58), shape.part(59), shape.part(60), shape.part(61)
            lip_movement += calculate_lip_movement(top_lip_landmarks, bottom_lip_landmarks)
            stress_lip_movement.append(lip_movement)
        # Extract eyebrows movements
            left_eyebrow_landmarks = shape.part(17), shape.part(18), shape.part(19), shape.part(20),shape.part(21), shape.part(22)
     
            right_eyebrow_landmarks = shape.part(22), shape.part(23), shape.part(24), shape.part(25),shape.part(26), shape.part(27)
        
            eyebrow_movement += calculate_eyebrow_movement(left_eyebrow_landmarks, right_eyebrow_landmarks)
            stress_eyebrow_movement.append(eyebrow_movement)

        # Extract eyebrows movements and facial expressions
        # you can use pre-trained models for facial expression recognition
        # or you can use the geometric measurements to extract features

# Compute the average blink rate and lip movement over the entire video
      

                  
        # Show the frame
        cv2.imshow("Frame", frame)


        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    

    # Release the video capture and close the windows
cap.release()
# Call the function with a video file as input
  # Extract features from the facial landmarks
blink_rate = calculate_blink_rate(left_eye_landmarks, right_eye_landmarks)
lip_movement = calculate_lip_movement(top_lip_landmarks, bottom_lip_landmarks)
eyebrow_movement = calculate_eyebrow_movement(left_eyebrow_landmarks, right_eyebrow_landmarks)
# Compute the final stress level
final_stress = 0.25 * blink_rate + 0.25 * eyebrow_movement + 0.25 * facial_expression + 0.25 * lip_movement  
          
            # Print the extracted features
print("Blink rate:", blink_rate)
print("Lip movement:", lip_movement)
print("Eyebrow movement:", eyebrow_movement)
print("final",final_stress)


  # Plot stress information graph
plt.plot(stress_blink_rate)
plt.xlabel('Time (seconds)')
plt.ylabel('blink Stress level')
plt.show()

plt.plot(stress_lip_movement)
plt.xlabel('Time (seconds)')
plt.ylabel('LIP Stress level')
plt.show()

plt.plot(stress_eyebrow_movement)
plt.xlabel('Time (seconds)')
plt.ylabel('eyebrow Stress level')
plt.show()

# Plot the blink rate, lip movement, and eyebrow movement values on the same axis
plt.plot(stress_blink_rate, label='Blink Rate')
plt.plot(stress_lip_movement, label='Lip Movement')
plt.plot(stress_eyebrow_movement, label='Eyebrow Movement')
plt.xlabel('Time (s)')
plt.ylabel('Stress Level')
plt.title('Stress Level - Aggregation of Factors')
plt.legend()
plt.show()