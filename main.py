import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Define video capture object
cap = cv2.VideoCapture(r"C:\Users\Alfa\Desktop\python720\Athletes-Video-Format.mp4")

# Check if video capture object is opened
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Define POE and reference landmark indices
poe_landmark = 11  # Ankle landmark
ref_landmark = 13  # Hip landmark

# Initialize angle list to store results
angles = []

# Initialize pose detection model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()

        # Convert the image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image with the pose model
        results = pose.process(image)

        # Convert the image back to BGR for OpenCV display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmark coordinates
        if results.pose_landmarks:
            poe_coord = results.pose_landmarks.landmark[poe_landmark]
            ref_coord = results.pose_landmarks.landmark[ref_landmark]

            # Calculate angle in degrees using atan2
            angle = np.arctan2(poe_coord.y - ref_coord.y, poe_coord.x - ref_coord.x) * 180 / np.pi

            # Append angle to results list
            angles.append(angle)

        # Draw pose landmarks and angle information on the image
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(image, f"POE Angle: {angle:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting image
        cv2.imshow('Pose Estimation', image)

        # Wait for a key press to continue
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()

# Print or analyze the calculated angles
print(f"POE Angles: {angles}")
