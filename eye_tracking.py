import cv2
import dlib

# Initialize dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the video capture
cap = cv2.VideoCapture(1)

while True:
    # Read the current frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Iterate over detected faces
    for face in faces:
        # Predict the facial landmarks for the face region
        landmarks = predictor(gray, face)

        # Extract the eye landmarks
        left_eye = landmarks.part(36)
        right_eye = landmarks.part(45)

        # Calculate the eye aspect ratio (EAR)
        ear = ((right_eye.x - left_eye.x) ** 2 + (right_eye.y - left_eye.y) ** 2) ** 0.5

        # Define the eye region of interest (ROI)
        left_eye_roi = frame[left_eye.y - 10: left_eye.y + 10, left_eye.x - 10: left_eye.x + 10]
        right_eye_roi = frame[right_eye.y - 10: right_eye.y + 10, right_eye.x - 10: right_eye.x + 10]

        # Draw the eye ROI border
        if ear < 0.2:
            cv2.rectangle(frame, (left_eye.x - 10, left_eye.y - 10), (left_eye.x + 10, left_eye.y + 10),
                          (0, 0, 255), 2)
            cv2.rectangle(frame, (right_eye.x - 10, right_eye.y - 10), (right_eye.x + 10, right_eye.y + 10),
                          (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (left_eye.x - 10, left_eye.y - 10), (left_eye.x + 10, left_eye.y + 10),
                          (0, 255, 0), 2)
            cv2.rectangle(frame, (right_eye.x - 10, right_eye.y - 10), (right_eye.x + 10, right_eye.y + 10),
                          (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Eye Tracking", frame)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()