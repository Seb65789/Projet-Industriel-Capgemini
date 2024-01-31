import cv2
import mediapipe as mp
import numpy as np
import csv
eyesclosed = True
def distance(p1, p2):
    ''' Calculate distance between two points
    :param p1: First Point 
    :param p2: Second Point
    :return: Euclidean distance between the points. (Using only the x and y coordinates).
    '''
    return (((p1[:2] - p2[:2])**2).sum())**0.5

def eye_aspect_ratio(landmarks, eye):
    ''' Calculate the ratio of the eye length to eye width. 
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :param eye: List containing positions which correspond to the eye
    :return: Eye aspect ratio value
    '''
    N1 = distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
    N2 = distance(landmarks[eye[2][0]], landmarks[eye[2][1]])
    N3 = distance(landmarks[eye[3][0]], landmarks[eye[3][1]])
    D = distance(landmarks[eye[0][0]], landmarks[eye[0][1]])
    return 100 * (N1 + N2 + N3) / (3 * D)

def run_face_mp(image, csv_writer):
    ''' Get face landmarks using the FaceMesh MediaPipe model. 
    Calculate facial features using the landmarks.
    :param image: Image for which to get the face landmarks
    :param csv_writer: CSV writer object
    :return: Feature 1 (Eye), image with mesh drawings
    '''
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    ear = -1000
    if results.multi_face_landmarks:
        landmarks_positions = []
        # assume that only face is present in the image
        for _, data_point in enumerate(results.multi_face_landmarks[0].landmark):
            landmarks_positions.append([data_point.x, data_point.y, data_point.z]) # saving normalized landmark positions
        landmarks_positions = np.array(landmarks_positions)
        landmarks_positions[:, 0] *= image.shape[1]
        landmarks_positions[:, 1] *= image.shape[0]

        # Write landmarks to CSV
        csv_writer.writerow(landmarks_positions.flatten())

        # draw face mesh over image
        for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

        ear = eye_aspect_ratio(landmarks_positions, left_eye)
    
    return ear, image

# Right eye, left eye, and mouth landmark positions
right_eye = [[33, 133], [160, 144], [159, 145], [158, 153]]
left_eye = [[263, 362], [387, 373], [386, 374], [385, 380]]
mouth = [[61, 291], [39, 181], [0, 17], [269, 405]]

# Declaring FaceMesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.3, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Example usage
video_path = "test_aurelie_avant.mp4"
output_csv_path = "video_path_landmark.csv"

cap = cv2.VideoCapture(video_path)
with open(output_csv_path, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        ear, image = run_face_mp(image, csv_writer)
        if ear != -1000:
            cv2.putText(image, "EAR: {:.2f}".format(ear), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            print("Eye Aspect Ratio:", ear)
            if ear < 22:
                eyesclosed = True
            else:
                if eyesclosed == True:
                    cv2.putText(image, "EYE CLOSED".format(ear), (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    eyesclosed = False

        cv2.imshow('MediaPipe FaceMesh', image)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
cap.release()
face_mesh.close()
