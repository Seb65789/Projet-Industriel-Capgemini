import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import shutil

# Liste des vidéos à traiter

# Dossier contenant les vidéos
video_folder = "video"
output_csv_path = "dataframe_final.csv"


# Global blink counter
blink_count = 0
FREQUENCE_EBR = 100
FREQUENCE_PERCLOS = 100

# Add this line to track the state of the eyes (open or closed)
eyes_state = "open"

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
    return (N1 + N2 + N3) / (3 * D)

def mouth_aspect_ratio(landmarks):
    ''' Calculate mouth feature as the ratio of the mouth length to mouth width
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :return: Mouth feature value
    '''
    N1 = distance(landmarks[mouth[1][0]], landmarks[mouth[1][1]])
    N2 = distance(landmarks[mouth[2][0]], landmarks[mouth[2][1]])
    N3 = distance(landmarks[mouth[3][0]], landmarks[mouth[3][1]])
    D = distance(landmarks[mouth[0][0]], landmarks[mouth[0][1]])
    return  (N1 + N2 + N3) / (3 * D)

def calculate_angle(x, y):
    return (-np.abs(np.arctan2(y, x) * 180/np.pi) + 180)

def hop(head):
    x = np.array(head[0])[0] - np.array(head[1])[0]
    y = np.array(head[0])[1] - np.array(head[1])[1]
    z = np.array(head[0])[2] - np.array(head[1])[2]
    angle_hb = calculate_angle(y, x)
    angle_gd = calculate_angle(z, y)
    return angle_gd, angle_hb



def run_face_mp(image):
    ''' Get face landmarks using the FaceMesh MediaPipe model.
    Calculate facial features using the landmarks.
    :param image: Image for which to get the face landmarks
    :param csv_writer: CSV writer object
    :return: Feature 1 (Eye), image with mesh drawings
    '''
    # Convert image to RGB and flip it
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Process image with FaceMesh
    results = face_mesh.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Initialize EAR to a value indicating no face detected
    ear_left = -1000
    ear_right = -1000
    hop_gd, hop_hb = 0, 0  # Initialize hop_gd and hop_hb

    if results.multi_face_landmarks:
        landmarks_positions = []
        for _, data_point in enumerate(results.multi_face_landmarks[0].landmark):
            landmarks_positions.append([data_point.x, data_point.y, data_point.z])

        landmarks_positions = np.array(landmarks_positions)
        landmarks_positions[:, 0] *= image.shape[1]
        landmarks_positions[:, 1] *= image.shape[0]

        # Extraire les coordonnées x, y, z des points 10 et 152 pour hop
       
        # faire une liste pour les points
        head_position = [landmarks_positions[idx] for idx in [10,152]]


        # Draw face mesh over image
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

        # Calculate Eye Aspect Ratio for the left eye
        ear_left = eye_aspect_ratio(landmarks_positions, left_eye)

        # Calculate Eye Aspect Ratio for the right eye
        ear_right = eye_aspect_ratio(landmarks_positions, right_eye)

        # Calculate the average of ear_left and ear_right for both eyes
        ear_moyen = (ear_left + ear_right) / 2

        # Calculate Mouth Aspect Ratio
        mar = mouth_aspect_ratio(landmarks_positions)

        # Calculate head orientation
        hop_gd, hop_hb = hop(head_position)

    return mar, ear_left, ear_right, ear_moyen, hop_gd, hop_hb, image


# Liste des vidéos à traiter
video_paths = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
video_paths = [os.path.join(video_folder, video) for video in video_paths]
nb_frame_ferme = 0;

# On crée une liste avec les noms des colonnes
liste_colonnes = ["video_name"]
for i in range(875):
    #liste_colonnes.append("MAR_{}".format(i))
    #liste_colonnes.append("EAR_left_{}".format(i))
    # liste_colonnes.append("EAR_right_{}".format(i))
    liste_colonnes.append("EAR_moyen_{}".format(i))
    # liste_colonnes.append("HOP_gd_{}".format(i))
    # liste_colonnes.append("HOP_hb_{}".format(i))
    # liste_colonnes.append("EBR_{}".format(i))
    # liste_colonnes.append("PERCLOS_{}".format(i))

# Define right eye, left eye, and mouth landmark positions
right_eye = [[33, 133], [160, 144], [159, 145], [158, 153]]
right_eye_no_tuples = [33, 133, 160, 144, 159, 145, 158, 153]
left_eye = [[263, 362], [387, 373], [386, 374], [385, 380]]
left_eye_no_tuples = [263, 362, 387, 373, 386, 374, 385, 380]
mouth = [[61, 291], [39, 181], [0, 17], [269, 405]]
mouth_no_tuples = [61, 291, 39, 181, 0, 17, 269, 405]
head = [[10, 152]]

# Declare FaceMesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.3, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Initialiser les header du csv avec listes colonnes en dehors de la boucle
with open(output_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(liste_colonnes)

# Boucle pour traiter chaque vidéo
for video_path in video_paths:
    eyesclosed = False
    blink_count = 0
    eyes_state = "open"
    nouvelle_ligne = [video_path]
    perclos = -1

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    frame_count = 0  # Initialize frame count
    # Process video frames
    while cap.isOpened():
        success, image = cap.read()
        if not success or frame_count == 876:
            print("Ignoring empty camera frame.")
            break

        # Run face mesh processing and get EAR, MAR, and annotated image
        mar, ear_left, ear_right, ear_moyen, hop_gd, hop_hb, image = run_face_mp(image)
        

        # nouvelle_ligne.append(mar)
        # nouvelle_ligne.append(ear_left)
        # nouvelle_ligne.append(ear_right)
        nouvelle_ligne.append(ear_moyen)
        # nouvelle_ligne.append(hop_gd)
        # nouvelle_ligne.append(hop_hb)

        # si le numéro de frame actuel est un multiple de FREQUENCE_EBR on ebre prend blink counter on reinitialise le blink counter et on continue
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % FREQUENCE_EBR == 0:
            # nouvelle_ligne.append(blink_count)
            blink_count = 0
            # ajouter à l'écran " Blink count enregistré"
            cv2.putText(image, "Blink count enregistré", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            perclos = (nb_frame_ferme/(FREQUENCE_EBR - nb_frame_ferme))*100
            # nouvelle_ligne.append(perclos)
            nb_frame_ferme = 0
            
        #else:
            #EBR à -1
            #nouvelle_ligne.append(-1)
            #PERCLOS à -1
            #nouvelle_ligne.append(-1)

        # Display blink count on screen
        cv2.putText(image, "Blink: {}".format(blink_count), (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


        # If EAR is valid, display it on the screen
        if ear_moyen != -1000:
            cv2.putText(image, "EAR Left: {:.2f}".format(ear_left), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image, "EAR Right: {:.2f}".format(ear_right), (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image, "EAR Moyen: {:.2f}".format(ear_moyen), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            #print("Eye Aspect Ratio - Left:", ear_left)
            #print("Eye Aspect Ratio - Right:", ear_right)
            #print("Eye Aspect Ratio - Moyen:", ear_moyen)

            # Check if eyes are closed and update blink count
            if ear_left < 0.22 or ear_right < 0.22:
                nb_frame_ferme +=1
                if eyes_state == "open":
                    eyes_state = "closed"
                    blink_count += 1
            else:
                eyes_state = "open"

                # Display "EYE CLOSED" when eyes close
                if eyes_state == "closed":
                    cv2.putText(image, "EYE CLOSED".format(ear_moyen), (20, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    eyes_state = "open"
            #print("EBR:", blink_count)

            # Display head orientation
            cv2.putText(image, "Head Angle GD: {:.2f}".format(hop_gd), (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image, "Head Angle HB: {:.2f}".format(hop_hb), (10, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Display MAR on the screen
            cv2.putText(image, "MAR: {:.2f}".format(mar), (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            print("Mouth Aspect Ratio:", mar)
            # Afficher PERCLOS sur l'écran
            cv2.putText(image, "PERCLOS: {:.2f}".format(perclos), (10, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display annotated image with face mesh
        cv2.imshow('MediaPipe FaceMesh', image)
        frame_count += 1
        # Exit if 'q' key is pressed
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    with open(output_csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(nouvelle_ligne)


    # Release resources
    cap.release()

# Close all windows and release the face mesh model
cv2.destroyAllWindows()
face_mesh.close()
