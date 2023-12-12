import cv2
import mediapipe as mp
import numpy as np
import time

# Ajoutez cette ligne pour spécifier la version de numpy
import numpy as np

# On teste la valeur courante par rapport a une liste mise a jour toutes les t secondes.
def F_seuil(val_EAR,list_EAR):
    ferme = False
    max = np.max(list_EAR)
    min = np.min(list_EAR)
    f_seuil = max - min/2  
    if val_EAR < f_seuil :
        ferme = True
    else :
        ferme = False
    return ferme

# Définition de la fonction eye_aspect_ratio
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Définition de la fonction mouth_aspect_ratio
def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[1] - mouth[5])
    B = np.linalg.norm(mouth[2] - mouth[4])
    C = np.linalg.norm(mouth[0] - mouth[3])
    mar = (A + B) / (2.0 * C)
    return mar

# Charger le modèle Mediapipe pour la détection des yeux et des lèvres
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialiser le détecteur de visage et le dessinateur pour les annotations
face_mesh = mp_face_mesh.FaceMesh()
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Charger la vidéo
video_path = 'kss#8-9#F#rldd#28-10.mp4_12.mp4'
cap = cv2.VideoCapture(video_path)

# Initialiser des listes pour stocker les données
ear_list = []
mar_list = []  # Ajout de la liste pour stocker les valeurs de MAR
blink_times = []  # Ajout de la liste pour stocker les temps de clignement

# Nombre de trames consécutives pour considérer un clignement
ear_consecutive_frames = 3

# Initialiser la variable pour suivre l'état de somnolence
drowsy_state = False

count = 0

blink = 0

# Lecture de la vidéo
start_time = time.time()  # Enregistrez le temps de début
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les visages dans l'image
    faces = face_mesh.process(frame)

    if faces.multi_face_landmarks:
        for face_landmarks in faces.multi_face_landmarks:
            # Extraire les coordonnées des points des yeux
            left_eye_landmarks = []
            keypoint_left_eye = [463, 385, 387, 263, 373, 380]
            for i in keypoint_left_eye:
                left_eye_landmarks.append([face_landmarks.landmark[i].x, face_landmarks.landmark[i].y])

            right_eye_landmarks = []
            keypoint_right_eye = [133, 158, 160, 33, 144, 153]
            for i in keypoint_right_eye:
                right_eye_landmarks.append([face_landmarks.landmark[i].x, face_landmarks.landmark[i].y])

            # Extraire les coordonnées des points de la bouche
            mouth_landmarks = []
            keypoint_mouth = [78, 82, 312, 308, 317, 87]
            for i in keypoint_mouth:
                mouth_landmarks.append([face_landmarks.landmark[i].x, face_landmarks.landmark[i].y])

            # Calculer l'Eye Aspect Ratio (EAR) pour chaque œil
            left_ear = eye_aspect_ratio(np.array(left_eye_landmarks))
            right_ear = eye_aspect_ratio(np.array(right_eye_landmarks))

            # Calculer le Mouth Aspect Ratio (MAR) pour la bouche
            mar = mouth_aspect_ratio(np.array(mouth_landmarks))

            # Moyenne des deux yeux pour obtenir l'EAR final
            ear = (left_ear + right_ear) / 2.0

            # Ajouter l'EAR et le MAR à leurs listes respectives
            ear_list.append(ear)
            mar_list.append(mar)

            # Dessiner les points des yeux sur l'image
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=drawing_spec)

            # Dessiner le texte indiquant l'EAR et le MAR sur l'image
            cv2.putText(frame, f'EAR: {ear:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'MAR: {mar:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Vérifier la somnolence en comparant l'EAR au seuil
            if F_seuil(ear,ear_list) == True:
                if count == 3:
                    blink = blink + 1
                count = count + 1
            
            else:
                count = 0

    # Afficher la vidéo
    cv2.imshow('Video', frame)

    # Arrêter la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        
print(f"Nombre de clignements des yeux pendant la video : {blink}")
# Libérer la capture vidéo
cap.release()
cv2.destroyAllWindows()