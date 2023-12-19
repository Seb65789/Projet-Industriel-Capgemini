import cv2
import mediapipe as mp
import csv
import os

# Ajoutez cette ligne pour spécifier la version de numpy
import numpy as np; np_version = np.__version__.split('.')

if int(np_version[0]) < 1 or (int(np_version[0]) == 1 and int(np_version[1]) < 20):
    raise ImportError("Numpy version 1.20.0 or above is required for this version of mediapipe. "
                      "Please upgrade numpy by running: pip install --upgrade numpy")

# Charger le modèle Mediapipe pour la détection des yeux et des lèvres
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def process_video(video_path):
    # Initialiser le détecteur de visage et le dessinateur pour les annotations
    face_mesh = mp_face_mesh.FaceMesh()
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # Initialiser des listes pour stocker les données
    landmarks_all_frames = []

    # Définir les noms de colonnes
    left_eye_columns = [f"left_eye_{i}" for i in [263, 387, 385, 326, 380, 373]]
    right_eye_columns = [f"right_eye_{i}" for i in [33, 160, 158, 133, 153, 144]]
    mouth_columns = [f"mouth_{i}" for i in [78, 82, 312, 308, 317, 87]]
    head_columns = [f"head_{i}" for i in [10, 152]] 

    # Créer le fichier CSV et écrire l'en-tête
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_file_path = f"{video_name}_landmarks.csv"
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Écrire l'en-tête du fichier CSV avec toutes les colonnes
        writer.writerow(left_eye_columns + right_eye_columns + mouth_columns+head_columns) 

    # Charger la vidéo
    cap = cv2.VideoCapture(video_path)

    # Lecture de la vidéo
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
                # Extraire les coordonnées des points des yeux et de la bouche
                left_eye_landmarks = [[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y,face_landmarks.landmark[i].z] for i in [463, 385, 387, 263, 373, 380]]
                right_eye_landmarks = [[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y,face_landmarks.landmark[i].z] for i in [133, 158, 160, 33, 144, 153]]
                mouth_landmarks = [[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y, face_landmarks.landmark[i].z] for i in [78, 82, 312, 308, 317, 87]]
                head_landmarks = [[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y, face_landmarks.landmark[i].z] for i in [10, 152]]
                
                # Ajouter les coordonnées aux listes respectives
                landmarks_all_frames.append(left_eye_landmarks + right_eye_landmarks + mouth_landmarks +head_landmarks)

                # Dessiner les points des yeux sur l'image
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=drawing_spec)

        # Afficher la vidéo
        cv2.imshow('Video', frame)

        # Arrêter la boucle si la touche 'q' est pressée
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # En dehors de la boucle, écrire les coordonnées dans le fichier CSV
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Écrire les coordonnées dans le fichier CSV
        writer.writerows(landmarks_all_frames)
        print(f"Les coordonnées des landmarks ont été sauvegardées dans : {csv_file_path}")

    # Libérer la capture vidéo
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()


# Liste des vidéos à traiter
video_paths = ['kss#1-3#F#rldd#10-0.mp4', 'kss#8-9#F#rldd#28-10.mp4']

# Traiter chaque vidéo
for video_path in video_paths:
    process_video(video_path)
print("Terminé")
