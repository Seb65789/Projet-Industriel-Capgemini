import cv2
import mediapipe as mp
import csv
import os
import save_video

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
    left_eye_columns = [f"left_eye_{i}" for i in [362, 385, 386, 387, 263, 373, 374, 380]]
    right_eye_columns = [f"right_eye_{i}" for i in [133, 158, 159, 160, 33, 144, 145, 153]]
    mouth_columns = [f"mouth_{i}" for i in [78, 82, 312, 308, 317, 87]]
    head_columns = [f"head_{i}" for i in [10, 152]] 

    # Créer le fichier CSV et écrire l'en-tête
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_file_path = f"./csv_videos/{video_name}_landmarks.csv"
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
                left_eye_landmarks = [[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y,face_landmarks.landmark[i].z] for i in [362, 385, 386, 387, 263, 373, 374, 380]]
                right_eye_landmarks = [[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y,face_landmarks.landmark[i].z] for i in [133, 158, 159, 160, 33, 144, 145, 153]]
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
    
    # Si les vidéos font plus de 875 frames, on supprime les frames en plus
    if len(landmarks_all_frames) > 875 :
        landmarks_all_frames = landmarks_all_frames[:875]
    print(len(landmarks_all_frames))

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


# Liste des vidéos à traiter (récupérer grâce à une liste)
video_paths = save_video.get_videos('./videos','noms_videos.txt')

# Traiter chaque vidéo
for video in video_paths:
    process_video(video)
print("Terminé")
