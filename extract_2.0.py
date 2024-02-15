import cv2
import mediapipe as mp
import os
import pandas as pd

# Les points qui nous interresse
right_eye = [33, 133, 160, 144, 159, 145, 158, 153]
left_eye = [263, 362, 387, 373, 386, 374, 385, 380]
mouth = [61,291,39,181,0,17,269,405]
head = [10,152]

# Création de la liste de points sans tuples 
list_points_no_tuples = right_eye + left_eye + mouth + head
print(list_points_no_tuples)

# Les vidéos à traiter
video_paths = [f for f in os.listdir("videos") if f.endswith('.mp4')]
video_paths = [os.path.join("videos", video) for video in video_paths]

# Declare FaceMesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)

# Le dessin du facemesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# liste des colonnes du dataframe
liste_colonnes = ["nom_video"]
for j in range(875) :
    for k in list_points_no_tuples : 
        liste_colonnes.append(f"x_{k}_{j}")
        liste_colonnes.append(f"y_{k}_{j}")
        liste_colonnes.append(f"z_{k}_{j}")

df = pd.DataFrame(columns=liste_colonnes)

# Compteur pour l'avancement
videos_processed = 0

for video_path in video_paths:
    videos_processed += 1

    print(f"Traitement {videos_processed}/{len(video_paths)} de : {video_path}")
    nouvelle_ligne = [video_path]

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    frame_count = 0  # Initialize frame count
    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or frame_count == 876:
            break

        # Convertir l'image en RGB pour améliorer la précision de Mediapipe
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Détecter les visages dans l'image
        faces = face_mesh.process(frame)

        if faces.multi_face_landmarks:
            for face_landmarks in faces.multi_face_landmarks:
                landmarks_positions = []
                for landmark_id in list_points_no_tuples:
                # Accéder aux coordonnées x, y et z du landmark spécifique
                    data_point = face_landmarks.landmark[landmark_id]
                    nouvelle_ligne.append(data_point.x * frame.shape[1])
                    nouvelle_ligne.append(data_point.y * frame.shape[0]) 
                    nouvelle_ligne.append(data_point.z)

        frame_count +=1

    # Ajouter des zéros supplémentaires à nouvelle_ligne pour atteindre la longueur souhaitée
    nb_zeros_a_ajouter = df.shape[1] - len(nouvelle_ligne)
    nouvelle_ligne += [0] * nb_zeros_a_ajouter
    df.loc[len(df)] = nouvelle_ligne
    
    # Release resources
    cap.release()

# Convertir le DataFrame en fichier CSV
df.to_csv('csv_videos/videos_coordinates.csv', index=False)

# Close all windows and release the face mesh model
cv2.destroyAllWindows()
face_mesh.close()
