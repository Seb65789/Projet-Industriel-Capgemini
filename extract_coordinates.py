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


if not os.path.exists("videos/videos_traitees.txt"):
    with open("videos/videos_traitees.txt", "w") as save_video_name:
        print("Création du fichier")

# Ouverture en mode append
with open("videos/videos_traitees.txt", "a+") as save_video_name:
    # Réinitialiser le curseur au début du fichier
    save_video_name.seek(0)

    # Compteur pour le nombre de lignes
    nombre_lignes = 0

    # Lire chaque ligne du fichier
    for ligne in save_video_name:
        nombre_lignes += 1

print("Il y a", nombre_lignes, "lignes dans le fichier")
df = 0

# Création du dataframe
if not os.path.exists("~/videos_coordinates.csv"):
  # liste des colonnes du dataframe
  liste_colonnes = ["nom_video"]
  for j in range(875) :
      for k in list_points_no_tuples :
          liste_colonnes.append(f"x_{k}_{j}")
          liste_colonnes.append(f"y_{k}_{j}")
          liste_colonnes.append(f"z_{k}_{j}")

  df = pd.DataFrame(columns=liste_colonnes)

# Si le fichier existe
else :
  df = pd.read_csv("videos_coordinates.csv")

print(df.shape)

# Compteur pour l'avancement
videos_processed = 0

# Liste des videos traitées
#list_video_processed = []

# Parcourir chaque chemin de vidéo
for video_path in video_paths:
    
    # Vérifier si la vidéo a déjà été traitée
    with open("videos/videos_traitees.txt", "a+") as save_video_name:
        save_video_name.seek(0)
        video_saved = save_video_name.read()
        if video_path[7:] in video_saved:
            continue  # Si oui, passer à la vidéo suivante
        else :
          videos_processed += 1

    print(f"Traitement {videos_processed}/{len(video_paths) - nombre_lignes} de : {video_path[7:]}")
    nouvelle_ligne = [video_path[7:]]

    # Ouvrir la capture vidéo
    cap = cv2.VideoCapture('videos/3#kss#8-9#F#rldd#60-10#029.mp4')

    frame_count = 0  # Initialiser le compteur de frames

    # Traiter les frames de la vidéo
    while cap.isOpened():
        ret, frame = cap.read()

        # Vérifier si la frame est lue correctement
        if (not ret) & (frame_count < 875) :
          if (frame_count == 875) :
            print("Video trop longue ", video_path[7:])
            break
          else :
            print("Je n'arrive pas à lire la vidéo")
            with open("videos/videos_illisibles.txt","a+") as video_non_traitée :
                video_non_traitée.write(video_path[7:]+"\n")
                
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

        frame_count += 1


    # Vérifier la longueur de nouvelle_ligne et ajuster si nécessaire
    nb_zeros_a_ajouter = df.shape[1] - len(nouvelle_ligne)

    #Si on a plus de 10% de valeurs nulle, on ajoute la video à la liste des videos à supprimer
    if nb_zeros_a_ajouter >= 0.10*len(nouvelle_ligne) :
      with open("videos/videos_illisibles.txt","a+") as video_non_traitée :
            video_non_traitée.write(video_path+"\n")
            print("Trop de NULL pour :", video_path[7:],'\n')
      continue

    #print(len(nouvelle_ligne))
    nouvelle_ligne += [0] * nb_zeros_a_ajouter

    # Ajouter nouvelle_ligne au DataFrame
    df = pd.concat([df, pd.DataFrame([nouvelle_ligne], columns=df.columns)], ignore_index=True)
    print(df.shape)

    # Libérer les ressources de la capture vidéo
    cap.release()

    # Ajouter le nom de la vidéo au fichier de vidéos traitées
    # liste_video_processed.append(video_path[53:])
    with open("videos/videos_traitees.txt", "a+") as save_video_name:
        save_video_name.write(video_path[7:] + "\n")

    # Pour traiter les videos par paquets
    if(videos_processed == 1200) :
      break

# Fermer toutes les fenêtres et libérer les ressources
cv2.destroyAllWindows()
face_mesh.close()

# with open("/content/drive/MyDrive/Capgemini/videos_entrainement/videos_traitees.txt", "a+") as save_video_name:
  #for video in list_video_processed :
    #save_video_name.write(video + "\n")

# Écrire le DataFrame dans un fichier CSV
df.to_csv('csv/videos_coordinates.csv', index=False)

# Fermer le fichier de vidéos traitées
save_video_name.close()

print("Terminé !")
