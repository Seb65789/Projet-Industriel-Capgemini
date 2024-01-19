import pandas as pd
import numpy as np
import save_video 
from rm_NaN_file import rm_NaN
import os

# Fonctions EAR et MAR
def eye_aspect_ratio(eye):
    A = np.linalg.norm(np.array(eval(eye[3])) - np.array(eval(eye[5])))
    B = np.linalg.norm(np.array(eval(eye[2])) - np.array(eval(eye[6])))
    C = np.linalg.norm(np.array(eval(eye[1])) - np.array(eval(eye[7])))
    D = np.linalg.norm(np.array(eval(eye[4])) - np.array(eval(eye[0])))

    
    ear = (A + B + C) / (3.0 * D)
    return ear

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(np.array(eval(mouth[1])) - np.array(eval(mouth[5])))
    B = np.linalg.norm(np.array(eval(mouth[2])) - np.array(eval(mouth[4])))
    C = np.linalg.norm(np.array(eval(mouth[0])) - np.array(eval(mouth[3])))
    mar = (A + B) / (2.0 * C)
    return mar

# Nettoyage des données des fichiers vidéos et CSV
rm_NaN('./csv_videos/','./videos/',75,75)

# Charger les données à partir des fichiers CSV
# Liste des vidéos à traiter
video_names = save_video.get_videos('./videos','noms_videos')

# Créer une liste video_files avec les noms de fichiers modifiés
# On supprime le chemin vers les fichiers vidéos 
# On supprime l'extension du nom de la video
# On ajoute le chemin vers le dossier des csv
videos_csv = [f'./csv_videos/{name[9:-4]}_landmarks.csv' for name in video_names]



signes = ["EAR_left","EAR_right", "EAR_mean","MAR"]

# Initialiser l'entête du data_frame
columns = ['Nom_video']
for videos_csv in videos_csv:
    # Charger le CSV
    df = pd.read_csv(videos_csv)
    taille_premiere_colonne = len(df.iloc[:, 0])


# Création des colonnes du dataframe, image par image (EX : EAR_left_0, EAR_right_0, EAR_mean_0, ...)
for frame in range(taille_premiere_colonne):  # Supposant 876 frames par vidéo
    for signe in signes:
        columns.append(f'{signe}_{frame}')

# Créer le DataFrame final
df_final = pd.DataFrame(columns=columns)
print(df_final)

# Parcourir chaque fichier vidéo
for video_name in video_names:
    print("Traitement en cours pour : ", video_name)

    # Charger le CSV
    df = pd.read_csv('./csv_videos/' + video_name[9:-4] + '_landmarks.csv')

    # Initialiser une liste pour stocker les valeurs
    values = [video_name[9:-4]]

    # Calculer l'EAR pour chaque œil et ajouter les valeurs à la liste
    for frame in range(len(df)):
        eye_coordinates_left = df.loc[frame, ['left_eye_362', 'left_eye_385', 'left_eye_386', 'left_eye_387', 'left_eye_263', 'left_eye_373', 'left_eye_374', 'left_eye_380']]
        eye_coordinates_right = df.loc[frame, ['right_eye_133', 'right_eye_158', 'right_eye_159', 'right_eye_160', 'right_eye_33', 'right_eye_144', 'right_eye_145', 'right_eye_153']]
        mouth_coordinates = df.loc[frame, ['mouth_78', 'mouth_82', 'mouth_312', 'mouth_308', 'mouth_317', 'mouth_87']]

        EAR_left = eye_aspect_ratio(eye_coordinates_left)
        EAR_right = eye_aspect_ratio(eye_coordinates_right)
        EAR_mean = (EAR_left + EAR_right) / 2
        MAR = mouth_aspect_ratio(mouth_coordinates)

        # Ajouter les valeurs à la liste
        values.extend([EAR_left, EAR_right, EAR_mean, MAR])

    # Ajouter des zéros pour égaler les longueurs
    values += [0] * (len(df_final.columns) - len(values))

    # Ajouter la nouvelle ligne au DataFrame
    df_final.loc[len(df_final)] = values

# Remplacer NaN par 0 dans le DataFrame final
df_final = df_final.fillna(0)

# Enregistrez le DataFrame final dans un fichier CSV
df_final.to_csv('resultat_final.csv', index=False)
