import pandas as pd
import numpy as np
from collections import deque

# Fonctions EAR et MAR
def eye_aspect_ratio(eye):
    A = np.linalg.norm(np.array(eval(eye[1])) - np.array(eval(eye[5])))
    B = np.linalg.norm(np.array(eval(eye[2])) - np.array(eval(eye[4])))
    C = np.linalg.norm(np.array(eval(eye[0])) - np.array(eval(eye[3])))
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(np.array(eval(mouth[1])) - np.array(eval(mouth[5])))
    B = np.linalg.norm(np.array(eval(mouth[2])) - np.array(eval(mouth[4])))
    C = np.linalg.norm(np.array(eval(mouth[0])) - np.array(eval(mouth[3])))
    mar = (A + B) / (2.0 * C)
    return mar

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

# Charger les données à partir des fichiers CSV
# Liste des vidéos à traiter
video_names = ['kss#8-9#F#rldd#28-10','kss#1-3#F#rldd#10-0']

video_paths = [name + '.mp4' for name in video_names]

# Créer une liste video_files avec les noms de fichiers modifiés
video_files = [name + '_landmarks.csv' for name in video_names]

signes = ["EAR_left","EAR_right", "EAR_mean","MAR", "EBS"]

count = 0       # Initialisation du compteur de frame ou EAR<seuil à 0
blink = 0       # Initialisation du compteur de clignement des yeux
deque_size = 20     # Definir la taille du tableau

# Initialize the deque with zeros
ear_list = deque([0.25] * deque_size, maxlen=deque_size)

# Initialiser l'entête du data_frame
columns = ['Nom_video']
for video_file in video_files:
    # Charger le CSV
    df = pd.read_csv(video_file)
    taille_premiere_colonne = len(df.iloc[:, 0])

for signe in signes:
    if signe == "EBS":
        for frame in range(0, taille_premiere_colonne, 25):  # Add columns only for multiples of 25
            columns.append(f'{signe}_{frame}')
    else:
        for frame in range(taille_premiere_colonne):  # Supposant 876 frames par vidéo
            columns.append(f'{signe}_{frame}')

# Créer le DataFrame final
df_final = pd.DataFrame(columns=columns)

df_final.to_csv('resultat_final.csv', index=False)

# Parcourir chaque fichier vidéo
for video_name in video_names:
    print("Traitement en cours pour : ", video_name)

    # Charger le CSV
    df = pd.read_csv(video_name + '_landmarks.csv')

    # Récupérer le nom de la vidéo à partir du nom du fichier
    df_final.at[video_name, 'Nom_video'] = video_name

    # Calculer l'EAR pour chaque œil et ajouter les valeurs à df_final
    for frame in range(len(df)):
        eye_coordinates_left = df.loc[frame, ['left_eye_463', 'left_eye_385', 'left_eye_387', 'left_eye_263', 'left_eye_373', 'left_eye_380']]
        eye_coordinates_right = df.loc[frame, ['right_eye_133','right_eye_158','right_eye_160','right_eye_33','right_eye_144','right_eye_153']]
        mouth_coordinates = df.loc[frame, ['mouth_78','mouth_82','mouth_312', 'mouth_308','mouth_317','mouth_87']]
        
        EAR_left = eye_aspect_ratio(eye_coordinates_left)
        EAR_right = eye_aspect_ratio(eye_coordinates_right)
        EAR_mean = (EAR_left+EAR_right)/2
        MAR = mouth_aspect_ratio(mouth_coordinates)

        # Ajouter la valeur EAR à la colonne correspondante dans df_final
        df_final.at[video_name, f'EAR_left_{frame}'] = EAR_left
        df_final.at[video_name, f'EAR_right_{frame}'] = EAR_right
        df_final.at[video_name, f'EAR_mean_{frame}'] = EAR_mean
        df_final.at[video_name, f'MAR_{frame}'] = MAR

        ear_list.append(EAR_mean)

        # Vérifier la somnolence en comparant l'EAR au seuil -> EBS
        if F_seuil(EAR_mean, ear_list) == True:
            if count == 3:
                blink = blink + 1
            count = count + 1
            
        else:
            count = 0

        # Ajouter une valeur à la colonne "EBS" toutes les 25 frames
        if frame % 25 == 0:
            df_final.at[video_name, f'EBS_{frame}'] = blink
            blink = 0



print(df_final)

# Transformer df_final en csv
df_final.to_csv('resultat_final.csv', index=False)

# Enregistrez le DataFrame final dans un fichier CSV
