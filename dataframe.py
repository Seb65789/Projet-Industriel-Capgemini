import pandas as pd
import numpy as np

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

def calculate_angle(x,y) :
    return(-np.abs(np.arctan2(y,x)*180/np.pi)+180)

def hop(head):
    x = np.array(eval(head[0]))[0] - np.array(eval(head[1]))[0]
    y = np.array(eval(head[0]))[1] - np.array(eval(head[1]))[1]
    z = np.array(eval(head[0]))[2] - np.array(eval(head[1]))[2]
    angle_hb = calculate_angle(y,x)
    angle_gd = calculate_angle(z,y)
    return angle_gd, angle_hb

# Charger les données à partir des fichiers CSV
# Liste des vidéos à traiter
video_names = ['kss#8-9#F#rldd#28-10','kss#1-3#F#rldd#10-0', 'test_aurelie']

video_paths = [name + '.mp4' for name in video_names]

# Créer une liste video_files avec les noms de fichiers modifiés
video_files = [name + '_landmarks.csv' for name in video_names]

signes = ["EAR_left","EAR_right", "EAR_mean","MAR"]

# Initialiser l'entête du data_frame
columns = ['Nom_video']
for video_file in video_files:
    # Charger le CSV
    df = pd.read_csv(video_file)
    taille_premiere_colonne = len(df.iloc[:, 0])

for frame in range(taille_premiere_colonne):  # Supposant 876 frames par vidéo
    for signe in signes:
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
        eye_coordinates_left = df.loc[frame, ['left_eye_362', 'left_eye_385','left_eye_386', 'left_eye_387', 'left_eye_263', 'left_eye_373', 'left_eye_374', 'left_eye_380']]
        eye_coordinates_right = df.loc[frame, ['right_eye_133', 'right_eye_158', 'right_eye_159','right_eye_160','right_eye_33','right_eye_144', 'right_eye_145','right_eye_153']]
        mouth_coordinates = df.loc[frame, ['mouth_78','mouth_82','mouth_312', 'mouth_308','mouth_317','mouth_87']]
        head_coordinates = df.loc[frame, ['head_10', 'head_152']]
        
        EAR_left = eye_aspect_ratio(eye_coordinates_left)
        EAR_right = eye_aspect_ratio(eye_coordinates_right)
        EAR_mean = (EAR_left+EAR_right)/2
        MAR = mouth_aspect_ratio(mouth_coordinates)
        HOP_dg, HOP_hb = hop(head_coordinates)

        # Ajouter la valeur EAR à la colonne correspondante dans df_final
        df_final.at[video_name, f'EAR_left_{frame}'] = EAR_left
        df_final.at[video_name, f'EAR_right_{frame}'] = EAR_right
        df_final.at[video_name, f'EAR_mean_{frame}'] = EAR_mean
        df_final.at[video_name, f'MAR_{frame}'] = MAR
        df_final.at[video_name, f'HOP_dg_{frame}'] = HOP_dg
        df_final.at[video_name, f'HOP_hb_{frame}'] = HOP_hb



print(df_final)

# Transformer df_final en csv
df_final.to_csv('resultat_final.csv', index=False)

# Enregistrez le DataFrame final dans un fichier CSV
