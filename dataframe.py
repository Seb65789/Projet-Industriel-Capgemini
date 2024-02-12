import pandas as pd
import numpy as np
import save_video 
from rm_NaN_file import rm_NaN
from perclos import perclos
import random

def distance(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2+(p1[2]-p2[2])**2)**0.5

# Fonctions EAR et MAR
def eye_aspect_ratio(eye):
    A = distance(eval(eye.iloc[1]),eval(eye.iloc[4]))
    B = distance(eval(eye.iloc[2]),eval(eye.iloc[5]))
    C = distance(eval(eye.iloc[0]),eval(eye.iloc[3]))
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = distance(np.array(eval(mouth.iloc[1])),np.array(eval(mouth.iloc[5])))
    B = distance(np.array(eval(mouth.iloc[2])),np.array(eval(mouth.iloc[4])))
    C = distance(np.array(eval(mouth.iloc[0])),np.array(eval(mouth.iloc[3])))

    mar = (A + B) / (2.0 * C)
    return mar

def calculate_angle(x,y) :
    return(-np.abs(np.arctan2(y,x)*180/np.pi)+180)

def hop(head):
    x = np.array(eval(head.iloc[0]))[0] - np.array(eval(head.iloc[1]))[0]
    y = np.array(eval(head.iloc[0]))[1] - np.array(eval(head.iloc[1]))[1]
    z = np.array(eval(head.iloc[0]))[2] - np.array(eval(head.iloc[1]))[2]
    angle_hb = calculate_angle(y,x)
    angle_gd = calculate_angle(z,y)
    return angle_gd, angle_hb

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



signes = ["EAR_left","EAR_right", "EAR_mean","MAR", "HOP_dg", "HOP_hb","Ferme","PERCLOS"]

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
    compteur_ferme = 0
    print(len(df))
    # Liste des EAR pour un seuil adaptatif
    EAR_list = []
    
    # Calculer l'EAR pour chaque œil et ajouter les valeurs à la liste
    for frame in range(len(df)) :
        eye_coordinates_left = df.loc[frame, ['left_eye_263','left_eye_387','left_eye_385','left_eye_362','left_eye_380','left_eye_373']]
        eye_coordinates_right = df.loc[frame, ['right_eye_133','right_eye_158','right_eye_160','right_eye_33','right_eye_144','right_eye_153','right_eye_144']]
        mouth_coordinates = df.loc[frame, ['mouth_78','mouth_82','mouth_312', 'mouth_308','mouth_317','mouth_87']]
        head_coordinates = df.loc[frame, ['head_10', 'head_152']]
        
        EAR_left = eye_aspect_ratio(eye_coordinates_left)
        EAR_right = eye_aspect_ratio(eye_coordinates_right)
        EAR_mean = (EAR_left + EAR_right) / 2

        EAR_list.append(EAR_mean)


        if frame <25 :
            EAR_list.append(EAR_mean) 
            ferme = EAR_mean < 0.2
        else :
            ferme = EAR_mean < np.mean(np.array(EAR_list))
            # On enlève le plus ancien élément
            EAR_list.pop(0)
            # On rajoute le nouveau
            EAR_list.append(EAR_mean)
        

        if ferme == True :
            compteur_ferme = compteur_ferme+1
        
        MAR = mouth_aspect_ratio(mouth_coordinates)
        HOP_dg, HOP_hb = hop(head_coordinates)
        PERCLOS = perclos(compteur_ferme,frame+1)

        # Ajouter les valeurs à la liste
        values.extend([EAR_left, EAR_right, EAR_mean, MAR, HOP_dg, HOP_hb, ferme, PERCLOS])

    # Ajouter des zéros pour égaler les longueurs
    values += [0] * (len(df_final.columns) - len(values))

    # Ajouter la nouvelle ligne au DataFrame
    df_final.loc[len(df_final)] = values

# Remplacer NaN par 0 dans le DataFrame final
df_final = df_final.fillna(0)

# Enregistrez le DataFrame final dans un fichier CSV
df_final.to_csv('resultat_final.csv', index=False)

