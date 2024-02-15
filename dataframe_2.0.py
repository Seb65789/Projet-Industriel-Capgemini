from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from hop import hop
from perclos import perclos
import numpy as np
import pandas as pd

def calculs_signes(list_points) :
    # les 8*3+1 premieres valeurs sont pour les coordonnées de l'oeil droit
    # toutes les 3 valeurs correspondent à un point
    # Création de la liste des EAR
    ear_list = []
    # Pour connaitre la frame courante chaque frame
    compt_frame = 0
    # Une liste pour stocker les resultats
    res = []
    ebr = -1
    # Compteur de yeux fermés
    compteur_ferme = 0
    # Compteur de yeux fermés reinitialisé toutes les 5 secondes
    compteur_ferme_EBR = 0
    eyes_state = "open"
    for i in range(1,len(list_points),8*3+2) :

# Extraction des points ==================================================================================================================
        right_eye_coord = list_points[i:i+8]
        left_eye_coord = list_points[i+8:i+16]
        mouth_coord = list_points[i+16:i+24]
        head_coord = list_points[i+24:i+26]
# ========================================================================================================================================
        
# Calcul des signes ======================================================================================================================
        ear_right = eye_aspect_ratio(right_eye_coord)
        ear_left =  eye_aspect_ratio(left_eye_coord)
        mar = mouth_aspect_ratio(mouth_coord)
        hop_hb , hop_gd = hop(head_coord) 
        ear_mean = (ear_right + ear_left)/2
        
        # Sur la première seconde on crée la liste
        if compt_frame <25 :
            ear_list.append(ear_mean) 
            ferme = ear_mean < 0.2
        
        # Sur le reste du temps
        else :
            # chute de 10 %
            ferme = ear_mean < 0.9*np.mean(np.array(ear_list))
            # On enlève le plus ancien élément
            ear_list.pop(0)
            # On rajoute le nouveau
            ear_list.append(ear_mean)

        # Si sur l'image actuel, les yeux sont considérés comme fermés alors on augmente le compteur 
        if ferme == True :
            compteur_ferme = compteur_ferme+1
            # si l'oeil était ouvert au début alors on compte un clignement
            if eyes_state == "open":
                    eyes_state = "closed"
                    compteur_ferme_EBR +=1
            # Si l'oeil était déjà fermé
        else:
            eyes_state = "open"

        
        # Calcul du perclos
        Perclos = perclos(compteur_ferme,compt_frame+1)
        if compt_frame ==874 :
            print("Perclos :",Perclos)

        # Calcul EBR, Eye-Blik Rate (fréquence toutes les 5 du nombre de ferme)
        if(compt_frame+1)%125==0 :
            ebr = compteur_ferme_EBR
            compteur_ferme_EBR = 0
            
        else :
            ebr= -1
        
#==========================================================================================================================================
        # On incrémente le compteur de frame
        compt_frame += 1

        # On ajoute nos résultats à la liste des résultats
        res.append(ear_left)
        res.append(ear_right)
        res.append(ear_mean)
        res.append(mar)
        res.append(ferme)
        res.append(ebr)
        res.append(Perclos)
        res.append(hop_gd)
        res.append(hop_hb)
        
            
    
    return res

# Ouverture du dataframe des coordonnées 
df_coordinates = pd.read_csv("csv_videos/videos_coordinates.csv")

# Création des colonnes pour le dataframe final
list_col = ["nom_video"]
signes = ["EAR_left","EAR_right","EAR_mean","MAR","Ferme","EBR","PERCLOS","HOP_gd", "HOP_hb"]
for i in range(875) :
    for signe in signes :
        list_col.append(f"{signe}_{i}")

# Ouverture du dataframe des résultats
df = pd.DataFrame(columns = list_col)

# Parcourir les lignes
for ligne in range(df_coordinates.shape[0]):
    # Extraire le nom de la vidéo
    video_name = df_coordinates.iloc[ligne, 0]

    # Initialiser une liste pour stocker le nom de la vidéo et les coordonnées de cette ligne
    video_and_coordinates = [video_name]
    print(f"Processing {video_name}...")

    # On itère sur les colonnes
    for i in range(1, df_coordinates.shape[1], 3):
        # Créer une liste avec les coordonnées x, y, z du point
        point_coordinates = [df_coordinates.iloc[ligne, i], df_coordinates.iloc[ligne, i+1], df_coordinates.iloc[ligne, i+2]]
        
        # Ajouter les coordonnées à la liste des coordonnées
        video_and_coordinates.append(point_coordinates)
    
    # Maintenant que nous avons nos points, nous pouvons calculer les signes et les ajouter au dataframe
    results = calculs_signes(video_and_coordinates)
    


    # Créer une liste avec le nom de la vidéo, les coordonnées et les résultats des signes
    row_data = [video_and_coordinates[0]] + results

    df.loc[len(df)] = row_data

import pandas as pd

# Créer un MultiIndex pour le DataFrame temporaire
iterables = [[i for i in range(0, 875)], ["EAR_left", "EAR_right", "EAR_mean", "MAR", "Ferme","EBR", "PERCLOS", "HOP_gd", "HOP_hb"]]
df_temp_index = pd.MultiIndex.from_product(iterables, names=["frame", "sign"])

# Créer un DataFrame temporaire avec les index multi-niveaux
data_temp = pd.DataFrame(columns=df_temp_index)

# Ajouter une colonne pour les noms de vidéos
data_temp["nom_video"] = ""

# Parcourir les lignes du DataFrame d'origine
for index, row in df.iterrows():
    video_name = row["nom_video"]
    # Ajouter le nom de la vidéo comme valeur dans la colonne "nom_video" pour chaque ligne
    data_temp.loc[index, "nom_video"] = video_name
    # Parcourir les colonnes et ajouter les valeurs correspondantes au DataFrame temporaire
    for i in range(0, 875):
        for sign in ["EAR_left", "EAR_right", "EAR_mean", "MAR", "Ferme","EBR", "PERCLOS", "HOP_gd", "HOP_hb"]:
            col_name = f"{sign}_{i}"
            data_temp.loc[index, (i, sign)] = row[col_name]


# Afficher le DataFrame temporaire
print(data_temp)

# Convertir le DataFrame en fichier CSV
data_temp.to_csv('csv_videos/résultats.csv', index=False)

print("Terminé !")