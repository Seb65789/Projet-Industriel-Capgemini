from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from hop import hop
from perclos import perclos
import matplotlib.pyplot as plt
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
    list_clignement = []
    eyes_state = "open"
    list_ebr = []
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
        
        # Seuil adaptatif =========================================================================================== 

        # Sur la première seconde 
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
            # À chaque frame on enlève le premier élement de la liste de la premiere seconde à la fin 
            list_clignement.pop(0)

        # ============================================================================================================

        # Détection des clignements ================================================================================
            
        # Si sur l'image actuel, les yeux sont considérés comme fermés alors on augmente le compteur 
        if ferme == True :
            compteur_ferme = compteur_ferme+1
            # si l'oeil était ouvert au début alors on compte un clignement
            if eyes_state == "open":
                eyes_state = "closed"
                # On ajoute le dernier élément
                list_clignement.append(1)
            
            # Si l'oeil était déjà fermé on ne compte pas un clignement de plus
            else :
                # On ajoute le dernier élément
                list_clignement.append(0)
        
        # Si l'oeil est considéré comme ouvert
        else:
            eyes_state = "open"
            # On ajoute le dernier élément
            list_clignement.append(0)

        # ==========================================================================================================
        
        # Calcul du perclos
        Perclos = perclos(compteur_ferme,compt_frame+1)

        # Calcul de l'EBR
        ebr = np.sum(list_clignement)
        list_ebr.append(ebr)
        
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
    
    return res, list_ebr

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
    video_and_coordinates = [video_name[7:]]
    print(f"Processing {video_name[7:]}...")

    # On itère sur les colonnes
    for i in range(1, df_coordinates.shape[1], 3):
        # Créer une liste avec les coordonnées x, y, z du point
        point_coordinates = [df_coordinates.iloc[ligne, i], df_coordinates.iloc[ligne, i+1], df_coordinates.iloc[ligne, i+2]]
        
        # Ajouter les coordonnées à la liste des coordonnées
        video_and_coordinates.append(point_coordinates)
    
    # Maintenant que nous avons nos points, nous pouvons calculer les signes et les ajouter au dataframe
    results, list_ebr = calculs_signes(video_and_coordinates)
    
    # Créez une nouvelle figure à chaque itération
    plt.figure()
    plt.plot(list_ebr)
    plt.title(f'Evolution des EBR pour {video_name[7:]}')
    plt.xlabel('Frames')
    plt.ylabel('Valeur EBR')
    plt.grid(True)
    plt.ylim(0, 7)
    # Sauvegardez le plot avec un nom de fichier unique
    plt.savefig(f"EBR_{video_name[7:]}.png")

    # Créer une liste avec le nom de la vidéo, les coordonnées et les résultats des signes
    row_data = [video_and_coordinates[0]] + results

    df.loc[len(df)] = row_data

df.to_csv('csv_videos/résultats.csv', index=False)
print(df)

print("Terminé !")