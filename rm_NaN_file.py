import os
import pandas as pd

def rm_NaN_CSV(dossier_csv, seuil_nan, nombre_lignes_attendu):
    fichiers_csv = [f for f in os.listdir(dossier_csv) if f.endswith('.csv')]

    for fichier in fichiers_csv:
        chemin_fichier = os.path.join(dossier_csv, fichier)

        # Charger le fichier CSV dans un DataFrame
        df = pd.read_csv(chemin_fichier)

        # Compter le nombre de valeurs NaN dans le DataFrame
        nb_nan = df.isna().sum().sum()

        # Vérifier si le nombre de NaN est supérieur au seuil
        # ou si le nombre de lignes est différent du nombre attendu
        if nb_nan > seuil_nan or 875-len(df) > nombre_lignes_attendu:
            os.remove(chemin_fichier)
            print(f"Fichier {fichier} supprimé. Nombre de NaN : {nb_nan}, Nombre de lignes : {len(df)}")
            return fichier  # Retourne le nom du fichier supprimé

    return None

def rm_NaN_MP4(dossier_video, fichier_video):
    if fichier_video:
        # Récupération du path du fichier
        chemin_fichier = os.path.join(dossier_video, fichier_video)

        # Supprime le fichier vidéo
        os.remove(chemin_fichier)
        print(f"Fichier {fichier_video} supprimé.")

def rm_NaN(dossier_csv, dossier_video, seuil_nan,nombre_lignes_attendu):
    fichier_csv_supprime = rm_NaN_CSV(dossier_csv, seuil_nan, nombre_lignes_attendu)
    
    
    # Vérifiez si la fonction a supprimé un fichier avant d'appeler rm_NaN_MP4
    if fichier_csv_supprime:
        # On remet le nom tel qu'il est dans le dossier video
        fichier_csv_supprime = fichier_csv_supprime[:-14] + '.mp4'
        rm_NaN_MP4(dossier_video, fichier_csv_supprime)
