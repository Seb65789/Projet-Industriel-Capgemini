import os

def get_videos(chemin_dossier, fichier_sortie):
    # Liste tous les fichiers du dossier
    fichiers_videos = ['./videos/' + f for f in os.listdir(chemin_dossier) if f.endswith('.mp4')]

    # Écrit les noms des fichiers dans le fichier texte
    with open(fichier_sortie, 'w') as fichier:
        for nom_fichier in fichiers_videos:
            fichier.write(f"{nom_fichier}\n")

    return fichiers_videos

def get_landmarks(chemin_dossier, fichier_sortie):
    # Liste tous les fichiers du dossier
    fichiers_csv = ['./csv_videos/' + f for f in os.listdir(chemin_dossier) if f.endswith('.csv')]

    # Écrit les noms des fichiers dans le fichier texte
    with open(fichier_sortie, 'w') as fichier:
        for nom_fichier in fichiers_csv:
            fichier.write(f"{nom_fichier}\n")

    return fichiers_csv