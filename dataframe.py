import numpy as np
import pandas as pd
from sources import signes 

# Ouverture du dataframe des coordonnées par groupe de 1000
df_coordinates_chunks = pd.read_csv("csv/videos_coordinates.csv",chunksize = 1000)

# Création des colonnes pour le dataframe final
list_col = ["nom_video"]
col_signes = ["EAR_left","EAR_right","EAR_mean","MAR","EBR","PERCLOS","HOP_gd", "HOP_hb"]
for i in range(875) :
    for signe in col_signes :
        list_col.append(f"{signe}_{i}")

# Ouverture du dataframe des résultats
df = pd.DataFrame(columns = list_col)

nb_chunk = 0
# Pour chaque chunk
for chunk in df_coordinates_chunks :
  # Parcourir les lignes
  print(f"Chunk {nb_chunk+1}...")
  for ligne in range(chunk.shape[0]):
      # Extraire le nom de la vidéo
      video_name = chunk.iloc[ligne, 0]

      # Initialiser une liste pour stocker le nom de la vidéo et les coordonnées de cette ligne
      video_and_coordinates = [video_name]
      print(f"Processing {ligne+1}/{chunk.shape[0]} {video_name}...")

      # On itère sur les colonnes
      for i in range(1, chunk.shape[1], 3):
          # Créer une liste avec les coordonnées x, y, z du point
          point_coordinates = [chunk.iloc[ligne, i], chunk.iloc[ligne, i+1], chunk.iloc[ligne, i+2]]

          # Ajouter les coordonnées à la liste des coordonnées
          video_and_coordinates.append(point_coordinates)

      # Maintenant que nous avons nos points, nous pouvons calculer les signes et les ajouter au dataframe
      results, list_ebr = signes.calculs_signes(video_and_coordinates)

      # Créer une liste avec le nom de la vidéo, les coordonnées et les résultats des signes
      row_data = [video_and_coordinates[0]] + results

      df.loc[len(df)] = row_data
      print(df.shape)

  nb_chunk +=1

df["classe"] = df["nom_video"].str[0]

# Enregistrer le DataFrame dans un fichier CSV
df.to_csv('csv/signes.csv', index=False)

print(df["classe"])

print("Terminé !")

