import Fonction_seuil
import pandas as pd
import Perclos

df_EAR_MAR = pd.read_csv("resultat_final.csv")

nbL,nbC = df_EAR_MAR.shape

# Pour renommer les colonnes du dataframe 
nouvelles_colonnes = ["Nom vidéo"]

for x in range(876):
    nouvelles_colonnes.append(f'ferme_left_{x}')
    nouvelles_colonnes.append(f'ferme_right_{x}')
    nouvelles_colonnes.append(f'MAR_{x}')
    nouvelles_colonnes.append(f'ferme_mean_{x}')

# Assigne les nouvelles colonnes au DataFrame
df_EAR_MAR.columns = nouvelles_colonnes

print(df_EAR_MAR.columns)
for x in range(876) :
    if x >= 30:
        df_EAR_MAR[f'ferme_left_{x}'] = df_EAR_MAR[f'ferme_left_{x}'].apply(lambda valeur: True if valeur < 0.15 else False)
        df_EAR_MAR[f'ferme_right_{x}'] = df_EAR_MAR[f'ferme_right_{x}'].apply(lambda valeur: True if valeur < 0.15 else False)
        df_EAR_MAR[f'ferme_mean_{x}'] = df_EAR_MAR[f'ferme_mean_{x}'].apply(lambda valeur: True if valeur < 0.15 else False)
    else:
        df_EAR_MAR[f'ferme_left_{x}'] = df_EAR_MAR[f'ferme_left_{x}'].apply(lambda valeur: True if valeur < 0.15 else False)
        df_EAR_MAR[f'ferme_right_{x}'] = df_EAR_MAR[f'ferme_right_{x}'].apply(lambda valeur: True if valeur < 0.15 else False)
        df_EAR_MAR[f'ferme_mean_{x}'] = df_EAR_MAR[f'ferme_mean_{x}'].apply(lambda valeur: True if valeur < 0.15 else False)


df_EAR_MAR.to_csv("ferme.csv")