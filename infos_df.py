import pandas as pd 

df_coordinates = pd.read_csv("videos_coordinates.csv")

print(df_coordinates.head())
print(df_coordinates.shape)

# Nombre de NULL sur la ligne 0
nb_null0 = df_coordinates.iloc[0].isnull().sum()
print("Il y a",nb_null0,"cases vides")

# Nombre de NULL sur la ligne 1
nb_null1 = df_coordinates.iloc[1].isnull().sum()
print("Il y a",nb_null1,"cases vides")

# Nombre de NULL sur la ligne 2
nb_null2 = df_coordinates.iloc[2].isnull().sum()
print("Il y a",nb_null2,"cases vides")


