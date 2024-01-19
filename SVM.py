import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# chargement des données
df = pd.read_csv("resultat_final.csv") 

# Extraction du niveau KSS de la video et ajout d'une plage 
df['KSS'] = df['Nom_video'].astype(str).str.extract('kss#\d+-(\d+)').astype(float).fillna(0).astype(int)


# Définir les plages KSS pour chaque niveau de somnolence
bins = [0, 3, 6, 9]
labels = ['alerte', 'intermédiaire', 'somnolent']

# Niveau de somnolence basée sur les plages KSS
df['niveau_somnolence'] = pd.cut(df['KSS'], bins=bins, labels=labels)

#Extraction des cractéristiques de somnolence
caract_colonnes = [f"EAR_left_{x}" for x in range(1, 875)] + \
                  [f"EAR_right_{x}" for x in range(1, 875)] + \
                  [f"EAR_mean_{x}" for x in range(1, 875)] + \
                  [f"MAR_{x}" for x in range(1, 875)]

# Nos caractéristiques
X = df[caract_colonnes]
# Nos étiquettes 
# Définir les plages KSS pour chaque niveau de somnolence
df['niveau_somnolence'] = pd.cut(df['KSS'], bins=bins, labels=labels).astype(str)

# Nos étiquettes 
y = df['niveau_somnolence']

# Suppression des NaN dans les données
# Supprimez les lignes contenant des valeurs NaN dans X et y
X = X.dropna(0)
y = y[X.index]

# Séparation des données pour avoir 20% de test et 80% d'entraînement pour les étiquettes et les caractéristiques
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du modèle SVM avec un noyau linéaire 
svm_model = SVC(kernel='linear')  # Vous pouvez ajuster d'autres paramètres également

# Entraînez le modèle sur l'ensemble d'entraînement
svm_model.fit(X_train, y_train)

# Faites des prédictions sur l'ensemble de test
predictions = svm_model.predict(X_test)

# Évaluez les performances du modèle
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')
