import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Supposez que vous avez un DataFrame appelé df avec vos caractéristiques et les étiquettes correspondantes
# Remplacez cela par vos propres données
df = pd.read_csv("resultat_final.csv")  # Remplacez cela par votre propre chargement de données

# Extrait le chiffre après le tiret dans la colonne 'Nom_video'
df['KSS'] = df['Nom_video'].str.extract('kss#\d+-(\d+)').astype(float).fillna(0).astype(int)

# Définir les plages KSS pour chaque niveau de somnolence
bins = [0, 3, 6, 9]
labels = ['alerte', 'intermédiaire', 'somnolent']

# Ajouter une colonne 'niveau_somnolence' basée sur les plages KSS
df['niveau_somnolence'] = pd.cut(df['KSS'], bins=bins, labels=labels)

# Utilisez une boucle for pour créer la liste de colonnes
caract_colonnes = [f"EAR_left_{x}" for x in range(1, 876)] + \
                  [f"EAR_right_{x}" for x in range(1, 876)] + \
                  [f"EAR_mean_{x}" for x in range(1, 876)] + \
                  [f"MAR_{x}" for x in range(1, 876)]

# Séparez les caractéristiques (X) et les étiquettes (y)
X = df[caract_colonnes]
colonne_etiquette = ["somnolent","pas somnolent"]
y = df['niveau_somnolence']  # Remplacez 'target_column' par le nom réel de votre colonne d'étiquettes

# Supprimez les lignes contenant des valeurs NaN dans X et y
X = X.dropna()
y = y[X.index]

# Divisez les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créez le modèle SVM avec un noyau linéaire (vous pouvez choisir d'autres types de noyaux)
svm_model = SVC(kernel='linear')  # Vous pouvez ajuster d'autres paramètres également

# Entraînez le modèle sur l'ensemble d'entraînement
svm_model.fit(X_train, y_train)

# Faites des prédictions sur l'ensemble de test
predictions = svm_model.predict(X_test)

# Évaluez les performances du modèle
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')
