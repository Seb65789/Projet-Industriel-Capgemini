# Importer la bibliothèque nécessaire
import pandas as pd

# Charger les données depuis le fichier CSV
data = pd.read_csv('resultat_final.csv')

# Afficher les premières lignes du dataframe pour vérifier le chargement
print("Étape 1 - Chargement des données :")
print(data.head())

#Etape 2

# Extrait le chiffre après le tiret dans la colonne 'Nom_video'
data['KSS'] = data['Nom_video'].str.extract('kss#\d+-(\d+)').astype(float).fillna(0).astype(int)

# Définir les plages KSS pour chaque niveau de somnolence
bins = [0, 3, 6, 9]
labels = ['alerte', 'intermédiaire', 'somnolent']

# Ajouter une colonne 'niveau_somnolence' basée sur les plages KSS
data['niveau_somnolence'] = pd.cut(data['KSS'], bins=bins, labels=labels)

# Afficher les premières lignes du dataframe après l'ajout de la colonne
print("\nÉtape 2 - Ajout de la colonne 'niveau_somnolence' :")
print(data.head())


#Etape 3 : 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Étape 3: Extrait les colonnes appropriées pour X et y
X_columns = ['EAR_left_{0}'.format(i) for i in range(876)] + \
            ['EAR_right_{0}'.format(i) for i in range(876)] + \
            ['EAR_mean_{0}'.format(i) for i in range(876)] + \
            ['MAR_{0}'.format(i) for i in range(876)]

X = data[X_columns]
y = data['niveau_somnolence']

# Afficher les premières lignes de X et y pour vérifier
print("\nÉtape 3 - Colonnes pour X et y :")
print("X.head():\n", X.head())
print("y.head():\n", y.head())

# Étape 4: Séparer les données en ensemble d'entraînement et ensemble de test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Afficher les dimensions des ensembles d'entraînement et de test
print("\nÉtape 4 - Séparation des ensembles d'entraînement et de test :")
print("Dimensions de l'ensemble d'entraînement X:", X_train.shape)
print("Dimensions de l'ensemble de test X:", X_test.shape)
print("Dimensions de l'ensemble d'entraînement y:", y_train.shape)
print("Dimensions de l'ensemble de test y:", y_test.shape)

# Étape 5: Créer un modèle Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
# Étape 6: Entraîner le modèle sur l'ensemble d'entraînement
model.fit(X_train, y_train)
# Étape 7: Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)
# Mesurer la précision du modèle
accuracy = accuracy_score(y_test, y_pred)
print("\nÉtape 7 - Évaluation du modèle :")
print("Précision du modèle : {:.2f}%".format(accuracy * 100))


