# Importation des fichiers src pertant le calcul des caractéristiques
from src import EAR  
from src import MAR
from src import hop
from src import signes
from src import distance_euclidienne
from src import PERCLOS

import matplotlib.pyplot as plt
import os
import cv2
import mediapipe as mp
from collections import deque
import joblib  # Pour charger le modèle RandomForest
import plotly.graph_objs as go # Pour les graphiques

version = input("Entrez le numéro de la version du modèle ( 00 avec les paramètres par défaut/précision 0.73 ou 01 avec les hyperparamètres 0.75) :")

# Charger le modèle RandomForest========================================================================================================================================
#model = joblib.load('modele_random_forest.pkl')
model = joblib.load(f'modele_ml/modele_random_forest_v{version}.pkl')

# Initialiser la variable de prédiction actuelle 
current_prediction = None
all_predictions = []  # Liste pour stocker toutes les prédictions

# Importer MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh

# Initialiser FaceMesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5
)

# Créer la figure pour le graphique interactif#===============================================================================
# Importer les fonctions de dessin de MediaPipe
mp_drawing = mp.solutions.drawing_utils
fig = go.Figure()
# Initialiser la trace pour le graphique de dispersion
scatter_trace = go.Scatter(x=[], y=[], mode='lines+markers', name='Prédictions en fonction du nombre d\'images')
# Ajouter le numéro de frame à la liste des valeurs sur l'axe des abscisses
x_values = []  # Créer une liste vide pour stocker les valeurs de frame_count
# Ajouter la trace à la figure
fig.add_trace(scatter_trace)

# Mettre à jour la mise en page de la figure
fig.update_layout(title='Prédictions en fonction du nombre d\'images',
                  xaxis_title='Nombre d\'images',
                  yaxis_title='Prédiction')


# Les points qui nous interresse
right_eye = [33, 133, 160, 144, 159, 145, 158, 153]
left_eye = [263, 362, 387, 373, 386, 374, 385, 380]
mouth = [61,291,39,181,0,17,269,405]
head = [10,152]

# Création de la liste de points sans tuples 
list_points_no_tuples = right_eye + left_eye + mouth + head



deque_signe = deque(maxlen=7000)
# Création d'une liste pour pouvoir utiliser le dataframe ensuite
np_signe = []
list_ear = []
list_ebr = []
list_perclos = []
list_ferme = []
list_clignement = []
eyes_state = "open"
# Les coordonnées de la frame
coordonnees = []

cap = cv2.VideoCapture(0)

frame_count = 0  # Initialize frame count
#Lancer une vidéo ou un démonstrateur en direct================================================================
# Demander à l'utilisateur s'il souhaite utiliser la caméra en direct ou une vidéo
source_choice = input("Voulez-vous utiliser la caméra en direct (tapez 'cam') ou une vidéo dans le dossier 'videos' (tapez 'video') ? ")

if source_choice.lower() == 'cam':
    # Utiliser la caméra en direct
    cap = cv2.VideoCapture(0)
elif source_choice.lower() == 'video':
    # Vérifier si le dossier 'videos' existe
    if not os.path.exists('videos'):
        print("Le dossier 'videos' n'existe pas.")
        exit()

    # Lister les fichiers dans le dossier 'videos'
    video_files = os.listdir('videos')

    # Vérifier s'il y a des fichiers vidéo dans le dossier 'videos'
    if not video_files:
        print("Aucune vidéo trouvée dans le dossier 'videos'.")
        exit()

    # Afficher les vidéos disponibles
    print("Vidéos disponibles :")
    for i, video_file in enumerate(video_files):
        print(f"{i+1}. {video_file}")

    # Demander à l'utilisateur de choisir une vidéo
    video_choice = input("Entrez le numéro de la vidéo que vous souhaitez utiliser : ")

    # Valider l'entrée de l'utilisateur
    try:
        video_index = int(video_choice) - 1
        if video_index < 0 or video_index >= len(video_files):
            print("Numéro de vidéo invalide.")
            exit()
    except ValueError:
        print("Numéro de vidéo invalide.")
        exit()

    # Construire le chemin complet de la vidéo choisie
    video_path = os.path.join('videos', video_files[video_index])

    # Initialiser l'objet VideoCapture pour la vidéo choisie
    cap = cv2.VideoCapture(video_path)
else:
    print("Choix invalide.")
    exit()

# Créer un objet VideoWriter
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Process video frames
while True:
    ret, frame = cap.read()

    # Vérifier si la lecture est réussie
    if not ret:
        print("Erreur: Impossible de lire la caméra.")
        break
    
    # Afficher l'image dans une fenêtre
    cv2.imshow("Caméra", frame)
    
    # Convertir l'image en RGB pour améliorer la précision de Mediapipe
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Détecter les visages dans l'image
    faces = face_mesh.process(frame)

    
    if faces.multi_face_landmarks:
        # Dans votre boucle while où vous dessinez les landmarks du visage :
        for face_landmarks in faces.multi_face_landmarks:
            # Dessiner les landmarks du visage (grille facemesh)
            # Dans votre boucle while où vous dessinez les landmarks du visage :
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing.DrawingSpec(color=(128,128,128), thickness=1, circle_radius=1))  # Utiliser le style par défaut pour le facemesh en gris

            landmarks_positions = []
            for landmark_id in list_points_no_tuples:
                # Accéder aux coordonnées x, y et z du landmark spécifique
                data_point = face_landmarks.landmark[landmark_id]
                point = []
                point.append(data_point.x * frame.shape[1])
                point.append(data_point.y * frame.shape[0])
                point.append(data_point.z)
                coordonnees.append(point)

            # Maintenant que nous avons nos points, nous pouvons calculer les signes et les ajouter au dataframe
            results, list_ear, list_ferme,list_clignement,eyes_state,list_perclos,list_ebr = signes.calculs_signes_live(coordonnees, frame_count, list_ear,list_ferme, list_clignement,eyes_state,list_perclos,list_ebr)
            
            
            # Ajouter les valeurs à chaque emplacement du deque
            for value in results:
                deque_signe.append(value)
                
            #print(deque_signe)

            # On créer une liste d'après la deque
            np_signe = list(deque_signe)
            #1 seconde = 25 frames
            #si on utilise pas HOP on a 5250
            if (len(np_signe) == 7000) and frame_count % 25 == 0:
                prediction = model.predict([np_signe])
                print(f"Prédiction : {prediction}")
                # Mettre à jour la prédiction actuelle
                current_prediction = prediction[0]
                print("Prédiction actuelle ajoutée à la liste des prédictions (calculée toutes les secondes):", current_prediction)
                # Ajouter la prédiction à la liste de toutes les prédictions
                all_predictions.append(current_prediction)
                # Ajouter le print à la liste de logs
                x_values.append(frame_count)
                

            # Afficher la prédiction actuelle
            #print("Prédiction actuelle affichée:", current_prediction)
            if current_prediction is not None:
                if current_prediction == 3:
                    cv2.putText(frame, "Somnolent !", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # Rouge
                elif current_prediction == 2:
                    cv2.putText(frame, "Intermediaire", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Bleu
                else:
                    cv2.putText(frame, "Alerte", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Vert (0, 255, 0)
                   
          
            coordonnees = []

            # Rajouter ici la boucle pour changer la liste en tournant
            frame_count +=1
            
            print(f"frame_count = {frame_count}")

            # Ajouter du texte à l'image
            texte = f"EAR :{results[2]}" 
            position = (50, 50)  # Position du texte dans l'image
            couleur = (255, 255, 255)  # Couleur du texte (blanc)
            epaisseur = 2  # Épaisseur du texte
            taille_police = 1  # Taille de la police
            cv2.putText(frame, texte, position, cv2.FONT_HERSHEY_SIMPLEX, taille_police, couleur, epaisseur)


            texte = f"EBR : {results[4]}"
            position = (50, 90)  # Position du texte dans l'image
            couleur = (0, 0, 255)  # Couleur du texte (blanc)
            cv2.putText(frame, texte, position, cv2.FONT_HERSHEY_SIMPLEX, taille_police, couleur, epaisseur)
            texte = f"PERCLOS : {results[5]}"
            position = (50, 130)  # Position du texte dans l'image
            couleur = (0, 255, 0)  # Couleur du texte (blanc)
            cv2.putText(frame, texte, position, cv2.FONT_HERSHEY_SIMPLEX, taille_police, couleur, epaisseur)
            
            # Convertir l'image de BGR à RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Afficher l'image avec le texte
            cv2.imshow('Affichage du texte', frame)
        
        # Enregistrer la trame dans la vidéo de sortie
        out.write(frame)

        # Attendre 1 milliseconde et vérifier si l'utilisateur a appuyé sur 'Esc', 'q', ou toute autre touche
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            print("Breaking both loops because 'Esc' or 'q' is pressed.")
            
            tailles_perclos = list(range(1, len(list_perclos) + 1))
            plt.figure()
            plt.plot(tailles_perclos, list_perclos)
            plt.title("Graphique du PERCLOS en fonction du temps")
            plt.xlabel("Temps en frame")
            plt.ylabel("PERCLOS")
            plt.savefig('PERCLOS.png')
            plt.ylim(0,30)
    
            tailles_ebr = list(range(1, len(list_ebr) + 1))
            plt.figure()
            plt.plot(tailles_ebr, list_ebr)
            plt.title("Graphique de l'EBR en fonction du temps")
            plt.xlabel("Temps en frame")
            plt.ylabel("EBR")
            plt.savefig('EBR.png')

            exit_outer_loop = True  # Définir la variable de contrôle pour sortir des deux boucles
            break
        elif key != -1:
            print(f"Breaking both loops because key {key} is pressed.")
            exit_outer_loop = True  # Définir la variable de contrôle pour sortir des deux boucles
            break
# Release resources

out.release()
cap.release()

# Close all windows and release the face mesh model
cv2.destroyAllWindows()
face_mesh.close()
print("Terminé !")