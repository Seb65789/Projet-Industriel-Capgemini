#demonstrateursanshop.py vrai 10/04/2024
# Importation des fichiers src pertant le calcul des caractéristiques
from src import EAR  
from src import MAR
#from src import hop
from src import signes
from src import distance_euclidienne
from src import PERCLOS

import cv2
import mediapipe as mp
from collections import deque
import joblib  # Pour charger le modèle RandomForest
import plotly.graph_objs as go # Pour les graphiques


version = input("Entre le numéro de la version du modèle (3 avec les paramètres par défaut ou 4 avec les hyperparamètres) :")

# Charger le modèle RandomForest========================================================================================================================================
#model = joblib.load('modele_random_forest.pkl')
model = joblib.load(f'modele_ml/modele_random_forest_v{version}.pkl')

# Initialiser la variable de prédiction actuelle 
current_prediction = None
all_predictions = []  # Liste pour stocker toutes les prédictions

# Importer MediaPipe FaceMesh========================================================================================================================================
mp_face_mesh = mp.solutions.face_mesh

# Initialiser FaceMesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5
)

# Importer les fonctions de dessin de MediaPipe
mp_drawing = mp.solutions.drawing_utils



# Initialiser la figure pour le graphique interactif========================================================================================================================================
# fig = go.Figure()
# # Initialiser la trace pour le graphique de dispersion
# scatter_trace = go.Scatter(x=[], y=[], mode='lines+markers', name='Prédictions en fonction du nombre d\'images')

# Ajouter la trace à la figure
# fig.add_trace(scatter_trace)

# # Mettre à jour la mise en page de la figure
# fig.update_layout(title='Prédictions en fonction du nombre d\'images',
#                   xaxis_title='Nombre d\'images',
#                   yaxis_title='Prédiction')



# Les points qui nous interresse
right_eye = [33, 133, 160, 144, 159, 145, 158, 153]
left_eye = [263, 362, 387, 373, 386, 374, 385, 380]
mouth = [61,291,39,181,0,17,269,405]


# Création de la liste de points sans tuples 
list_points_no_tuples = right_eye + left_eye + mouth 

# Declare FaceMesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)

# print(list_col)
deque_signe = deque(maxlen=5250)
#deque_signe = deque(maxlen=7000)
# Création d'une liste pour pouvoir utiliser le dataframe ensuite
np_signe = []
list_ear = []
list_ebr = []
list_ferme = []
list_clignement = []
eyes_state = "open"
# Les coordonnées de la frame
coordonnees = []

cap = cv2.VideoCapture(0)

frame_count = 0  # Initialize frame count

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
        for face_landmarks in faces.multi_face_landmarks:
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
            results, list_ear, list_ferme,list_clignement,eyes_state = signes.calculs_signes_live(coordonnees, frame_count, list_ear,list_ferme, list_clignement,eyes_state)
            
            #print("On est sorti de la fonction calculs_signes\n")
            
            # On ajoute nos résultats au deque les résultats
            
            # Ajouter les valeurs à chaque emplacement du deque
            for value in results:
                deque_signe.append(value)
                
            #print(deque_signe)

            # On créer une liste d'après la deque
            np_signe = list(deque_signe)
            #1 seconde = 25 frames
            #si on utilise pas HOP on a 5250
            if (len(np_signe) == 5250) and frame_count % 25 == 0:
            #if (len(np_signe) == 7000) and frame_count % 25 == 0:
                prediction = model.predict([np_signe])
                print(f"Prédiction : {prediction}")
                # Mettre à jour la prédiction actuelle
                current_prediction = prediction[0]
                print("Prédiction actuelle ajoutée à la liste des prédictions (calculée toutes les secondes):", current_prediction)
                # Ajouter la prédiction à la liste de toutes les prédictions
                all_predictions.append(current_prediction)
                # Ajouter le print à la liste de logs
                

            # Afficher la prédiction actuelle
            #print("Prédiction actuelle affichée:", current_prediction)
            if current_prediction is not None:
                if current_prediction == 3:
                    cv2.putText(frame, "Somnolent", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # Rouge
                elif current_prediction == 2:
                    cv2.putText(frame, "Intermediaire", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Bleu
                else:
                    cv2.putText(frame, "Alerte", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Vert (0, 255, 0)
                   
            #print("Évolution des prédictions : ", all_predictions)
            

        
        

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

            # graphique 
            # if all_predictions:
            #         # Ajouter les nouvelles données à la trace du graphique de dispersion
            #         scatter_trace.x = list(range(len(all_predictions)))
            #         scatter_trace.y = all_predictions

            #         # Mettre à jour la mise en page de la figure
            #         fig.update_layout(xaxis_range=[0, len(all_predictions)])

            #         # Afficher la figure mise à jour
            #         fig.show()

      

        # Attendre 1 milliseconde et vérifier si l'utilisateur a appuyé sur 'Esc', 'q', ou toute autre touche
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            print("Breaking both loops because 'Esc' or 'q' is pressed.")
            exit_outer_loop = True  # Définir la variable de contrôle pour sortir des deux boucles
            break
        elif key != -1:
            print(f"Breaking both loops because key {key} is pressed.")
            exit_outer_loop = True  # Définir la variable de contrôle pour sortir des deux boucles
            break


# Release resrc
cap.release()

# Close all windows and release the face mesh model
cv2.destroyAllWindows()
face_mesh.close()



print("Terminé !")