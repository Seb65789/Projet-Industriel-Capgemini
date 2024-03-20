# Importation des fichiers sources pertant le calcul des caractéristiques
from sources import EAR  
from sources import MAR
from sources import hop
from sources import signes
from sources import distance_euclidienne
from sources import PERCLOS

import sklearn
print(sklearn.__version__)
import cv2
import mediapipe as mp
from collections import deque


# Les points qui nous interresse
right_eye = [33, 133, 160, 144, 159, 145, 158, 153]
left_eye = [263, 362, 387, 373, 386, 374, 385, 380]
mouth = [61,291,39,181,0,17,269,405]
head = [10,152]

# Création de la liste de points sans tuples 
list_points_no_tuples = right_eye + left_eye + mouth + head

# Declare FaceMesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)

# print(list_col)
deque_signe = deque(maxlen=7000)
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
            #print(np_signe)
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
            

            # Afficher l'image avec le texte
            cv2.imshow('Affichage du texte', frame)


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


# Release resources
cap.release()

# Close all windows and release the face mesh model
cv2.destroyAllWindows()
face_mesh.close()


print("Terminé !")