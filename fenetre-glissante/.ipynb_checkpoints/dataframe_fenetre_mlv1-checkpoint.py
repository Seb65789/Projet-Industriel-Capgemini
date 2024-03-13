from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from hop import hop
from perclos import perclos
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import os
from collections import deque
import joblib  # Pour charger le modèle RandomForest


# Charger le modèle RandomForest========================================================================================================================================
#model = joblib.load('modele_random_forest.pkl')
model = joblib.load('modele_random_forest_v3.pkl')
compteur_35s = 0


def calculs_signes(list_points, frame_count, list_ear, list_ebr, list_clignement):
    #print("Je suis rentré dans la fonction\n")
    # les 8*3+1 premieres valeurs sont pour les coordonnées de l'oeil droit
    # toutes les 3 valeurs correspondent à un point
    # Création de la liste des EAR
    ear_list = list_ear
    # Pour connaitre la frame courante chaque frame
    compt_frame = frame_count
    # Une liste pour stocker les resultats
    res = []
    ebr = -1
    # Compteur de yeux fermés
    compteur_ferme = 0
    # Compteur de yeux fermés reinitialisé toutes les 5 secondes
    #list_clignement = []
    eyes_state = "open"
    ebr_list = list_ebr

# Extraction des points ==================================================================================================================
    right_eye_coord = list_points[0:8]
    #print(right_eye_coord)
    left_eye_coord = list_points[8:16]
    mouth_coord = list_points[16:24]
    head_coord = list_points[24:27]
# ========================================================================================================================================
    #print("Liste des points fait\n")
# Calcul des signes ======================================================================================================================
    ear_right = eye_aspect_ratio(right_eye_coord)
    #print(f"ear_right = {ear_right}")
    ear_left =  eye_aspect_ratio(left_eye_coord)
    #print(f"ear_left = {ear_left}")
    mar = mouth_aspect_ratio(mouth_coord)
    #print(f"mar = {mar}")
    hop_hb , hop_gd = hop(head_coord)
    ear_mean = (ear_right + ear_left)/2
    
    # Seuil adaptatif ===========================================================================================
    #print("Signe de fatigue calculé\n")
    # Sur la première seconde
    if compt_frame <25 :
        ear_list.append(ear_mean)
        ferme = ear_mean < 0.2
        #print(ear_list)

    # Sur le reste du temps
    else :
        # chute de 10 %
        #print(ear_list)
        ferme = ear_mean < 0.9*np.mean(ear_list)
        # On enlève le plus ancien élément
        ear_list.pop(0)
        # On rajoute le nouveau
        ear_list.append(ear_mean)
        # À chaque frame on enlève le premier élement de la liste de la premiere seconde à la fin
        list_clignement.pop(0)

    # ============================================================================================================

    # Détection des clignements ================================================================================
    # Si sur l'image actuel, les yeux sont considérés comme fermés alors on augmente le compteur 
    if ferme == True :
        compteur_ferme = compteur_ferme+1
        # si l'oeil était ouvert au début alors on compte un clignement
        if eyes_state == "open":
            eyes_state = "closed"
            # On ajoute le dernier élément
            list_clignement.append(1)
        
        # Si l'oeil était déjà fermé on ne compte pas un clignement de plus
        else :
            # On ajoute le dernier élément
            list_clignement.append(0)
    
    # Si l'oeil est considéré comme ouvert
    else:
        eyes_state = "open"
        # On ajoute le dernier élément
        list_clignement.append(0)

    # ==========================================================================================================
    # Calcul du perclos
    Perclos = perclos(compteur_ferme,compt_frame+1)
    # Calcul de l'EBR
    ebr = np.sum(list_clignement)
    ebr_list.append(ebr)
    
#==========================================================================================================================================
    # On incrémente le compteur de frame
    # compt_frame += 1
    # On ajoute nos résultats à la liste des résultats
    res.append(ear_left)
    res.append(ear_right)
    res.append(ear_mean)
    res.append(mar)
    res.append(ebr)
    res.append(Perclos)
    res.append(hop_gd)
    res.append(hop_hb)
    
    return res, ebr_list, ear_list, list_clignement

# Les points qui nous interresse
right_eye = [33, 133, 160, 144, 159, 145, 158, 153]
left_eye = [263, 362, 387, 373, 386, 374, 385, 380]
mouth = [61,291,39,181,0,17,269,405]
head = [10,152]


# Création de la liste de points sans tuples 
list_points_no_tuples = right_eye + left_eye + mouth + head
print(list_points_no_tuples)

# Les vidéos à traiter
# video_paths = [f for f in os.listdir("videos") if f.endswith('.mp4')]
# video_paths = [os.path.join("videos", video) for video in video_paths]

# Declare FaceMesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)

# Le dessin du facemesh
# mp_drawing = mp.solutions.drawing_utils
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Création des colonnes pour le dataframe final
list_col = []
signes = ["EAR_left","EAR_right","EAR_mean","MAR","EBR","PERCLOS","HOP_gd", "HOP_hb"]
for i in range(875) :
    for signe in signes :
        list_col.append(f"{signe}_{i}")

# print(list_col)
deque_signe = deque(maxlen=7000)
# Création d'une liste pour pouvoir utiliser le dataframe ensuite
noms_colonnes = []
list_ear = []
list_ebr = []
list_clignement = []

# Ouverture du dataframe des résultats
df = pd.DataFrame(columns = list_col)

#print(df)

nouvelle_ligne = []

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
                nouvelle_ligne.append(data_point.x * frame.shape[1])
                nouvelle_ligne.append(data_point.y * frame.shape[0])
                nouvelle_ligne.append(data_point.z)

            # Maintenant que nous avons nos points, nous pouvons calculer les signes et les ajouter au dataframe
            results, list_ebr, list_ear, list_clignement = calculs_signes(nouvelle_ligne, frame_count, list_ear, list_ebr, list_clignement)
            
            #print("On est sorti de la fonction calculs_signes\n")
            
            # On ajoute nos résultats au deque les résultats
            
            # Ajouter les valeurs à chaque emplacement du deque
            for value in results:
                deque_signe.append(value)
                
            #print(deque_signe)

            # On créer une liste d'après la deque
            noms_colonnes = list(deque_signe)
            #print(noms_colonnes)


            # On change le dataframe ensuite selon la fenetre glissante
            df = pd.DataFrame([noms_colonnes])
            #On récupère le nombre de colonnes
            # print("Colonnes: ", len(df.columns)) 
            if len(df.columns) == 7000:
                #and frame_count % 875 == 0
                compteur_35s += 1
                #j'ai rajouté 
                #print("avant prédiction : ", df)
                #df.columns = list_col[:len(noms_colonnes)]
                #print("après prédiction : ", df)
                #print("compteur", compteur_35s)
#Ajouter la prédiction=======================================================================================================================================
                #Remplacer df_5260 par df
                #df_5260 = df.iloc[:, :5260]
                #prediction = model.predict(df_5260)
                prediction = model.predict(df)

                #print("prediction",prediction)
                #print("predictiontype",type(prediction))
                #numpy.ndarray
                #print("prediction [0]",prediction[0])
                #mettre du rouge si somnolent ,bleu si c'est intermédiaire et vert si c'est alerte
                if prediction[0]==2:
                    cv2.putText(frame, "Somnolent", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) #Rouge 
                elif prediction[0]==1:
                    cv2.putText(frame, "Intermédiaire", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  #Bleu , (255, 0, 0)   #Vert(0, 255, 0), 2) 
                else :
                    cv2.putText(frame, "Alerte", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)   #Vert   (0, 0, 255), 2) #Rouge 
                
                #cv2.imshow('OpenCV Feed', frame)
                # Convertir l'image de BGR à RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imshow("Caméra", frame_rgb)
              
                #Quelle est le type de predction ? int 
                #cv2.putText(frame, str(prediction), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)     
            #print(df)
  
            nouvelle_ligne = []

            # Rajouter ici la boucle pour changer la liste en tournant
            frame_count +=1
            print(f"frame_count = {frame_count}")
            
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

#df.columns = list_col[:len(noms_colonnes)]
print(df)

print("Terminé !")