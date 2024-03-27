import numpy as np
from sources import PERCLOS
from sources import hop
from sources import EAR
from sources import MAR

def calculs_signes(list_points) :
    # les 8*3+1 premieres valeurs sont pour les coordonnées de l'oeil droit
    # toutes les 3 valeurs correspondent à un point
    # Création de la liste des EAR
    ear_list = []
    # Pour connaitre la frame courante chaque frame
    compt_frame = 0
    # Une liste pour stocker les resultats
    res = []
    ebr = -1
    # Compteur de yeux fermés
    compteur_ferme = 0
    # Compteur de yeux fermés reinitialisé toutes les 5 secondes
    list_clignement = []
    eyes_state = "open"
    list_ebr = []
    for i in range(1,len(list_points),8*3+2) :

# Extraction des points ==================================================================================================================
        right_eye_coord = list_points[i:i+8]
        left_eye_coord = list_points[i+8:i+16]
        mouth_coord = list_points[i+16:i+24]
        head_coord = list_points[i+24:i+26]
# ========================================================================================================================================

# Calcul des signes ======================================================================================================================
        ear_right = EAR.eye_aspect_ratio(right_eye_coord)
        ear_left =  EAR.eye_aspect_ratio(left_eye_coord)
        mar = MAR.mouth_aspect_ratio(mouth_coord)
        hop_hb , hop_gd = hop.hop(head_coord)
        ear_mean = (ear_right + ear_left)/2

        # Seuil adaptatif ===========================================================================================

        # Sur la première seconde
        if compt_frame <25 :
            ear_list.append(ear_mean)
            ferme = ear_mean < 0.2

        # Sur le reste du temps
        else :
            # chute de 10 %
            ferme = ear_mean < 0.9*np.mean(np.array(ear_list))
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
        Perclos = PERCLOS.perclos(compteur_ferme,compt_frame+1)

        # Calcul de l'EBR
        ebr = np.sum(list_clignement)
        print(list_clignement)
        list_ebr.append(ebr)

#==========================================================================================================================================
        # On incrémente le compteur de frame
        compt_frame += 1

        # On ajoute nos résultats à la liste des résultats
        res.append(ear_left)
        res.append(ear_right)
        res.append(ear_mean)
        res.append(mar)
        res.append(ebr)
        res.append(Perclos)
        res.append(hop_gd)
        res.append(hop_hb)

    return res, list_ebr

def calculs_signes_live(list_points,compt_frame, ear_list, list_ferme,list_clignement,eyes_state) :
    # les 8*3+1 premieres valeurs sont pour les coordonnées de l'oeil droit
    # toutes les 3 valeurs correspondent à un point

    # Une liste pour stocker les resultats
    res = []

# Extraction des points ==================================================================================================================
    right_eye_coord = list_points[:8]
    left_eye_coord = list_points[8:16]
    mouth_coord = list_points[16:24]
    head_coord = list_points[24:26]
# ========================================================================================================================================

# Calcul des signes ======================================================================================================================
    ear_right = EAR.eye_aspect_ratio(right_eye_coord)
    ear_left =  EAR.eye_aspect_ratio(left_eye_coord)
    mar = MAR.mouth_aspect_ratio(mouth_coord)
    hop_hb , hop_gd = hop.hop(head_coord)
    ear_mean = (ear_right + ear_left)/2

    # Seuil adaptatif ===========================================================================================

    # Sur la première seconde
    if compt_frame <25 :
        ear_list.append(ear_mean)
        ferme = ear_mean < 0.2
        if ferme == True :
            list_ferme.append(1)
            list_clignement.append(1)
            print(f"{compt_frame} j'ajoute {ferme} à list_ferme qui est de taille {len(list_ferme)}")
            print(list_ferme)
        else :
            list_ferme.append(0)
            list_clignement.append(0)
            print(f"{compt_frame} j'ajoute {ferme} à list_ferme qui est de taille {len(list_ferme)}")
            print(list_ferme)


    # Sur le reste du temps
    else :
        # chute de 10 %
        ferme = ear_mean < 0.9*np.mean(np.array(ear_list))
        # On enlève le plus ancien élément
        ear_list.pop(0)
        # On rajoute le nouveau
        ear_list.append(ear_mean)
        # À chaque frame on enlève le premier élement de la liste de la premiere seconde à la fin
        list_clignement.pop(0)

    # ============================================================================================================

    # Détection des clignements ================================================================================

        # Si sur l'image actuel, les yeux sont considérés comme fermés alors on augmente le compteur
        if ferme:
            list_ferme.append(1)
            # Si l'œil était ouvert au début, alors on compte un clignement
            if eyes_state == "open":
                eyes_state = "closed"
                # On ajoute un élément à la liste de clignements
                list_clignement.append(1)
            else:
                # Si l'œil était déjà fermé, on ne compte pas un nouveau clignement
                list_clignement.append(0)
        # Si l'œil est considéré comme ouvert
        else:
            # Réinitialiser l'état des yeux
            eyes_state = "open"
            # On ajoute un élément à la liste de clignements
            list_clignement.append(0)
            list_ferme.append(0)

        # ==========================================================================================================

    # Calcul du perclos
    compteur_ferme = np.sum(list_ferme[-875:])
    print("Taille ",len(list_ferme))
    Perclos = PERCLOS.perclos(compteur_ferme,len(list_ferme))


    # Calcul de l'EBR
    # Un clignement dure 3 frames donc on récupère le quotient par 3
    print(list_clignement)
    ebr = np.sum(list_clignement) 


#==========================================================================================================================================

    # On ajoute nos résultats à la liste des résultats
    res.append(ear_left)
    res.append(ear_right)
    res.append(ear_mean)
    res.append(mar)
    res.append(ebr)
    res.append(Perclos)
    res.append(hop_gd)
    res.append(hop_hb)

    return res, ear_list, list_ferme,list_clignement,eyes_state