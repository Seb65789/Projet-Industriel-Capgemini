import numpy as np

def ferme(seuil,EAR) :
    """Fonction qui prends la valeur de l'EAR actuelle et la valeur du seuil sur les 10 premières secondes.
    Renvoie True si l'oeil est fermé, False sinon."""
    if(EAR < seuil) :
        return True
    return False

def seuil(list_EAR) :
    """Fonction qui prends une liste de valeurs EAR sur 10 
    secondes et qui renvoie la valeur seuil de fermeture des yeux.
    """
    # On prends la moyenne entre le max et le min
    max_list = np.max(list_EAR)
    min_list = np.min(list_EAR)
    return (max_list+min_list)/2

    