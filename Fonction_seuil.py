import numpy as np
import matplotlib as plt

# On teste la valeur courante par rapport a une liste mise a jour toutes les t secondes.
def F_seuil(val_EAR,list_EAR):
    ferme = False
    max = np.max(list_EAR)
    min = np.min(list_EAR)
    f_seuil = max - min/2  
    if val_EAR < f_seuil :
        ferme = True
    else :
        ferme = False
    return ferme