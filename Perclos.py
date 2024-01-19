# La fonction prend donc en parmètre le nombre de frames totale et le nombre de frames passées les yeux fermés
# Elle retourne donc un pourcentage PERCLOS
def perclos(nb_frame_tot,nb_frame_ferme) :
    return (nb_frame_ferme/nb_frame_tot) * 100