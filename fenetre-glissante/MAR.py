from distance_euclidienne import distance

def mouth_aspect_ratio(mouth):
    ''' Calculate mouth feature as the ratio of the mouth length to mouth width
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :return: Mouth feature value
    '''
    #print("On est entré dans MAR\n")
    N1 = distance(mouth[2],mouth[3])
    N2 = distance(mouth[4],mouth[5])
    N3 = distance(mouth[6],mouth[7])
    D = distance(mouth[0],mouth[1])
    if D==0 :
        return 0
    return  (N1 + N2 + N3) / (3 * D)