from distance_euclidienne import distance

def eye_aspect_ratio(eye):
    ''' Calculate the ratio of the eye length to eye width.
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :param eye: List containing positions which correspond to the eye
    :return: Eye aspect ratio value
    '''
    print("On est rentré dans fonction-EAR\n")
    N1 = distance(eye[2], eye[3])
    N2 = distance(eye[4], eye[5])
    N3 = distance(eye[6], eye[7])
    D = distance(eye[0], eye[1])
    if D == 0 :
        return 0
    return (N1 + N2 + N3) / (3 * D)

