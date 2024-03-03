import numpy as np

def distance(p1, p2):
    ''' Calculate distance between two points
    :param p1: First Point
    :param p2: Second Point
    :return: Euclidean distance between the points. (Using only the x and y coordinates).
    '''
    #print("On est en train de calculé la distance")
    p1 = np.array(p1)
    p2 = np.array(p2)
    return (((p1 - p2)**2).sum())**0.5