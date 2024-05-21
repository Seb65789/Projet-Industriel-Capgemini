import numpy as np

def calculate_angle(x, y):
    """
    Calculates the angle between two vectors.

    Parameters:
    x (float): first vector.
    y (float): second vector.

    Returns:
    float: The angle between the two vectors in degrees.
    """
    return (-np.abs(np.arctan2(y, x) * 180/np.pi) + 180)

def hop(head):
    """
    Performs the hop operation.

    Parameters:
    head (list): A list containing two vectors.

    Returns:
    tuple: A tuple containing the angles between the vectors.
    """
    x = np.array(head[0])[0] - np.array(head[1])[0]
    y = np.array(head[0])[1] - np.array(head[1])[1]
    z = np.array(head[0])[2] - np.array(head[1])[2]
    angle_hb = calculate_angle(y, x)
    angle_gd = calculate_angle(z, y)
    return angle_hb, angle_gd

def hophb(c1, c2, c0):
    """
    Calculates a factor based on the ratio between c1-c0 and c2-c0, only taking the x and y coordinates into account.
    For gd : c0 = 0, c1 = 61, c2 = 291
    For hb : c0 = 0, c1 = 10, c2 = 152
    
    Parameters:
    c1 (list): A list containing the x, y and z coordinates of a point.
    c2 (list): A list containing the x, y and z coordinates of a point.
    c0 (list): A list containing the x, y and z coordinates of a point.
    
    Returns:
    float: The factor calculated.
    """
    length_0_1 = np.sqrt((c1[0] - c0[0])**2 + (c1[1] - c0[1])**2)
    length_0_2 = np.sqrt((c2[0] - c0[0])**2 + (c2[1] - c0[1])**2)
    return length_0_1/length_0_2    