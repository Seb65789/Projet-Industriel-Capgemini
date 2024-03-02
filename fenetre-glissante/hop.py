import numpy as np

def calculate_angle(x, y):
    return (-np.abs(np.arctan2(y, x) * 180/np.pi) + 180)

def hop(head):
    print("On est entré dans HOP\n")
    x = head[0]
    y = head[1]
    z = head[2]
    angle_hb = calculate_angle(y, x)
    angle_gd = calculate_angle(z, y)
    return angle_hb, angle_gd ;