import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d  # Fonction pour la 3D

def calculer_equation_plan(point1, point2, point3):
    # Créer deux vecteurs à partir des trois points
    vecteur1 = np.subtract(point2, point1)
    vecteur2 = np.subtract(point3, point1)

    # Calculer le produit vectoriel des deux vecteurs
    normal = np.cross(vecteur1, vecteur2)

    # Calculer le terme constant de l'équation du plan
    constante = -np.dot(normal, point1)

    # Retourner les coefficients de l'équation du plan
    return (*normal, constante)

def calcul_angle(x,y):
    return (np.linalg.norm(x)/np.linalg.norm(y))

def calculate_angle(x,y) :
    return(-np.abs(np.arctan2(y,x)*180/np.pi)+180)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For static images:
IMAGE_FILES = []
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            continue
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            print('face_landmarks:', face_landmarks)
            mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
        cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            # for face_landmarks in results.multi_face_landmarks:
            #     hop_face_landmarks = []
            #     keypoint_hop = [27, 257, 17] # 27 = oeil gauche, 257 = oeil droit, 17 = bouche
            # for i in keypoint_hop:
            #     hop_face_landmarks.append([face_landmarks.landmark[i].x, face_landmarks.landmark[i].y, face_landmarks.landmark[i].z])
                
            # MARCHE MOINS BIEN
            for face_landmarks in results.multi_face_landmarks:
                hop2 = []
                keypoint_hop2 = [10, 152]
            for i in keypoint_hop2:
                hop2.append([face_landmarks.landmark[i].x, face_landmarks.landmark[i].y, face_landmarks.landmark[i].z])
            x = hop2[0][0] - hop2[1][0]
            y = hop2[0][1] - hop2[1][1]
            z = hop2[0][2] - hop2[1][2]

            
            
            # x,y,z,d = calculer_equation_plan(hop_face_landmarks[0], hop_face_landmarks[1], hop_face_landmarks[2])
            # print(x,y,z)
            # print('hop_face_landmarks:', hop_face_landmarks)
            mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
        # Flip the image horizontally for a selfie-view display.
            angle_xy = calculate_angle(y,x)
            angle_yz = calculate_angle(z,y)
            
            cv2.putText(image, f'HOP: {angle_xy:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f'HOP: {angle_yz:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.putText(image, f'HOP: {calcul_angle(x,y):.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('MediaPipe Face Mesh', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()