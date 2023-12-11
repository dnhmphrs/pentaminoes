import numpy as np
from cv2 import aruco
import cv2


PENTOMINOS_SHAPE = [
    [[0,1,1],
     [1,'x',0],
     [0,1,0]],

    [[2],[2],['x'],[2],[2]],

    [[3, 0],
     [3, 0],
     ['x', 0],
     [3, 3]],

    [[0, 4],
    ['x', 4],
    [4, 0],
    [4, 0]],

    [[5, 5],
    ['x', 5],
    [5, 0]],

    [[6,'x',6],
     [0,6,0],
     [0,6,0]],

    [[7,0,7],
     [7,'x',7]],

    [[8,0,0],
     [8,0,0],
     ['x',8,8]],

    [[9,0,0],
     [9,'x',0],
     [0,9,9]],

    [[ 0,10, 0],
     [10,'x',10],
     [ 0,10, 0]],

    [[0, 11],
    [11, 'x'],
    [0, 11],
    [0, 11]],

    [[12,12, 0],
     [ 0,'x', 0],
     [ 0,12,12]]
]

pentomino_names = "FILNPTUVWXYZ"

class pentominos:

    def __init__(self):

        self.PENTOMINOS_SHAPE = PENTOMINOS_SHAPE

        # dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
        # parameters =  aruco.DetectorParameters()
        # self.detector = aruco.ArucoDetector(dictionary, parameters)

        self.arucoDict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
        self.arucoParams = aruco.DetectorParameters_create()


    def check_pentominos(self, img: np.array):

        id_list, coords, orientation = self.get_positions(img)



    def get_positions(self, img: np.array) -> list:

        corners, ids, rejectedImgPoints = aruco.detectMarkers(img, self.arucoDict, parameters=self.arucoParams)

        # corners, ids, rejectedImgPoints = detector.detectMarkers(img)

        if ids is not None:
            ids = ids.flatten()
            center_coords = list()
            orientation = list()
            id_list = list()
            for (markerCorner, markerID) in zip(corners, ids):
                if markerID < 50:
                    M = cv2.moments(markerCorner)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    pX = int(img.shape[1]/cX)
                    pY = int(img.shape[0]/cY)

                    centerCoordinates = (pX, pY)
                    center_coords.append(centerCoordinates)

                    corn = markerCorner.reshape((4, 2)).astype(int)
                    corn = corn[0].tolist()

                    if (int(corn[0]) < cX) and (int(corn[1]) < cY):
                        temp_orientation = 0
                    if (int(corn[0]) < cX) and (int(corn[1]) > cY):
                        temp_orientation = 1
                    if (int(corn[0]) > cX) and (int(corn[1]) > cY):
                        temp_orientation = 2
                    if (int(corn[0]) > cX) and (int(corn[1]) < cY):
                        temp_orientation = 3

                    if markerID > 20:
                        markerID -= 20
                        temp_orientation += 4

                    id_list.append(markerID)
                    orientation.append(temp_orientation)
            return [id_list, center_coords, orientation]
        else:
            return None
