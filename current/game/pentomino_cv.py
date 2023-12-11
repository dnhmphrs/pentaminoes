import cv2
from cv2 import aruco
import numpy as np

class PentominoDetector:

    def __init__(self):

        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
        parameters =  aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(dictionary, parameters)
        # Create a sharpening kernel
        self.kernel = np.array([[-1, -1, -1], [-1,  9, -1], [-1, -1, -1]])

    def find_aruco_markers(self, img):

        # Apply the kernel to the image
        sharpened = cv2.filter2D(img, -1, self.kernel)

        corners, ids, rejectedImgPoints = self.detector.detectMarkers(sharpened)

        if ids is not None:

            # aruco.drawDetectedMarkers(img, corners)
            ids = ids.flatten()
            center_coords = list()
            orientation = list()
            id_list = list()
            # print(f'Aruco-IDs: {ids}')

            for (markerCorner, markerID) in zip(corners, ids):
                if markerID < 50:
                    markerID = markerID - 19

                    M = cv2.moments(markerCorner)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    pX = int(img.shape[1]/cX)
                    pY = int(img.shape[0]/cY)

                    centerCoordinates = (cX, cY)
                    center_coords.append(centerCoordinates)

                    corn = markerCorner.reshape((4, 2)).astype(int)
                    corn = corn[0].tolist()

                    temp_orientation = 0

                    if markerID > 12:
                        markerID = markerID - 12
                        temp_orientation += 4

                    if (int(corn[0]) <= cX) and (int(corn[1]) <= cY):
                        temp_orientation += 0
                    if (int(corn[0]) <= cX) and (int(corn[1]) >= cY):
                        temp_orientation += 1
                    if (int(corn[0]) >= cX) and (int(corn[1]) >= cY):
                        temp_orientation += 2
                    if (int(corn[0]) >= cX) and (int(corn[1]) <= cY):
                        temp_orientation += 3

                    id_list.append(markerID)
                    orientation.append(temp_orientation)
            return id_list, center_coords, orientation
        else:
                return None

    def find_corners(self, img):

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = self.detector.detectMarkers(img)
        # arucoDict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        # arucoParams = aruco.DetectorParameters_create()
        # corners, ids, rejectedImgPoints = aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
        if ids is not None:
            ids = ids.flatten()
            if all(x in ids for x in [50, 51, 52, 53]):
                coords = list()
                id_list = list()
                for (markerCorner, markerID) in zip(corners, ids):

                    # M = cv2.moments(markerCorner)
                    # cX = int(M["m10"] / M["m00"])
                    # cY = int(M["m01"] / M["m00"])
                    # center_coordinates = (cX, cY)
                    # coords.append(center_coordinates)
                    coord = None
                    corn = markerCorner.reshape((4, 2)).astype(int)

                    if markerID == 50:
                        coord = corn[2].tolist()
                    elif markerID == 51:
                        coord = corn[1].tolist()
                    elif markerID == 52:
                        coord = corn[0].tolist()
                    elif markerID == 53:
                        coord = corn[3].tolist()

                    if coord != None:
                        coords.append(coord)
                        id_list.append(markerID)

                coords = [x for _, x in sorted(zip(id_list, coords))]
                return coords
            else:
                return None
        else:
            return None
        # aruco.drawDetectedMarkers(img, corners)


    def transform_image(self, img: np.array):

        corners = self.find_corners(img)

        if corners is None:
            return None

        pt_A = corners[0]
        pt_B = corners[1]
        pt_C = corners[2]
        pt_D = corners[3]

        width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
        width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
        maxWidth = max(int(width_AD), int(width_BC))


        height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
        height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
        maxHeight = max(int(height_AB), int(height_CD))

        input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
        output_pts = np.float32([[0, 0],
                                [0, maxHeight - 1],
                                [maxWidth - 1, maxHeight - 1],
                                [maxWidth - 1, 0]])

        M = cv2.getPerspectiveTransform(input_pts,output_pts)

        return cv2.warpPerspective(img,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)


    def find_pentominos(self, img: np.array):

        pentominos = self.find_aruco_markers(img)
        if pentominos is None:
            return None

        return pentominos


if __name__=='__main__':

    detector = PentominoDetector()

    WIDTH = 1920
    HEIGHT = 1080
    # ip = 'http://172.24.191.212:4747/'
    # ip = 'http://192.168.2.30:4747/'
    #cam_source = f'{ip}video?{WIDTH}x{HEIGHT}'

    cap = cv2.VideoCapture(4)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    while True:

        ret, frame = cap.read()
        if not ret:
            continue

        image = frame.copy()

        transformed = detector.transform_image(image)

        pentimentos = detector.find_aruco_markers(transformed)
        if pentimentos is not None:
            print(f"Laenge: {len(pentimentos[0])}, {pentimentos[0]}")
        cv2.imshow('Frame', transformed)

        k = cv2.waitKey(1)
        if k== ord('q'):
            break

    cv2.destroyAllWindows()
