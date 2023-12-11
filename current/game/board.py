import numpy as np
import cv2
import pentominos as pentominos
import pentomino_cv as pcv
import pandas as pd
from threading import Thread, Lock
import os
# import pygame

BOARD_GRID = 20
BOARD_SIZE = 500

class VideoStream:

    def __init__(self, src=0, width=1920, height=1080):

        # ip = 'http://172.24.191.212:4747/'
        # cam_source = f'{ip}video?{width}x{height}'
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.started = False
        # self.read_lock = Lock()
        #to get the current working directory
        # directory = os.getcwd()

        # print(directory)
        df_raw = pd.read_csv('solutions-126.csv', header=None)
        self.df = df_raw.iloc[:,1:]
        self.board = BOARD()
        self.detector = pcv.PentominoDetector()


    def start(self):
        if self.started:
            print('already started!!')
        else:
            self.started = True
            self.thread = Thread(target=self.update, args=())
            self.thread.start()
        return self

    def update(self):
        # Create a named window
        cv2.namedWindow('Pentomino', cv2.WINDOW_NORMAL)

        # Set the window to fullscreen
        cv2.setWindowProperty('Pentomino', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        show_image = self.board.cv2_gui.copy()
        show_image = cv2.resize(show_image, (int(show_image.shape[1]*1.6),int(show_image.shape[0]*1.6)))

        cv2.imshow('Pentomino', show_image)
        cv2.waitKey(1)

        while self.started:

            # self.read_lock.acquire()
            self.ret, self.frame = self.stream.read()
            if not self.ret:
                continue
            # self.read_lock.release()
            # self.frame = cv2.imread("game/20230622_173541.jpg")
            cv2.imshow('camera', self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.started = False
                cv2.destroyAllWindows()
            self.get_status()

    def get_status(self):

            image = self.detector.transform_image(self.frame.copy())

            if image is None:
                # if not pygame.mixer.music.get_busy():
                #         pygame.mixer.init()
                #         pygame.mixer.music.load("corners.mp3")
                #         pygame.mixer.music.play()
                return None
            transformed_image = image.copy()
            pent = self.detector.find_pentominos(image)

            if pent is None:
                return None

            [ids, coords, orientations] = pent

            for id, coord in zip(ids, coords):
                cv2.putText(transformed_image, f'{pentominos.pentomino_names[id-1]}', coord, cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 10)
                cv2.putText(transformed_image, f'{pentominos.pentomino_names[id-1]}', coord, cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 5)

            if len(ids) > 0:

                self.board.reset_board()
                coords_list = list()
                for idd, coord, orientation in zip(ids, coords, orientations):

                    scale_width = 22 / image.shape[1]
                    scale_height = 22 / image.shape[0]

                    x = int(coord[0]*scale_width)
                    y = int(coord[1]*scale_height)

                    flip = False

                    if orientation >= 4:
                        orientation -= 4
                        flip = True

                    self.board.place_pentomino(idd-1, (x,y), orientation, flip)

                    coords_list.append([y, x, idd])

                self.board.show_board()

                temp_board = self.board.board.copy()
                temp_board = np.where(temp_board!=0, 0, 255).astype(np.uint8)

                connected = cv2.connectedComponentsWithStats(temp_board,
                                                    4,
                                                    cv2.CV_32S)

                (totalLabels, label_ids, values, centroid) = connected
                unique, counts = np.unique(label_ids, return_counts=True)


                edge_areas = set()
                edge_areas.add(0)
                for line in label_ids:
                    first = line[0]
                    last = line[-1]
                    if first >= 1:
                        edge_areas.add(first)
                    if last >= 1:
                        edge_areas.add(last)

                for column in np.rot90(label_ids, 1):
                    first = column[0]
                    last = column[-1]
                    if first >= 1:
                        edge_areas.add(first)
                    if last >= 1:
                        edge_areas.add(last)

                sum_area = list()
                count_area = 1

                self.board.cv2_gui[85:550,580:980] = (0, 0, 0)

                for count, each in enumerate(centroid):
                    if count not in edge_areas:
                        area = values[count][-1]
                        sum_area.append(area)
                        # cv2.putText(self.board.cv2_gui, f'{count_area}', (round(each[0]*self.board.SIZE/self.board.GRID+55),round(each[1]*self.board.SIZE/self.board.GRID+70)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                        # cv2.putText(self.board.cv2_gui, f'Area ({count_area}): {area}', (600, 125+count_area*50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                        count_area += 1

                self.board.write_fenced_area(sum_area)
                # cv2.imwrite('test.png', self.board.cv2_gui)

                x_offset = centroid[0][0]
                y_offset = centroid[0][1]

                sorted_list = sorted(coords_list, key=lambda x:np.arctan2(x[0]-y_offset,x[1]-x_offset))

                while sorted_list[0][2] != 1:
                    sorted_list = sorted_list[-1:] + sorted_list[:-1]

                if sorted_list[1][2] < sorted_list[-1][2]:
                    temp_pos = sorted_list[1:]
                    temp_pos.reverse()
                    sorted_list = sorted_list[:1] + temp_pos


                s = np.array([x[2] for x in sorted_list])

                if len(ids) == 12:
                    a = self.df[(self.df == s).all(1)].index.tolist()

                    if len(a) == 0:
                        max_area = 125
                    else:
                        max_area = self.df_raw.iloc[a[0],0]

                    self.board.write_max_area(max_area)

            show_image = self.board.cv2_gui.copy()
            show_image = cv2.resize(show_image, (int(show_image.shape[1]*1.6),int(show_image.shape[0]*1.6)))

            transformed_image = cv2.resize(transformed_image, (500,500))
            show_image[300:800,965:965+500] = transformed_image
            cv2.imshow('Pentomino', show_image)
            if cv2.waitKey(1) == ord("q"):
                self.started = False
                cv2.destroyAllWindows()

    def __exit__(self, exc_type, exc_value, traceback) :
        if self.thread.is_alive():
            self.stream.release()
            self.started = False
            self.thread.join()

class BOARD:

    def __init__(self):

        self.colors = [(150,255,255),
                  (255,150,150),
                  (150,150,255),
                  (200,0,0),
                  (150,255,150),
                  (255,255,150),
                  (255,150,255),
                  (200,200,0),
                  (0,200,0),
                  (0,0,200),
                  (200,0,200),
                  (0,200,200)]

        self.border = 1
        self.cv2_gui = np.full(shape=(600,1000,3), fill_value=0, dtype=np.uint8)


        self.SIZE = BOARD_SIZE
        self.GRID = BOARD_GRID

        self.board = np.zeros((self.GRID,self.GRID), dtype=int)

        self.cv2_board = np.full(shape=(self.SIZE,self.SIZE,3), fill_value=0, dtype=np.uint8)


        self.draw_grid()
        self.show_board()


    def pick_and_place(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:

            if 50 <= x < 550 and 50 <= y < 550:
                x = int((x-50) / (self.SIZE/self.GRID))
                y = int((y-50) / (self.SIZE/self.GRID))
                shape_id = self.board[y,x]
                if shape_id < 0:
                    shape_id *= -1
                if shape_id > 0:
                    self.last_item(shape_id)

    def last_item(self, shape_id: int):
        self.item = np.full(shape=(150,150,3), fill_value=0, dtype=np.uint8)
        size = int(self.SIZE / self.GRID)
        pentomino = np.array(pentominos.PENTOMINOS_SHAPE[shape_id-1])
        pentomino[pentomino == 'x'] = shape_id + 1
        pentomino = pentomino.astype(int)
        y_start = int((self.item.shape[0]-pentomino.shape[0]*size) / 2)
        x_start = int((self.item.shape[1]-pentomino.shape[1]*size) / 2)

        for row in range(pentomino.shape[0]):
            for col in range(pentomino.shape[1]):
                if pentomino[row,col] > 0:

                    x = x_start + col*size
                    y = y_start + row*size

                    self.item[y:y+size:,x:x+size] = 255
                    self.item[y+self.border:y+size-self.border,x+self.border:x+size-self.border] = self.colors[shape_id-1]


    def show_board(self):

        self.cv2_gui[50:550,50:550] = self.cv2_board

        self.fontFace = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 1
        self.fontColor = (255, 255, 255)
        self.fontThickness = 2


    def write_fenced_area(self, area: list):
        cv2.putText(self.cv2_gui, f'Fenced areas: {sum(area)}', (600, 125), self.fontFace, self.fontScale, self.fontColor, thickness=self.fontThickness)

    def write_max_area(self, area: int):
        self.cv2_gui[25:85,580:780] = (0, 0, 0)
        if area < 126:
            area = "up to 125"
        cv2.putText(self.cv2_gui, f'Max area: {area}', (600, 75), self.fontFace, self.fontScale, self.fontColor, thickness=self.fontThickness)

    def place_pentomino(self, shape_id: int, position: tuple, rotate: int=0, flip: bool=False):


        pentomino = np.array(pentominos.PENTOMINOS_SHAPE[shape_id])


        if flip:
            pentomino = np.flip(pentomino, 1)

        for angle in range(rotate):
            pentomino = np.rot90(pentomino)

        offset_y, offset_x = np.where(pentomino == 'x')

        position_lt_x = position[0] - offset_x[0] - 1
        position_lt_y = position[1] - offset_y[0] - 1

        # print(shape_id, offset_y, offset_x)
        if (position_lt_x+pentomino.shape[1]) <= self.GRID and (position_lt_y+pentomino.shape[0]) <= self.GRID:

            board_cutout = self.board[position_lt_y:position_lt_y+pentomino.shape[0],position_lt_x:position_lt_x+pentomino.shape[1]].copy()
            board_cutout = np.where(board_cutout<0,-board_cutout,board_cutout)
            board_cutout = np.where(board_cutout>0,-1,-2)

            pentomino_replaced = pentomino.copy()
            pentomino_replaced[pentomino == 'x'] = shape_id + 1
            pentomino_replaced = pentomino_replaced.astype(int)

            if True in np.where(board_cutout==np.where(pentomino_replaced>0,-1, shape_id+1), True, False):

                print('Rule check! - Blocked')
                return None
            else:
                board_cutout = self.board[position_lt_y:position_lt_y+pentomino.shape[0],position_lt_x:position_lt_x+pentomino.shape[1]] | pentomino_replaced

                board_cutout[pentomino == 'x'] = -(shape_id + 1)

                self.board[position_lt_y:position_lt_y+pentomino.shape[0],position_lt_x:position_lt_x+pentomino.shape[1]] = board_cutout

                for row in range(pentomino.shape[0]):
                    for col in range(pentomino.shape[1]):

                        size = int(self.SIZE / self.GRID)
                        x = (row + position_lt_y) * size
                        y = (col + position_lt_x) * size



                        if abs(self.board[row + position_lt_y,col + position_lt_x]) == shape_id + 1:

                            self.cv2_board[x:x+size:,y:y+size] = 255
                            self.cv2_board[x+self.border:x+size-self.border,y+self.border:y+size-self.border] = self.colors[shape_id]
                        # elif self.board[row + position_lt_y,col + position_lt_x] == -(shape_id + 1):
                        #
                        #     self.cv2_board[x:x+size:,y:y+size] = 255
                        #     self.cv2_board[x+self.border:x+size-self.border,y+self.border:y+size-self.border] = 0
                        #     self.cv2_board[x+int(3/8*size):x+int(5/8*size),y+int(3/8*size):y+int(5/8*size)] = 255
        else:
            print('Rule Check! - Out of Area')
            return None

        # self.last_item(shape_id+1)

    def reset_board(self):
        self.board = np.zeros((self.GRID,self.GRID), dtype=int)
        self.draw_grid()

    def draw_grid(self):
        self.cv2_board[:,:] = (80,80,0)
        for row in range(self.GRID):
            for col in range(self.GRID):
                size = int(self.SIZE / self.GRID)
                x = row * size
                y = col * size
                self.cv2_board[x+self.border:x+size-self.border,y+self.border:y+size-self.border] = 0

if __name__ == "__main__":

    VideoStream().start()
