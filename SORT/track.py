import re
import numpy as np
import cv2 as cv

from kalman_filter.Kalman_filter2 import KalmanFilter3
from yolov3.model import ObjectDetection


def get_xtytxbyb(bb):
    xt = bb[0]
    yt = bb[1]
    xb = xt + bb[2]
    yb = yt + bb[3]
    return [xt, yt, xb, yb]


def get_xcyc(bb):
    xc = bb[0] + bb[2]/2
    yc = bb[1] + bb[3]/2
    return (xc, yc)


def xywh_xcyc(xc, yc, w, h):
    x = int(xc - w/2)
    y = int(yc - h/2)
    return x, y


class Track(object):
    def __init__(self, id, bb_lb) -> None:
        self.id = id
        self.state = 0  # not visible
        self.kalman = KalmanFilter3(0.1, 0.01, 0.01)
        self.bb = bb_lb[0]
        self.lb = bb_lb[1]
        self.not_detected_count = 0

    def initialize(self):
        xc, yc = get_xcyc(self.bb)
        self.kalman.x_k = np.array([[xc], [yc], [10.1], [.1]])

    def predict(self):
        xc, yc = self.kalman.predict()[:2].squeeze()
        w = self.bb[2]
        h = self.bb[3]
        x, y = xywh_xcyc(xc, yc, w, h)
        self.bb = [x, y, w, h]

        return xc, yc

    def update(self, z_k):
        z_k = self.kalman.update(z_k)
        # print(z_k)

    def visibile(self):
        self.state = 1

    def invisible(self):
        self.state = 0

    def check_visibility(self):
        return self.state

    def not_detected_counter(self):
        self.not_detected_count += 1

    def return_not_detected_count(self):
        return self.not_detected_count

    def reset_not_detected_count(self):
        self.not_detected_count = 0

    def draw_bb(self, image):
        if self.state == 1:
            x, y, w, h = self.bb
            font = cv.FONT_HERSHEY_COMPLEX
            cv.rectangle(image, (x, y), (x+w, y+h), 255, 2)
            cv.putText(image, self.lb + " " + str(self.id+1),
                       (x, y-5), font, .25, 255, 1)

        return image


class Tracker(object):
    def __init__(self) -> None:
        self.detector = ObjectDetection()

        self.tracks = []
        self.detections = None

    def get_detections(self, image):
        self.detector.load_image(image)
        self.detector.predict()
        self.detector.get_detection_values()
        self.detections = self.detector.bounding_boxes(image)
        self.detector.reset()
        # print(len(self.detections))

    def assign_id(self):
        if self.tracks == []:
            for id, obj in enumerate(self.detections):
                if obj is not None:
                    track = Track(id, obj)
                    track.initialize()
                    # track.changeState()
                    self.tracks.append(track)
        else:
            
            def calc_dist(x1, y1, x2, y2):
                return (x2-x1)**2 + (y2 - y1)**2

            def calc_iou(bb1, bb2):
                bbtb1 = get_xtytxbyb(bb1)
                bbtb2 = get_xtytxbyb(bb2)
                (x1, y1) = (max(bbtb1[0], bbtb2[0]), max(bbtb1[1], bbtb2[1]))
                (x2, y2) = (min(bbtb1[2], bbtb2[2]), min(bbtb1[3], bbtb2[3]))
                if (x2-x1) > 0 and (y2 - y1) > 0:
                    overlap = (x2 - x1) * (y2-y1)
                    # print("overlap", overlap)
                    union = (bb1[2] * bb1[3]) + (bb2[2] * bb2[3]) - overlap
                    iou = overlap / union
                    return iou
                else:
                    return 0
            print(len(self.detections))
            for obj in self.detections:
                xc, yc = get_xcyc(obj[0])
                count = 0
                for track in self.tracks:
                    # print(track.kalman.x_k, i)
                    x2, y2 = track.predict()
                    dist = calc_dist(xc, yc, x2, y2)
                    iou = calc_iou(obj[0], track.bb)
                    # print(i , dist, iou)
                    if dist < 10000 and iou >= 0.3:
                        track.update([[xc], [yc]])
                        track.visibile()

                    else:
                        count += 1
                        track.not_detected_counter()

                if count == len(self.tracks):
                    # print(obj)
                    track = Track(len(self.detections), obj)
                    track.initialize()
                    self.tracks.append(track)
                
            for track in self.tracks:
                # print("visible",track.state)
                if track.return_not_detected_count() == len(self.detections) or len(self.detections) == 0:
                    # track.invisible()
                    track.predict()
                track.reset_not_detected_count()

    def draw_bb(self, image):

        for track in self.tracks:
            image = track.draw_bb(image)

        return image

    def draw_detection(self, image):
        for obj in self.detections:
            x, y, w, h = obj[0][0], obj[0][1], obj[0][2], obj[0][3]

            font = cv.FONT_HERSHEY_COMPLEX
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
            # cv.putText(image, obj[1], (x, y-5), font, 0.25, 255, 1)
            
        # width = int(image.shape[1] * 50 /100) # resizes by 50%
        # height = int(image.shape[0] * 60 / 100)
        # image = cv.resize(image, (width, height), interpolation=cv.INTER_AREA)

        return image


if __name__ == "__main__":
    obj = Tracker()
    # fourcc = cv.VideoWriter_fourcc(*'DIVX')
    video = cv.VideoCapture("SORT\\video4.mp4")
    _, frame1 = video.read()
    # print(frame1.shape)
    # out = cv.VideoWriter('output.avi', apiPreference=0, fourcc=fourcc,
                        #  fps=20, frameSize=(frame1.shape[1], frame1.shape[0]))
    while True:
        suc, frame = video.read()
        if not suc:
            break
        obj.get_detections(frame)
        obj.assign_id()
        frame = obj.draw_bb(frame)
        frame = obj.draw_detection(frame)
        # print(frame.shape)
        
        frame1 = cv.resize(frame, (frame1.shape[1], frame1.shape[0]))
        # out.write(frame1)

        cv.imshow("frame", frame)
        if cv.waitKey(10) & 0xff == ord('q'):
            break
    video.release()
    # out.release()
    cv.destroyAllWindows()
