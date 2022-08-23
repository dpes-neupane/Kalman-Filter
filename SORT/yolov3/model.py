from codecs import backslashreplace_errors
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.pyplot import box
import numpy as np




    
    
    
    
class ObjectDetection(object):
    
    def __init__(self) -> None: #initialize the model with trained weights
        self.outputs = None
        self.boxes = []
        self.confs = []
        self.class_ids = []
        self.dnn = cv.dnn.readNet("SORT\yolov3\yolov3.weights", "SORT\yolov3\yolov3.cfg")    
        with open("SORT\yolov3\coco.names") as fp:
            self.classes = [line.strip() for line in fp.readlines()]
        layers_names = self.dnn.getLayerNames()
        self.output_layers = [layers_names[int(i)-1] for i in self.dnn.getUnconnectedOutLayers()]
        
        self.height = None
        self.width = None
        self.blob  = None
        
    
    def load_image(self, image):
        
        image2 = cv.resize(image, None, fx=0.4, fy=0.4)
        self.height, self.width, _ = image.shape
        self.blob = cv.dnn.blobFromImage(image=image2, scalefactor=0.00392,size=(320, 320), mean=(0,0,0), swapRB=True, crop=False)
        
    

    def predict(self):
        self.dnn.setInput(self.blob)
        self.outputs = self.dnn.forward(self.output_layers)
        
        

    def get_detection_values(self):
        
        for output in self.outputs:
            # print(output.shape)
            for detect in output:
                
                scores = detect[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]
                center_x = int(detect[0]* self.width)
                center_y = int(detect[1]* self.height)
                w = int(detect[2]*self.width)
                h = int(detect[3] * self.height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                self.boxes.append([x, y, w, h])
                self.confs.append(float(conf))
                self.class_ids.append(class_id)
        
            
            
    def bounding_boxes(self, image) -> list:
        
        bb_lbl =[]
        # print(image.shape)
        indices = cv.dnn.NMSBoxes(self.boxes, self.confs, 0.5, 0.3) #Non-maximum supression
        # font = cv.FONT_HERSHEY_COMPLEX
        for i in range(len(self.boxes)):
            label = str(self.classes[self.class_ids[i]])
            if i in indices and label == "person":
                x,y,w,h = self.boxes[i]
                label = str(self.classes[self.class_ids[i]])
                final_y = int((y / image.shape[0]) * image.shape[0])
                final_h = int((((y+h) / image.shape[0]) * image.shape[0] ) - final_y)
                final_x = int((x / image.shape[1]) * image.shape[1])
                final_w = int((((x+w) / image.shape[1]) * image.shape[1] ) - final_x)
                bb_lbl.append( [[final_x, final_y, final_w, final_h], label])
                # cv.rectangle(image, (final_x,final_y), (final_x+final_w, final_y+final_h), 255, 2)
                # cv.putText(image, label, (final_x, final_y-5), font, 1, 255, 2)
        # width = int(image.shape[1] * 50 /100) # resizes by 50%
        # height = int(image.shape[0] * 60 / 100)
        # image = cv.resize(image, (width, height), interpolation=cv.INTER_AREA)
        # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # cv.imshow("img", image)
        return bb_lbl

    def reset(self):
        self.boxes = []
        self.confs = []
        self.class_ids = []    
    
    
if __name__ == "__main__":
    obj = ObjectDetection()
    
    video = cv.VideoCapture("SORT\\video7.mp4")
    while True:
        suc, image = video.read()
        
        obj.load_image(image)
        obj.predict()
        obj.get_detection_values()

        obj.bounding_boxes(image)
        obj.reset()
        # cv.imshow('frame', image)
        if cv.waitKey(10) & 0xff ==ord("q"):
            break
    video.release()
    cv.destroyAllWindows()

    
    
