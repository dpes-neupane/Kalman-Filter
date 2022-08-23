## A simple Object tracker using YOLOv3 and Kalman Filter

In this project, a simple object tracking algorithm was written with the ability to assign person IDs each. 

It uses Kalman filter for predicting the future position of each object that are tracked and tries to assign correctly to the same object in the video using IoU and distance threshold. 
 
YOLOV3 is used for object detection. The weight file, .cfg file and the coco.names files have to separately downloaded from the internet and placed in the yolov3 folder. You can read about it in https://pjreddie.com/darknet/yolo/ 



![twopeople](https://user-images.githubusercontent.com/57759185/186122103-d60858c5-92c0-404d-b471-20c9de2cec36.gif)



https://user-images.githubusercontent.com/57759185/186120421-f9139ab4-83d0-434e-948b-3e4a912cb52c.mp4

