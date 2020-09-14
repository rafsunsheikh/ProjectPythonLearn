import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("../yolo_object_detection/yolov3.weights","../yolo_object_detection/yolov3.cfg")
classes = []
with open("../yolo_object_detection/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

print(classes)
layer_names = net.getLayerNames()
outputLayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#Loading Images
img = cv2.imread("../Resources/goru1.jpg")
img  = cv2.resize(img,None, fx = 0.4,fy = 0.4)
height, width, channels = img.shape

#Detecting Objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0, 0, 0), True, crop=False)

# for b in blob:
#     for n, img_blob in enumerate(b):
#         cv2.imshow(str(n),img_blob)

net.setInput(blob)
outs = net.forward(outputLayers)
#print(outs)

# Show informationon the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            #Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            #cv2.circle(img,(center_x,ceter_y), 10, (0,255,0), 2)
            #Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y -h / 2)

            boxes.append([x,y,w,h])
            confidences.append([float(confidence)])
            class_ids.append(class_id)
# print(len(boxes))
#number_objects_detected = len(boxes)
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indexes)

for i in range(len(boxes)):
    for i in indexes:
        x,y,w,h = boxes[i]
        label = str(classes[class_ids[i]])
        #print(label)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img,label,(x, y+30),cv2.FONT_HERSHEY_SIMPLEX ,.5,(0,0,0),3)

cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()