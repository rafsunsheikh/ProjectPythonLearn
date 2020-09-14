import cv2
import numpy as np

# Load Yolo
net = cv2.dnn.readNet("../yolo_object_detection/yolov3.weights", "../yolo_object_detection/yolov3.cfg")
classes = []
with open("../yolo_object_detection/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture("../Resources/goproVideo.mkv")
while True:
    ret, img = cap.read()
    #img = cv2.resize(img, None, fx=0.4, fy=0.4)
    #img = cv2.resize(img, (frameWidth, frameHeight))
    height, width, channels = img.shape
    print(img.shape)
    #cv2.imshow("Result", img)
    #img = cv2.imread("room_ser.jpg")
    # if cv2.waitKey(1) and 0xFF == ord('q'):
    #      break

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indexes)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 3, color, 3)


cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()