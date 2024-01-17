import cv2
import numpy as np

print("Loading YOLO")

net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

classes = []

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
layer_names = net.getLayerNames()

output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
print("YOLO LOADED")

print("\nCLASSES: \{}".format(classes))

#Object detection in real time
video_capture = cv2.VideoCapture(0)
while True:
    #Capture frame-by-frame
    re,img = video_capture.read()
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    
    #Using blob function of opencv to preprocess image
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), 
                                swapRB = True, crop=False)
    #Detecting objects 
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    #Showing informations on the screen
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
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                
                #Rectangle coordinates
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    #We use NPS functions in opencv to perform non-maximum supression
    #we give it score threshold and nms threshold as argument..

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 2, color, 3)
            
    cv2.imshow("Image", cv2.resize(img, (800, 600)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
video_capture.release()
cv2.destroyAllWindows()