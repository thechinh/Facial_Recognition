import cv2
import argparse
import os
import tensorflow as tf
from numpy import load
from statistics import mode, mean
import threading
from utils import *
from facenet_architecture import *

################################################################################
#######################        GLOBAL SETTINGS        ##########################
################################################################################

WEIGHT = 'yolo/yolov3-wider_16000.weights'
MODEL = 'yolo/yolov3-face.cfg'

net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
IMG_SIZE = 416
CROP_SIZE = 160 # required

# BGR
BLUE = (255,0,0)
RED  = (0,0,255)

# Facenet
model_facenet = InceptionResNetV1()

# KNN
K_UNKWOWN_THRESHOLD = 9
K_NB = 5
MODEL_NAME = 'holy_KNN.h5'

# Init
thread_finished = True
total_save = 0
boxes_buffer = []
text_buffer = [] 

################################################################################
##############################     THREAD 2      ###############################
################################################################################

def detect_and_predict(net, frame, IMG_SIZE, predict_mode, model_facenet):
    global thread_finished, boxes_buffer, text_buffer
    thread_finished = False
    
    boxes_buffer = []
    text_buffer = []

    boxes_buffer, confidences = face_detection(net, frame, IMG_SIZE)
    
    for box in boxes_buffer:
        # Extract position data
        topleft_x, topleft_y, width, height = box[0], box[1], box[2], box[3]
        
        # Extract face area
        face = frame[topleft_y:topleft_y+height, topleft_x:topleft_x+width]

        if predict_mode and (face.size > 0):
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            resized_face = cv2.resize(face, (CROP_SIZE, CROP_SIZE))
            # Extract features
            face_emb = get_embedding(model_facenet, resized_face)
            face_emb_array = np.asarray(face_emb)
            face_emb_array = face_emb_array.reshape(1,-1)
            # Predict
            text = KNN_predict(knn_model, face_emb_array, trainy, out_encoder, K_UNKWOWN_THRESHOLD)
        else:
            text = f'{confidences[i]:.2f}'

        # BUFFER
        text_buffer.append(text)

    thread_finished = True
    return thread_finished


################################################################################
#############################     PREPROCESS      ##############################
################################################################################

# Read arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str, default=None, help="Save face photo to folder")
args = vars(ap.parse_args())

if args["output"] != None:
    print("Mode: Collect photo")
    predict_mode = False

    # Create corresponding folder if not exist
    if not os.path.exists(os.path.join('data', args["output"])):
        os.makedirs(os.path.join('data', args["output"]))
else: 
    print("Mode: Predict")
    predict_mode = True
    
    # Load photos from folder 'data' and feed to KNN
    trainX, trainy = load_dataset('data')
    model_facenet.load_weights('facenet_keras_weights.h5')
    newTrainX, trainy, out_encoder = convert_dataset(model_facenet, trainX, trainy)
    knn_model = KNN_fit(newTrainX, MODEL_NAME, K_NB)

    print(trainX.shape, trainy.shape)
    print(newTrainX.shape)

################################################################################
#############################        MAIN        ###############################
################################################################################

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if thread_finished == True:
        # Save result
        boxes_cache = boxes_buffer.copy()
        text_cache  = text_buffer.copy()
        x = threading.Thread(target=detect_and_predict, args=(net, frame, IMG_SIZE, predict_mode, model_facenet,)).start()  

    # Draw
    for i, box in enumerate(boxes_cache):
        x, y, w, h = box[:4]
        margin_x, margin_y = 10, 20
        topleft       = (x-margin_x, y-margin_y)
        bottomright   = (x+w+margin_x, y+h+margin_y)
        cv2.rectangle(frame, topleft, bottomright, BLUE, 2)
        cv2.putText(frame, text_cache[i], topleft, cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
    
    cv2.imshow('face detection', frame)

    # Wait for key press
    key = cv2.waitKey(25) & 0xFF
    # Capture when only 1 face detected
    if  predict_mode==False and len(boxes_cache)==1 and key==ord("k"):
        total_save += 1
        print(f'Save a photo of {args["output"]}. ID {total_save}')
        p = os.path.join('data', args["output"], "{}.png".format(str(total_save).zfill(3)))
        x, y, w, h = boxes_cache[:4]
        cv2.imwrite(p, frame[y:y+h, x:x+w])
    elif key == ord("q"):
        break

print("Camera closed!")
cap.release()
cv2.destroyAllWindows()