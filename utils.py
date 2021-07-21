import os, cv2, pickle, statistics
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

# Load all face images in a directory
def load_faces(directory, required_size=(160, 160)):
    faces = list()
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, required_size)
        face = np.asarray(image)
        # store
        faces.append(face)
	
    return faces

# Load data folder containing images of each person
def load_dataset(directory):
	X, y = list(), list()
	for subdir in os.listdir(directory):
		path = os.path.join(directory, subdir)
		# scan folders only 
		if not os.path.isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces(path)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>>> Loaded %d examples for class: %s' % (len(faces), subdir))
		# save
		X.extend(faces)
		y.extend(labels)
	return np.asarray(X), np.asarray(y)

# Get embedding (feature vector) of 1 face
def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # make prediction to get embedding
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]

# Convert each face in the dataset to an embedding
def convert_dataset(model, trainX, trainy):
    newTrainX = list()
    for face_pixels in trainX:
        embedding = get_embedding(model, face_pixels)
        newTrainX.append(embedding)
    newTrainX = np.asarray(newTrainX)

    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    return newTrainX, trainy, out_encoder

def KNN_predict(knn_model, face_emb_array, trainy, out_encoder, K_UNKOWN_THRESHOLD):
    distances, indices = knn_model.kneighbors(face_emb_array)
    labels = list()
    if min(distances[0]) < K_UNKOWN_THRESHOLD:
        for i in indices[0]:
            label = trainy[i]
            labels.extend([label])
            prediction = statistics.mode(labels)
            predict_name = out_encoder.inverse_transform([prediction])
    else:
        predict_name = "Unknown"
    
    text = f'{predict_name}, {min(distances[0]):.2f}'
    return text

def face_detection(net, frame, IMG_SIZE):
    blob = cv2.dnn.blobFromImage(frame, 1/255, (IMG_SIZE, IMG_SIZE),
                                [0, 0, 0], 1, crop=False)

    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames() # take 3 output layers
    outs = net.forward(output_layers)

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    confidences = []
    boxes = []

    # Each frame produces 3 outs corresponding to 3 output layers
    for out in outs:
        # 1 out has multiple predictions with length of 6
        for detection in out:
            confidence = detection[-1]
            # Extract position data of face area (only area with high confidence)
            if confidence > 0.5:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width    = int(detection[2] * frame_width)
                height   = int(detection[3] * frame_height)
                
                # Find the top left point of the bounding box 
                topleft_x = center_x - width//2 
                topleft_y = center_y - height//2
                confidences.append(float(confidence))
                boxes.append([topleft_x, topleft_y, width, height])

    # Perform non maximum suppression to eliminate redundant
    # overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    final_boxes = [boxes[i[0]] for i in indices]
    return final_boxes, confidences
