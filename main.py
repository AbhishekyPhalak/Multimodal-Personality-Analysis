import librosa
import librosa.display
import numpy as np
import pandas as pd
import ffmpeg as ff
import os
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import random
import cv2
import pickle
import datetime

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from keras_vggface import VGGFace
from typing import List, Tuple, Dict

import os
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import moviepy.editor

# Define the sampling rate.
SR = 16000

# Load the YAMNet model.
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')


import math

def extract_audio_from_video(file_path: str) -> np.ndarray:
    # Load the video using moviepy editor
    video = moviepy.editor.VideoFileClip(file_path)
    
    # Extract the audio from the video
    audio = video.audio.to_soundarray()
    
    # Resample the audio to 44100 Hz
    audio_resampled = librosa.resample(audio[:,0], orig_sr=video.audio.fps, target_sr=44100)
    
    # Define segment length and overlap
    segment_length = 1 # in seconds
    overlap = 0.5 # in seconds
    
    # Calculate number of segments
    segment_size = int(segment_length * 44100)
    hop_size = int((segment_length - overlap) * 44100)
    num_segments = int(math.ceil(len(audio_resampled) / hop_size))
    
    # Extract embeddings for each segment using YamNet model
    embeddings = []
    for i in range(num_segments):
        start = i * hop_size
        end = min(start + segment_size, len(audio_resampled))
        segment = audio_resampled[start:end]
        segment_embedding = yamnet_model(segment)
        embeddings.append(segment_embedding)
    
    # Concatenate embeddings into a single array
    embeddings_array = np.concatenate(embeddings)
    
    return np.frombuffer(embeddings_array,np.float32)


import cv2

def get_number_of_frames(file_path: str) -> int:
    video = cv2.VideoCapture(file_path)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return num_frames



def extract_N_video_frames(file_path: str, number_of_samples: int = 6) -> List[np.ndarray]:
    nb_frames = int(get_number_of_frames(file_path= filePath))
    
    video_frames = []
    random_indexes = random.sample(range(0, nb_frames), number_of_samples)
    
    cap = cv2.VideoCapture(filePath)
    for ind in random_indexes:
        cap.set(1,ind)
        res, frame = cap.read()
        video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    del cap, random_indexes
    return video_frames



def extract_vggface_features(image: np.ndarray) -> np.ndarray:
    # Load VGGFace model
    vggface = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3))
    
    # Preprocess image for VGGFace
    image = cv2.resize(image, (224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.vgg16.preprocess_input(image)
    
    # Extract features
    features = vggface.predict(np.array([image]))
    
    return features.flatten()

def reading_label_data(file_name: str, dictionary: Dict[str,str]) -> np.ndarray:
    features = ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']
    extracted_data = [float(dictionary[label][file_name]) for label in features]
    return np.stack(extracted_data).reshape(5,1)



def preprocessing_input(file_path: str, file_name: str, dictionary: Dict[str,str], training: bool = True) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    #Audio
    extracted_audio_raw = extract_audio_from_video(file_path= filePath)
    
    #Video
    sampled = extract_N_video_frames(file_path= file_path, number_of_samples= 6)
    features = [extract_vggface_features(im) for im in sampled]
    preprocessed_video = np.stack(features)
    
    #Ground Truth
    video_gt = reading_label_data(file_name= file_name, dictionary= dictionary)
    del sampled
    return (extracted_audio_raw, preprocessed_video, video_gt)


training_set_data = []
path = r'train'
gt = pickle.load( open( "annotation_training.pkl", "rb" ), encoding='latin1')
t1 = datetime.datetime.utcnow()
for filename3 in os.listdir(path):
	fp3 = path + '/' + filename3
	for filename2 in os.listdir(fp3):
    		filePath2 = fp3 + '/' + filename2
    		for filename in os.listdir(filePath2):
        		filePath = filePath2 + '/' + filename
        		print(filePath)
        		training_set_data.append(preprocessing_input(file_path= filePath, file_name= filename, dictionary= gt, training= True))
t2 = datetime.datetime.utcnow()
#Measuring execution time
print('Elapsed time: ' + str(t2-t1))


savename = 'training_set_1.dat'
with open(savename, "wb") as f:
    pickle.dump(training_set_data, f)

validation_set_data = []
path = r'val'
gt = pickle.load( open( "annotation_validation.pkl", "rb" ), encoding='latin1' )
t1 = datetime.datetime.utcnow()
for filename3 in os.listdir(path):
	fp3 = path + '/' + filename3
	for filename2 in os.listdir(fp3):
    		filePath2 = fp3 + '/' + filename2
    		for filename in os.listdir(filePath2):
        		filePath = filePath2 + '/' + filename
        		print(filePath)
        		validation_set_data.append(preprocessing_input(file_path= filePath, file_name= filename, dictionary= gt, training= True))
t2 = datetime.datetime.utcnow()
#Measuring execution time
print('Elapsed time: ' + str(t2-t1))

savename = 'validation_set_1.dat'
with open(savename, "wb") as f:
    pickle.dump(validation_set_data, f)

test_set_data = []
path = r'test'
gt = pickle.load( open( "annotation_test.pkl", "rb" ), encoding='latin1' )
t1 = datetime.datetime.utcnow()
for filename3 in os.listdir(path):
	fp3 = path + '/' + filename3
	for filename2 in os.listdir(fp3):
    		filePath2 = fp3 + '/' + filename2
    		for filename in os.listdir(filePath2):
        		filePath = filePath2 + '/' + filename
        		print(filePath)
        		test_set_data.append(preprocessing_input(file_path= filePath, file_name= filename, dictionary= gt, training= True))
t2 = datetime.datetime.utcnow()
#Measuring execution time
print('Elapsed time: ' + str(t2-t1))

savename = 'test_set.dat'
with open(savename, "wb") as f:
    pickle.dump(test_set_data, f)

import pickle

with open('training_set_1.dat', "rb") as training_file:
    train_set_data = pickle.load(training_file)

with open('validation_set_1.dat', "rb") as validation_file:
    validation_set_data = pickle.load(validation_file)

with open('test_set_video3.dat', "rb") as test_file:
    test_set_data = pickle.load(test_file)

train_random_index = random.randint(0, len(train_set_data)-1)
validation_random_index = random.randint(0, len(validation_set_data)-1)
test_random_index = random.randint(0, len(test_set_data)-1)

personality_train = train_set_data[train_random_index][2]
personality_validation = validation_set_data[validation_random_index][2]
personality_test = test_set_data[test_random_index][2]
personalities = ['Neuroticism','Extraversion','Agreeableness','Conscientiousness','Openness']


def reshape_to_expected_input(dataset: List[Tuple[np.ndarray,np.ndarray,np.ndarray]]) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    
    x0_list = []
    x1_list = []
    y_list = []
    for i in range(0,len(dataset)):
        if np.shape(dataset[i][0]) == (186,):
            x0_list.append(dataset[i][0])
            x1_list.append(dataset[i][1])
            y_list.append(dataset[i][2])
    return (np.stack(x0_list),np.stack(x1_list),np.stack(y_list))


train_input = reshape_to_expected_input(dataset= train_set_data)
del train_set_data
validation_input = reshape_to_expected_input(dataset= validation_set_data)
del validation_set_data
test_input = reshape_to_expected_input(dataset= test_set_data)
del test_set_data


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, LSTM, Bidirectional, Lambda, Dropout, Concatenate, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling1D, Conv1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import TimeDistributed

audio_input = Input(shape=(186,))
audio_reshape = Reshape((186, 1))(audio_input)
audio_model = Conv1D(32, kernel_size=3, activation='relu')(audio_reshape)
audio_model = BatchNormalization()(audio_model)
audio_model = MaxPooling1D(pool_size=2)(audio_model)
audio_model = Flatten()(audio_model)
audio_model = Dense(128, activation='relu')(audio_model)
audio_subnetwork = Model(inputs=audio_input, outputs=audio_model)

from keras.applications import vgg16
from tensorflow.keras.layers import Reshape
from tensorflow.keras.applications import ResNet50

from tensorflow.keras.layers import Input

visual_input = Input(shape=(6, 25088))

# Define the LSTM model
visual_model = Sequential()
visual_model.add(LSTM(64, input_shape=(6, 25088)))  # input_shape should match the shape of preprocessed_video
visual_model.add(Dense(1, activation='sigmoid'))

# Compile the model
visual_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Create a Model that takes visual_input and outputs the output of the last layer of visual_model
visual_subnetwork = Model(inputs=visual_input, outputs=visual_model(visual_input))

combined = Concatenate()([audio_subnetwork.output, visual_subnetwork.output])
final1 = Dense(256, activation='relu')(combined)
final2 = Dense(5, activation='linear')(final1)
combined_network = Model(inputs=[audio_input, visual_input], outputs=final2)
combined_network.summary()

combined_network.compile(optimizer = 'adam',loss = 'mean_absolute_error',metrics=['mae'])

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(patience=10)

history = combined_network.fit(x = [train_input[0],train_input[1]],
                               y = train_input[2],
                               validation_data = ([validation_input[0],validation_input[1]],validation_input[2]),
                               epochs = 20,
                               verbose = 1,
                               callbacks = [early_stopping])

plt.figure(figsize=(9,4))

plt.subplot(1,2,1)
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train MAE', 'Validation MAE'], loc='upper right')


y_pred = combined_network.predict([test_input[0], test_input[1]])

acc = 0
for i in range(len(y_pred)):
    for j in range(len(y_pred[i])):
        if abs(y_pred[i][j] - test_input[2][i][j]) <= 0.5:
            acc += 1

accuracy = acc / (len(y_pred) * len(y_pred[0]))
print("Accuracy: {:.2f}%".format(accuracy * 100))






