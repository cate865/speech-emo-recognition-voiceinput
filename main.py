import os
import numpy as np
import streamlit as st
from io import BytesIO
import streamlit.components.v1 as components
from st_custom_components import st_audiorec
import argparse
from model_arch import TIMNET_Model
import librosa
from sklearn.preprocessing import OneHotEncoder

# Feature extraction function


def get_feature(file_path: str, mfcc_len: int = 39, mean_signal_length: int = 100000):

    signal, fs = librosa.load(file_path)
    s_len = len(signal)

    if s_len < mean_signal_length:
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem),
                        'constant', constant_values=0)
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]
    mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=39)
    mfcc = mfcc.T
    feature = mfcc
    return feature


# One Hot Encoding emotions
emotions = [['angry', 'calm', 'disgust', 'fear', 'happy',
             'neutral', 'sad', 'surprise']]  # Emotions as ordered by the training one hot encoder

# One hot encode the emotions
encoder = OneHotEncoder()
encoder.fit_transform(np.array(emotions).reshape(-1, 1)).toarray()

# Creating and loading model weights
model_path = os.path.join('cpmodel_weights.h5')
args = argparse.Namespace(activation='relu', batch_size=64, beta1=0.93, beta2=0.98, data='RAVDE', dilation_size=8, dropout=0.1, epoch=300,
                          filter_size=39, kernel_size=2, lr=0.001, model_path='./Models/', random_seed=46, result_path='./Results/', split_fold=5, stack_size=1)
CLASS_LABELS = ('angry', 'calm', 'disgust', 'fear',
                'happy', 'neutral', 'sad', 'surprise')
ser_model = TIMNET_Model(args=args, input_shape=(
    196, 39), class_label=CLASS_LABELS)
ser_model.create_model()
ser_model.model.load_weights(model_path)


# Model prediction + UI
st.title('Speech Emotion Recognition with Live Input')
st.write('Record something, the app will display the predicted emotion.')
wav_audio_data = st_audiorec()


if wav_audio_data is not None:
    with open('myfile.wav', mode='bw') as f:
        f.write(wav_audio_data)
    # display audio data as received on the backend
    # st.audio(wav_audio_data, format='audio/wav')

    if st.button('Make Prediction'):
        
    
        X = get_feature('myfile.wav')
        X = np.expand_dims(X, axis=1)  # New shape will be (None, 1, 39)
        X = np.tile(X, (1, 196, 1))  # New shape will be (None, 196, 39)

        y = ser_model.model.predict(X)

        emotion = encoder.inverse_transform(y)[0][0]

        st.write('Predicted emotion: ' + emotion)

    # if emotion == 'fear':
    #     pass


# INFO: by calling the function an instance of the audio recorder is created
# INFO: once a recording is completed, audio data will be saved to wav_audio_data
