from flask import Flask, render_template, redirect, request, flash
import os
from os.path import join, dirname, realpath
from werkzeug.utils import secure_filename
import tensorflow as tf
import librosa
import numpy as np
import matplotlib.pyplot as plt
UPLOADS_PATH = join(dirname(realpath(__file__)), 'static\\uploads\\')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOADS_PATH
app.secret_key = "secret key"


def preemphasis(audio_signal):
    alpha = 0.95
    preemphasized_signal = np.append(
        audio_signal[0], audio_signal[1:] - alpha * audio_signal[:-1])
    return preemphasized_signal


def make_mel_spectrogram(audio_signal, sr):
    mel_spectrogram = librosa.feature.melspectrogram(
        y=np.mean(audio_signal, axis=1), sr=sr, n_mels=128)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    return mel_spectrogram


def frame_and_remove_silence(signal):
    frames = librosa.util.frame(signal, frame_length=2048, hop_length=512)
    frame_energy = np.sum(frames ** 2, axis=1)
    frames_non_silent = frames[frame_energy > np.percentile(frame_energy, 5)]
    return frames_non_silent


def preprocessing(audio):
    signal, sr = librosa.load(audio)
    preemphasized_signal = preemphasis(signal)
    # Frame and remove silence from the audio signal
    framed_signal = frame_and_remove_silence(preemphasized_signal)
    # Make the mel spectrogram of the framed audio signal
    mel_spectrogram = make_mel_spectrogram(framed_signal, sr)
    return mel_spectrogram


with tf.device('/CPU:0'):
    model = tf.keras.models.load_model('model.keras')
def getPrediction(filename):
    spectrogram = preprocessing(UPLOADS_PATH+filename)
    print("spectrogram created")
    S_np = librosa.core.db_to_amplitude(spectrogram)
    plt.imsave('./static/spectro/' + filename.split(".")[0]+'.png',S_np, format='png')
    plt.close()
    print(filename.split(".")[0]+'.png'+" saved")
    image = tf.keras.utils.load_img('./static/spectro/' + filename.split(".")[0]+'.png', target_size=(480, 640, 3))
    image = tf.keras.utils.img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    with tf.device('/CPU:0'):
        ypre = model.predict(image)
    print(ypre)
    labels = ['CommonCuckoo', 'CommonKestrel', 'CommonRingedPlover', 'LittleOwl', 'NorthernRaven']

    label = labels[np.argmax(ypre)]
    return label


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/file', methods=['GET', 'POST'])
def submit_file():
    if request.method == 'POST':
        if 'audio_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['audio_file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            label= getPrediction(filename)
            flash(label)
            # flash(acc)
            flash(filename)
            return redirect('/')


if __name__ == "__main__":
    app.run()
