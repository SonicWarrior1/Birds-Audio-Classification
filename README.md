
# Birds Audio Classification using CNN

### Project Goal:
The goal of this project is to develop a convolutional neural network (CNN) model that can classify bird species from their audio recordings.
### Project Background: 
Bird sound classification is a challenging task due to the high variability in bird calls, the presence of background noise, and the need to classify a large number of species. However, CNNs have been shown to be effective in image classification tasks, and they have also been used for audio classification tasks.
### Project Methodology: 
* The proposed methodology will involve the following steps:
* Collect a dataset of audio recordings of bird calls.
* Extract features from the audio recordings, such as the spectrogram.
* Train a CNN model on the extracted features.
* Evaluate the performance of the CNN model on a test set of audio recordings.
### Prerequiste
* Python(v3.10.2)
* Modules used:
    * Tensorflow (v2.10.0)
    * pandas (v1.4.1)
    * scikit-learn (v1.1.2)
    * numpy (v1.24.4)
    * librosa (v0.10.0.post2) => For audio processing
    * opencv-python (v4.6.0.66)
* IDE used : VSCode

### Usage
1. Clone the Repository
```sh
git clone https://github.com/SonicWarrior1/Birds-Audio-Classification.git
```
2. First run the [data_preperation.ipynb](data_preperation.ipynb) to download the audio set and generate spectrograms for the audios.

3. Now run the [model_train.ipynb](model_train.ipynb) file to train the CNN and save the model.

4. Run the Flask app file [app.py](app.py) using the following command in your terminal.
```python
flask run
```
5. Now select the audio file for input and press the submit button to predict the bird class.
