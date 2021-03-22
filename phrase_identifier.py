# https://github.com/musikalkemist/Deep-Learning-Audio-Application-From-Design-to-Deployment/blob/master/4-%20Making%20Predictions%20with%20the%20Speech%20Recognition%20System/keyword_spotting_service.py

import librosa
import tensorflow as tf
import numpy as np

SAVED_MODEL_PATH = "C:/Users/Brandon/Desktop/Speech/C_model.h5"
SAMPLES_TO_CONSIDER = 22050 * 2

"""Singleton class for keyword spotting inference with trained models.
:param model: Trained model
"""

class _Phrase_Identifier:

    """model = None"""
    _mapping = [
        "KaFeiDaiZou",
        "NiHao",
        "XieXie",
        "ZaiJian",
        "ZaoShangHao"
    ]
    _instance = None

    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):

        # load audio file
        signal, sample_rate = librosa.load(file_path)

        if len(signal) >= SAMPLES_TO_CONSIDER:
            # ensure consistency of the length of the signal
            signal = signal[:SAMPLES_TO_CONSIDER]

            # extract MFCCs
            MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                         hop_length=hop_length)
        return MFCCs.T

    """Extract MFCCs from audio file.
    :param file_path (str): Path of audio file
    :param num_mfcc (int): # of coefficients to extract
    :param n_fft (int): Interval we consider to apply STFT. Measured in # of samples
    :param hop_length (int): Sliding window for STFT. Measured in # of samples
    :return MFCCs (ndarray): 2-dim array with MFCC data of shape (# time steps, # coefficients)
    """

    def predict(self, file_path):

        # extract MFCC
        MFCCs = self.preprocess(file_path)

        # 4 dimensional array to feed the model for prediction: (# samples, # time steps, # coefficients, 1)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # making the prediction and returning the phrase
        predictions = self.model.predict(MFCCs)
        print(predictions)
        predicted_index = np.argmax(predictions)
        predicted_phrase = self._mapping[predicted_index]
        return predicted_phrase

    """
    :param file_path (str): Path to audio file to predict
    :return predicted_keyword (str): Keyword predicted by the model
    """

def Phrase_Identifier():


    # ensure an instance is created only the first time the factory function is called
    if _Phrase_Identifier._instance is None:
        _Phrase_Identifier._instance = _Phrase_Identifier()
        _Phrase_Identifier.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    return _Phrase_Identifier._instance

    """Factory function for Keyword_Spotting_Service class.
    :return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
    """


if __name__ == "__main__":

    # create 2 instances of the keyword spotting service
    kss = Phrase_Identifier()
    kss1 = Phrase_Identifier()

    # check that different instances of the keyword spotting service point back to the same object (singleton)
    assert kss is kss1

    keyword2 = kss.predict("C:/Users/Brandon/Desktop/Speech/Chinese/ZaoShangHao/zaoshanghao010.wav")
    print(keyword2)


    # make a prediction
    # keyword = kss.predict(TEST_SAMPLE)
    # print(keyword)


    keyword2 = kss.predict("C:/Users/Brandon/Desktop/Speech/Chinese/NiHao/NiHao038.wav")
    print(keyword2)
    keyword2 = kss.predict("C:/Users/Brandon/Desktop/Speech/Chinese/KaFeiDaiZou/KaFeiDaiZou023.wav")
    print(keyword2)
    keyword2 = kss.predict("C:/Users/Brandon/Desktop/Speech/Chinese/XieXie/XieXie021.wav")
    print(keyword2)
    keyword2 = kss.predict("C:/Users/Brandon/Desktop/Speech/Chinese/ZaiJian/ZaiJian025.wav")
    print(keyword2)

