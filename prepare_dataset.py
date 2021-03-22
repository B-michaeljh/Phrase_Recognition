# https://github.com/musikalkemist/Deep-Learning-Audio-Application-From-Design-to-Deployment/blob/master/3-%20Implementing%20a%20Speech%20Recognition%20System%20in%20TensorFlow%202/prepare_dataset.py

import librosa
import os
import json

DATASET_PATH = "C:/Users/Brandon/Desktop/Speech/Chinese/"
JSON_PATH = "C:/Users/Brandon/Desktop/Speech/Chinese_data.json"
SAMPLES_TO_CONSIDER = 22050 * 2

"""
Mel Frequency Cepstral Coefficients
Extracts MFCCs from music dataset and saves them into a json file.
:param dataset_path (str): Path to dataset
:param json_path (str): Path to json file used to save MFCCs
:param num_mfcc (int): Number of coefficients to extract
:param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
:param hop_length (int): Sliding window for FFT. Measured in # of samples
:return:
"""

def preprocess_dataset(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512):

    # JSON file data mapping
    data = {
        "mapping": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    # loop all sub-folders
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # ensuring sub-folder level
        if dirpath is not dataset_path:
            # save label (sub-folder)
            label = dirpath.split("/")[-1]
            data["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))
            # process the audio files in sub-folder and store MFCCs
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                # slice audio files to ensure length consistency among different files
                signal, sample_rate = librosa.load(file_path)
                # validating sample lengths
                if len(signal) >= SAMPLES_TO_CONSIDER:
                    # ensure consistency of signal length
                    signal = signal[:SAMPLES_TO_CONSIDER]
                    # extract MFCCs
                    MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    # store data for audio file
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["labels"].append(i-1)
                    data["files"].append(file_path)
                    print("{}: {}".format(file_path, i-1))

    # save data in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, JSON_PATH)