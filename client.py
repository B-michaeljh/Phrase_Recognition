import requests

# server url
URL = "http://127.0.0.1:5000/predict"


# audio file we'd like to send for predicting keyword
FILE_PATH = "C:/Users/Brandon/AppData/LocalLow/DefaultCompany/LanguageVR/myfile.wav"

if __name__ == "__main__":

    # open files
    # file = open(FILE_PATH, "rb")

    # package stuff to send and perform POST request
    # values = {"file": ("Audio_File", file, "audio/wav")}
    # print(values)

    myfiles = {'file': open(FILE_PATH, 'rb')}
    
    response = requests.post(URL, files=myfiles)
    data = response.json()

    print("Predicted keyword: {}".format(data["keyword"]))

    