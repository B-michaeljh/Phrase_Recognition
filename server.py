import random
import os
from flask import Flask, request, jsonify
from phrase_identifier import Phrase_Identifier

# instantiate flask app
app = Flask(__name__)


# :return (json): This endpoint returns a json file with the following format:
#    {
#        "keyword": "NiHao"
#    }
# get file from POST request and save it

@app.route("/predict", methods=["POST"])
def predict():

    audio_file = request.files["file"]
    print(audio_file)
    file_name = str(random.randint(0, 100000))
    audio_file.save(file_name)

    # instantiate phrase identifier singleton and get prediction
    PI = Phrase_Identifier()
    predicted_phrase = PI.predict(file_name)

    # remove the file as it's no longer needed
    os.remove(file_name)

    # return the identified phrase in a JSON file
    result = {"keyword": predicted_phrase}
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False)


