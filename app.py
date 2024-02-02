from flask import Flask, render_template, request
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from keras.models import load_model
import pyttsx3  # Text-to-speech library

app = Flask(__name__)

# Load the saved model and data
model = load_model('chatbot_model.h5')
intents = json.loads(open('intent.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Utility functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def speak_response(response):
    engine.say(response)
    engine.runAndWait()  # Wait for speech to finish

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    message = request.form['message']
    try:
        res = chatbot_response(message)
        speak_response(res)  # Speak the response
    except:
        res = 'You may need to rephrase your question.'
    return render_template('chat.html', message=message, response=res)

if __name__ == '__main__':
    app.run(debug=True)
