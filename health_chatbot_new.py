from flask import Flask, jsonify, request, render_template
from flask_cors import CORS, cross_origin
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import random
import json

app = Flask(__name__)
CORS(app)


#I'm looking for a workout tracker app. Any recommendations?
# Load your chatbot_new model and tokenizer


model_path = 'chatbot_new'  # Adjust the path based on your actual model location
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
chatbot = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Load the data from the JSON file
with open("Data.json") as file:
    data = json.load(file)

@app.route('/')
def index():
    return render_template('chat_interface.html')

@app.route('/get-response', methods=['POST'])
@cross_origin()
def get_response():
    # Handle the actual POST request
    message = request.json.get('message', '')
    
    if not message:
        return jsonify({'response': 'No message received.'})

    # Generate response using chatbot_new
    result = chatbot(message)
    label_str = result[0]['label']
    
    for intent in data['intents']:
        if intent['tag'] == label_str:
            response_text = random.choice(intent['responses'])
            response = {'response': response_text}
            print('Server Response:', response)
            return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
