import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import random
import json

# Load your chatbot_new model and tokenizer
model_path = 'chatbot_new'  # Adjust the path based on your actual model location
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
chatbot = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Load the data from the JSON file
with open("Data.json") as file:
    data = json.load(file)

def get_response(message):
    # Generate response using chatbot_new
    result = chatbot(message)
    label_str = result[0]['label']
    
    for intent in data['intents']:
        if intent['tag'] == label_str:
            response_text = random.choice(intent['responses'])
            response = {'response': response_text}
            print('Server Response:', response)
            return response

def main():
    st.title('Chatbot')
    message = st.text_input("Enter your message")
    if st.button('Send'):
        if message:
            response = get_response(message)
            st.write(response)
        else:
            st.write('No message received.')

if __name__ == '__main__':
    main()
