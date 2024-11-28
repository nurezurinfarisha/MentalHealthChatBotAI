import streamlit as st
from collections import Counter
import joblib
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
import string
import re
import joblib
import json
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.metrics import label_ranking_average_precision_score as mean_reciprocal_rank
from sklearn.metrics import ndcg_score

# Load pre-trained model and artifacts
model1 = load_model(r"C:\Users\user\OneDrive\Desktop\University\Sem_5\try\mentalhealth - Copy\model1_weights.h5")  # Replace with the actual path
model2 = load_model(r"C:\Users\user\OneDrive\Desktop\University\Sem_5\try\mentalhealth - Copy\model2_weights.h5")
model3 = load_model(r"C:\Users\user\OneDrive\Desktop\University\Sem_5\try\mentalhealth - Copy\model3_weights.h5")
model4 = load_model(r"C:\Users\user\OneDrive\Desktop\University\Sem_5\try\mentalhealth - Copy\model4_weights.h5")
model5 = load_model(r"C:\Users\user\OneDrive\Desktop\University\Sem_5\try\mentalhealth - Copy\model5_weights.h5")

tokenizer_t = joblib.load(r"C:\Users\user\OneDrive\Desktop\University\Sem_5\try\Dumps\tokenizer_t.pkl")  # Replace with the actual path
vocab = joblib.load(r"C:\Users\user\OneDrive\Desktop\University\Sem_5\try\Dumps\vocab.pkl")

df2 = pd.read_csv(r"C:\Users\user\OneDrive\Desktop\University\Sem_5\try\mentalhealth - Copy\response.csv")  # Replace with the actual path

# preprocessing text

lemmatizer = WordNetLemmatizer()

vocab = Counter()
labels = []
def tokenizer(entry):
    tokens = entry.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens]
    tokens = [word.lower() for word in tokens if len(word) > 1]
    return tokens

def remove_stop_words(tokenizer,df,feature):
    doc_without_stopwords = []
    for entry in df[feature]:
        tokens = tokenizer(entry)
        doc_without_stopwords.append(' '.join(tokens))
    df[feature] = doc_without_stopwords
    return

def tokenizer(entry):
    tokens = entry.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens]
    # stop_words = set(stopwords.words('english'))
    # tokens = [w for w in tokens if not w in stop_words]
    tokens = [word.lower() for word in tokens if len(word) > 1]
    return tokens

def remove_stop_words_for_input(tokenizer,df,feature):
    doc_without_stopwords = []
    entry = df[feature][0]
    tokens = tokenizer(entry)
    doc_without_stopwords.append(' '.join(tokens))
    df[feature] = doc_without_stopwords
    return df

def encode_input_text(tokenizer_t,df,feature):
    t = tokenizer_t
    entry = entry = [df[feature][0]]
    encoded = t.texts_to_sequences(entry)
    padded = pad_sequences(encoded, maxlen=10, padding='post')
    return padded

def get_pred(model,encoded_input):
    pred = np.argmax(model.predict(encoded_input))
    return pred

def bot_precausion(df_input,pred):
    words = df_input.questions[0].split()
    if len([w for w in words if w in vocab])==0 :
        pred = 1
    return pred


def get_response(df2,pred):
    upper_bound = df2.groupby('labels').get_group(pred).shape[0]
    r = np.random.randint(0,upper_bound)
    responses = list(df2.groupby('labels').get_group(pred).response)
    return responses[r]

# Streamlit app
st.title("Mental health chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Accept user input
prompt = st.text_input("What is up?")
send_button = st.button("Send")

if send_button and prompt:
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Preprocess user input for the model
    df_input = pd.DataFrame([prompt], columns=['questions'])
    df_input = remove_stop_words_for_input(tokenizer, df_input, 'questions')
    encoded_input = encode_input_text(tokenizer_t, df_input, 'questions')

    # Get predictions from all models for the user input
    pred1 = get_pred(model1, encoded_input)
    pred2 = get_pred(model2, encoded_input)
    pred3 = get_pred(model3, encoded_input)
    pred4 = get_pred(model4, encoded_input)
    pred5 = get_pred(model5, encoded_input)

    pred1 = bot_precausion(df_input, pred1)
    pred2 = bot_precausion(df_input, pred2)
    pred3 = bot_precausion(df_input, pred3)
    pred4 = bot_precausion(df_input, pred4)
    pred5 = bot_precausion(df_input, pred5)

    response1 = get_response(df2, pred1)
    response2 = get_response(df2, pred2)
    response3 = get_response(df2, pred3)
    response4 = get_response(df2, pred4)
    response5 = get_response(df2, pred5)

    models = [model1, model2, model3, model4, model5]

    # Use Counter to find the most common prediction
    all_predictions = [pred1, pred2, pred3, pred4, pred5]
    final_prediction = Counter(all_predictions).most_common(1)[0][0]

    # Get response based on the final prediction
    final_response = get_response(df2, final_prediction)

    # Display the final response in the chat message container
    with st.chat_message("bot"):
        st.markdown(final_response)
    # Add bot response to chat history
    st.session_state.messages.append({"role": "bot", "content": final_response})
