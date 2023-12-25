from nltk.stem.porter import PorterStemmer
import pandas as pd
from pandas.core.algorithms import mode
from tensorflow.keras.models import load_model
import streamlit as st
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
import re
import numpy as np
from nltk import PorterStemmer
from tensorflow.keras.layers import Dense
ignore_words = [',','?','/','!','=','+','|','#','&','~','(','{',']','}',')','^','-','_','$']
pr = PorterStemmer()
def preprocces_word(word_):
    voc_size = 5000
    sent_legth = 40
    word_ = re.sub('[^a-zA-z]',' ',word_)
    word_ = word_.lower()
    word_ = word_.split()
    word_  = [pr.stem(w) for w in word_ if w not in ignore_words]
    word_ = ' '.join(word_)
    on_hot_test = [one_hot(word_,voc_size)]
    embedd_sequeces_test = pad_sequences(on_hot_test,maxlen=sent_legth,padding='pre')
    embedd_sequeces_test = np.array(embedd_sequeces_test)
    return embedd_sequeces_test


def load_model_():
    model = load_model('fake_new.h5')
    return model

model =  load_model_()
classes = ['Flase','True']
st.set_page_config(layout='wide')
st.title("Fake news Detection Project")
input = st.text_area("Enter news to detect")
if st.button("Detect"):
    word = preprocces_word(input)
    ypredict = model.predict(word)
    if ypredict<0.5:
        st.subheader("The news is fake")
    else:
        st.subheader("The news is True")

