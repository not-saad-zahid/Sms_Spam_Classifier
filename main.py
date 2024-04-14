import streamlit as st
import nltk
import pickle
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
stopwords.words('english')
ps = PorterStemmer()
tfidf = pickle.load(open('vectorized.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


def text_transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


st.title('Spam Classifier')

input_sms = st.text_area("Enter Your Message")

if st.button('Predict'):

    # 1: Preprocess
    transformed_sms = text_transform(input_sms)
    # 2: Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3: Predict
    result = model.predict(vector_input)[0]
    # 4: Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
