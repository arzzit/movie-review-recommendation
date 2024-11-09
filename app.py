
import streamlit as st
import tensorflow

from tensorflow.keras.datasets import imdb # type: ignore
from tensorflow.keras.preprocessing import sequence # type: ignore

(x_train,y_train) , (x_test,y_test) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()

model = tensorflow.keras.models.load_model('/workspaces/movie-review-recommendation/model.h5')

def transform_text(text):
  words = text.lower().split()
  encode = [word_index.get(word,2) + 3 for word in words]
  padded = sequence.pad_sequences([encode],maxlen = 500)
  return padded

def predict_sentiment(text):
  padded = transform_text(text)
  prediction = model.predict(padded)

  if prediction[0][0] > 0.55:
    sentiment = 'positive'
  else:
    sentiment = 'negative'
  return prediction[0][0] , sentiment

input = st.text_input(label="Review",value="")

score , sentiment = predict_sentiment(input)

st.write(f"Sentiment : {sentiment}")
st.write(f"Score : {round(score,2)}")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)





