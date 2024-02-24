
from Translate.pipeline.prediction import PredictionPipeline
import streamlit as st # pip install streamlit==0.82.0


st.set_page_config(page_title='Simply! Translate', layout='wide', initial_sidebar_state='expanded')
st.title("Language Translator using Transformers from Scratch")
text = st.text_area("Enter text:",height=None,max_chars=None,key=None,help="Enter your text here")


if st.button('Translate Sentence'):
    if text == "":
        st.warning('Please **enter text** for translation')

    else:
        obj = PredictionPipeline()
        translation_text = obj.translate(text)
        st.write(translation_text)
else:
    pass