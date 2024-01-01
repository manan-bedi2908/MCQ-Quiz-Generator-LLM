import streamlit as st
from langchain.llms import OpenAI, HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import pandas as pd
import os
from utils import parse_file

load_dotenv()

llm = HuggingFaceHub(repo_id = 'distilbert-base-uncased',
                         model_kwargs = {'temperature': 0.5,
                                         'max_length': 512})

st.title('MCQ Creator Application using LangChain')

with st.form('user_inputs'):
    uploaded_file = st.file_uploader('Upload a PDF or Text File')
    mcq_count = st.number_input('No. of MCQs', min_value = 3, max_value = 50)
    grade = st.number_input('Grade', min_value = 1, max_value = 12)
    tone = st.text_input('Quiz Tone', max_chars = 100, placeholder='simple')
    button = st.form_submit_button('Create MCQs')

    if button and uploaded_file is not None and mcq_count and grade and tone:
        with st.spinner('Generating...'):
            try:
                text = parse_file(uploaded_file)
            


