import streamlit as st
from langchain.llms import OpenAI, HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import pandas as pd
import os
import traceback
import json
from utils import parse_file, RESPONSE_JSON, get_table_data

load_dotenv()

llm = HuggingFaceHub(repo_id = 'google/flan-t5-small',
                         model_kwargs = {'temperature': 0.5,
                                         'max_length': 512})

template = """
Text: {text}
You are an expert MCQ Creator. Given the above text, it is your job to \
create a quiz of {number} multiple choice questions for grade {grade} students in {tone} tone.
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}
"""

quiz_generation_prompt = PromptTemplate(
    input_variables=['text', 'number', 'grade', 'tone', 'response_json'],
    template = template
)

quiz_chain = LLMChain(llm = llm, 
                      prompt = quiz_generation_prompt,
                      output_key = 'quiz',
                      verbose = True)

template = """
You are an expert English Grammarian and Writer. Given a Multiple Choice Quiz for {grade} grade students. \
You need to evaluate the complexity of the question and give a complete analysis of the quiz if the students
will be able to understand the questions and answer them. Only use at max 50 words for complexity analysis.
If the quiz is not at par with the cognitive and analytical abilities of the students, ]
Update the quiz questions which needs to be changed and change the tone such that it perfectly fits the students cognitive and analytical capabilities.
Quiz_MCQ:
{quiz}

Critique from an expert English Writer of the above quiz
"""

quiz_evaluation_prompt = PromptTemplate(
    input_variables=['grade', 'quiz'],
    template = template
)

review_chain = LLMChain(llm = llm,
                        prompt = quiz_evaluation_prompt,
                        output_key = 'review', 
                        verbose = True)

generate_evaluate_chain = SequentialChain(chains = [quiz_chain, review_chain],
                                          input_variables=['text', 'number', 'grade', 'tone', 'response_json'],
                                          output_variables=['quiz', 'review'],
                                          verbose=True)

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
                # with get_openai_callback() as cb:
                response=generate_evaluate_chain(
                        {
                        "text": text,
                        "number": mcq_count,
                        "grade":grade,
                        "tone": tone,
                        "response_json": json.dumps(RESPONSE_JSON)
                        }
                    )
                st.write(response)
                

            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error('Error!...')
            
            # else:
            #     if isinstance(response, dict):
            #         #Extract the quiz data from the response
            #         quiz=response.get("quiz", None)
            #         if quiz is not None:
            #             # table_data=get_table_data(quiz)
            #             # st.write(quiz)
            #             # if table_data is not None:
            #             #     df=pd.DataFrame(table_data)
            #             #     df.index=df.index+1
            #             #     st.table(df)
            #             #     #Display the review in atext box as well
            #             #     st.text_area(label="Review", value=response["review"])
            #             # else:
            #             #     st.error("Error in the table data")

            #     else:
            #         st.write(response)
