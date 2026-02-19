#!/usr/bin/env python
# coding: utf-8


import streamlit as st
import pandas as pd
import os
##from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
##from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
#from langchain.chains import LLMChain
#from langchain.chains.llm import LLMChain

from dotenv import load_dotenv
from Step0 import run_preprocessing
#from Step2 import run_step2

# Set environment variables LLM KeyError

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3  # Set your desired temperature here
)


# Define the prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
You are a senior ITSM knowledge management expert.

Your task is to convert resolved incident tickets into clear, reusable
knowledge base articles used by L1/L2 support engineers.

Incident Details:
- Issue Description: {Issue_Description}
- Category: {Category}
- Sub-Category: {Subcategory}
- Resolution Notes: {Resolution_Notes}

Instructions:
1. Understand the full incident lifecycle from issue to resolution.
2. Rewrite the resolution in a concise, professional, and actionable manner.
3. Convert the resolution into clear step-by-step instructions.
4. If multiple scenarios exist, clearly separate them.
5. Do NOT include unnecessary background text.

Output Format (STRICT):
Category: {Category} | Sub-Category: {Subcategory}
---
Summary:
<2–3 line professional incident summary>

Resolution Steps:
- Step 1:
- Step 2:
- Step 3:

Notes:
- Optional clarifications if needed
---
"""
)


# Create the LLM chain
chain = prompt_template | llm


st.title("AI Bees - Operational Assistant")



tab1, tab2 = st.tabs(["Step 0: Incident Preprocessor","Step 1: Incident Summarization"])

with tab1:

    run_preprocessing()
    
    
with tab2:
    st.header("Incident Summarization")
    # Input for number of rows to process
    param = st.number_input("Enter number of rows to process:", min_value=1, value=20, step=1)

    # Start button
    if st.button("▶ Compose Incident Summary"):
    ##if st.button("Start Processing"):
        try:
            df2 = pd.read_csv('inc_data_final.csv', encoding="latin1")
            df2 = df2.dropna(subset=['Subcategory', 'Assignment_group', 'Resolution_Notes'])
            df = df2.head(param)
            
            def generate_notes(row):
                inputs = {
                    "Issue_Description": row["Issue_Description"],
                    "Category": row["Category"],
                    "Subcategory": row["Subcategory"],
                    "Resolution_Notes": row["Resolution_Notes"]
                }
                response = chain.invoke(inputs)
                return response.content

            df["Incident_Summary"] = df.apply(generate_notes, axis=1)
            # Save the output to a CSV file
            output_file = "step1_output.csv"
            header = ["Issue_Description", "Category", "Subcategory", "Assignment_group", "Resolution_Notes", "Incident_Summary"]
            df.to_csv(output_file, index=False, columns=header)

            # Display the top 5 rows
            st.success(f"✅   Incident Summarization is completed.. {output_file}")
            st.dataframe(df)

        except Exception as e:
            st.error(f"An error occurred: {e}")

