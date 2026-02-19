import streamlit as st
import pandas as pd
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pypdf import PdfReader
load_dotenv()

def set_bg():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://mooddp.com/wp-content/uploads/2025/10/aesthetic-simple-dp-for-whatsapp.webp");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Make content readable */
        .main {{
            background-color: rgba(255,255,255,0.85);
            padding: 20px;
            border-radius: 15px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3 
)

prompt_template = ChatPromptTemplate.from_template(
"""
You are an expert ATS (Applicant Tracking System) and senior technical recruiter.

Your task is to evaluate how well a candidate's RESUME matches a JOB DESCRIPTION.

Be strict, realistic, and professional. Do NOT be overly positive.If the JD matches the skills mentioned in the resume is arround 70 to 80 percent than we can proceed with the candidate.
At the end , also suggest the possible cv improvments .

JOB DESCRIPTION:
{job_description}

CANDIDATE RESUME:
{resume_text}

Perform the following analysis:

1) Give a MATCH SCORE out of 10 based on:
   - Skills match
   - Experience match
   - Tools & technologies
   - Domain relevance
   - Seniority alignment

2) Decide if the profile is:
   - Highly Relevant
   - Moderately Relevant
   - Irrelevant

3) Provide MATCHED KEYWORDS / SKILLS (bullet points)

4) Provide MISSING KEYWORDS / SKILLS (bullet points)

5) Identify EXPERIENCE GAPS (bullet points)

6) Final Recommendation:
   Should the candidate APPLY for this job?
   Answer ONLY: YES / NO / MAYBE
7) Provide recommendation for possible improvement needed to apply this job.

Return output in STRICT format:

MATCH SCORE: X/10  
RELEVANCE: <Highly Relevant / Moderately Relevant / Irrelevant>

MATCHED SKILLS: 
- 

MISSING SKILLS:
- 

EXPERIENCE GAPS:
- 

FINAL VERDICT:
<YES / NO / MAYBE>

RECOMMENDARION : PROVIDES SKILS WHICH NEEDED IMPROVEMENTS IF ANY .
"""
)

# Function to read PDF resume
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

chain = prompt_template | llm
set_bg()

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: white;'>🤖 ATS Resume Checker</h1>", unsafe_allow_html=True)


st.write("Upload your resume and paste Job Description to check match score.")

# Job Description input
job_description = st.text_area("📄 Paste Job Description")

# Resume upload
uploaded_file = st.file_uploader("📎 Upload Resume (PDF)", type=["pdf"])
# Button
if st.button("🔍 Analyze Resume Match"):
    if uploaded_file and job_description:

        resume_text = extract_text_from_pdf(uploaded_file)

        inputs = {
            "job_description": job_description,
            "resume_text": resume_text
        }

        with st.spinner("Analyzing with AI..."):
            response = chain.invoke(inputs)

        st.subheader("📊 ATS Result")
        st.write(response.content)

    else:
        st.warning("Please upload resume and paste job description.")
