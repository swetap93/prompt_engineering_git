from dotenv import load_dotenv
import os

load_dotenv()  

from langchain_google_genai import ChatGoogleGenerativeAI


llm = ChatGoogleGenerativeAI(
    model= "gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,
    max_output_tokens=80
    )

#fewshot prompting for email sunject 

messages = [
    {
        "role": "system",
        "content": "You generate a short professional email subject from the email content."
    },

    {
        "role": "user",
        "content": "Email content: 'Hi Manager, I am not feeling well today and would like to take a sick leave. I will resume work tomorrow.'"
    },
    {
        "role": "assistant",
        "content": "Sick Leave Request"
    },

    {
        "role": "user",
        "content": "Email content: 'Hello Team, Due to a scheduling conflict, I won’t be able to attend today’s meeting. Can we move it to tomorrow afternoon?'"
    },
    {
        "role": "assistant",
        "content": "Request to Reschedule Meeting"
    },

       # 👇 new email to predict subject
    {
        "role": "user",
        "content": "Email content: 'Hello Team, The project milestone has been completed successfully and sharing the progress report.'"
    }
]


response=llm.invoke(messages)
print(response.content)