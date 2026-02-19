from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()  

llm = ChatGoogleGenerativeAI(
    model= "gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,
    #max_output_tokens=200
    )
""" *******Use Case: Automated Customer Support Ticket Categorization & Priority Routing

Problem Statement:
Customer support teams receive a large number of tickets daily through emails, chat, and forms. Manually reading each ticket, understanding the issue, assigning a category, and deciding the priority is time-consuming and inconsistent. This often leads to delayed responses and poor customer experience.

Solution:
This GenAI application uses Few-Shot Prompting + Chain-of-Thought reasoning to automatically analyze incoming customer support tickets and perform:

Ticket category classification (Billing, Technical Issue, Account, Refund, etc.)

Priority detection (Low, Medium, High, Critical)

Basic reasoning based on urgency, tone, and impact

Intelligent routing to the correct support team ******* """

examples_str = """
Example 1:
Ticket: "My password isn't working and I have a huge presentation in 10 minutes! Please help!"
Reasoning:
1. The user is locked out (Access Issue).
2. The tone is high-stress ("huge presentation").
3. The timeframe is immediate (10 minutes).
Category: Account Access
Priority: CRITICAL

Example 2:
Ticket: "I noticed a small typo on the 'About Us' page. It's not urgent, just thought you'd like to know."
Reasoning:
1. This is a visual/content error (UI/UX).
2. The tone is calm and helpful.
3. The user explicitly stated it is not urgent.
Category: Website Content
Priority: LOW
"""

current_ticket="""The payment was deducted from my account but the order is still showing as failed.
This has happened twice today and I need this resolved urgently as its blocking my purchase."""

messages=[
    {
        "role": "system",
        "content": f""" You are a support Triage Bot.follow the resoning style and categorization as per the example give {examples_str}"""
},
{
     "role": "user",
    "content": f"Analyse the current ticket and provide the stsp by step Reasoning,category and priority{current_ticket}"
}
]

response=llm.invoke(messages)
print(response.content)

