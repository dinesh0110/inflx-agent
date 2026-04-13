\# 🔥 Inflx AI Agent – Social-to-Lead Workflow for AutoStream

Inflx is a \*\*Conversational AI Sales Agent\*\* built for a fictional SaaS product \*\*AutoStream\*\*, which provides automated video editing tools for content creators.

This project demonstrates how an AI agent can:

\* Understand user intent

\* Answer product queries using RAG

\* Identify high-intent users

\* Convert conversations into \*\*qualified leads\*\*

The application is built using \*\*Streamlit + LangGraph + Gemini AI\*\*, and simulates a real-world \*\*AI-powered sales funnel\*\*.

\---

\# 📌 Features

\### 🧠 Intent Classification

\* Classifies user messages into:

\* Greeting

\* Inquiry

\* High-intent (ready to sign up)

\* Implemented using an LLM-based classifier node

\---

\### 📚 RAG-Powered Responses

\* Uses a local \`knowledge\_base.json\`

\* Answers grounded in:

\* Pricing

\* Features

\* Policies

\* Prevents hallucinations

\---

\### 🎯 Lead Capture Workflow

\* Step-by-step data collection:

\* Name

\* Email

\* Creator Platform

\* Triggers tool \*\*only after all fields are collected\*\*

\---

\### 🔄 Stateful Conversations

\* Maintains context across multiple turns

\* Built using LangGraph \`StateGraph\`

\---

\### 💬 Streamlit Chat UI

\* Multiple chat sessions

\* Sidebar leads dashboard

\* Download leads as JSON

\---

\# 🛠 Tech Stack

\### Language

\* Python 3.9+

\### Core Libraries

\* \*\*Streamlit\*\* – UI & chat interface

\* \*\*LangChain & LangGraph\*\* – Agent orchestration

\* \*\*Google Generative AI (Gemini)\*\* – LLM + embeddings

\* \*\*FAISS\*\* – Vector store for RAG

\### Models Used

\* \`gemini-flash-lite-latest\` → chat model

\* \`gemini-embedding-001\` → embeddings

\---

\# 📂 Project Structure

\`\`\`

.

├── app.py # Main Streamlit app (agent + UI)

├── knowledge\_base.json # RAG data (plans, pricing, policies)

├── leads.json # Captured leads storage

├── requirements.txt # Dependencies

└── README.md

\`\`\`

\---

\# ⚙️ Setup & Installation

\## 1. Prerequisites

\* Python 3.9+

\* Gemini API Key

\---

\## 2. Clone Repository

\`\`\`bash

git clone

cd

\`\`\`

\---

\## 3. Create Virtual Environment (Recommended)

\`\`\`bash

python -m venv .venv

\# Windows

.venv\\Scripts\\activate

\# macOS/Linux

source .venv/bin/activate

\`\`\`

\---

\## 4. Install Dependencies

\`\`\`bash

pip install -r requirements.txt

\`\`\`

\---

\## 5. Environment Variables

Create a \`.env\` file:

\`\`\`

GEMINI\_API\_KEY=your\_api\_key\_here

\`\`\`

\---

\## 6. Knowledge Base

Ensure \`knowledge\_base.json\` includes:

\* Plans (name, price, features)

\* Policies (refunds, support)

\---

\# ▶️ How to Run

\`\`\`bash

streamlit run app.py

\`\`\`

Open:

\`\`\`

http://localhost:8501

\`\`\`

\---

\# 🧠 Architecture Overview

The agent is built using \*\*LangGraph State Machine\*\*

\### Agent State Includes:

\* \`messages\` → conversation history

\* \`intent\` → classified intent

\* \`lead\_stage\` → current funnel stage

\* \`name\`, \`email\`, \`platform\` → lead data

\* \`context\` → RAG output

\* \`reply\` → generated response

\---

\## 🔄 Graph Flow

\`\`\`

classify → retrieve → generate → END

\`\`\`

\### Nodes:

\#### 1. Intent Classifier

\* Uses LLM

\* Outputs: greeting / inquiry / high\_intent

\#### 2. Retriever (RAG)

\* FAISS similarity search

\* Returns relevant context

\#### 3. Generator

\* Combines:

\* Context

\* Lead stage

\* Produces final response

\---

\# 📚 RAG Pipeline

\### Build Phase

\* Load JSON

\* Split text

\* Embed using Gemini

\* Store in FAISS

\### Query Phase

\* Retrieve top-k chunks

\* Inject into prompt

\* Generate grounded response

\---

\# 🎯 Lead Capture Logic

\### Workflow

1\. Detect high intent

2\. Ask name

3\. Ask email (validate \`@\`)

4\. Ask platform

5\. Capture lead

\---

\### Tool Function

\`\`\`python

def mock\_lead\_capture(name, email, platform):

\`\`\`

\### Behavior:

\* Stores in session

\* Saves to \`leads.json\`

\* Displays success message

\---

\# 💻 Streamlit UI

\## Sidebar

\* New Chat

\* Chat switching

\* Clear all chats

\### 📊 Leads Dashboard

\* Total leads count

\* Lead list

\* Download JSON

\---

\## Main Chat Area

\* Chat messages (user + AI)

\* Welcome screen

\* Input box

\* Typing indicator

\---

\# Leads Storage

Leads are stored in:

\`\`\`

leads.json

\`\`\`

Example:

\`\`\`json

\[

{

"name": "John",

"email": "john@gmail.com",

"platform": "YouTube"

}

\]

\`\`\`

\---

\# WhatsApp Integration (Concept)

To deploy on WhatsApp:

\### Required:

\* Twilio / Meta WhatsApp API

\* Backend (Flask / FastAPI)

\### Flow:

1\. Receive message via webhook

2\. Load user state

3\. Call \`graph.invoke()\`

4\. Send response back

\---

\# 🚀 Future Improvements

\* Database (PostgreSQL / MongoDB)

\* Google Sheets integration

\* Streaming responses

\* Analytics dashboard

\* WhatsApp deployment

\---
