import os
import json
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# ───────────────────────────────
# ENV
# ───────────────────────────────
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# ───────────────────────────────
# STREAMLIT CONFIG
# ───────────────────────────────
st.set_page_config(page_title="Inflx Agent", layout="wide")

# ───────────────────────────────
# LOAD KNOWLEDGE BASE
# ───────────────────────────────
@st.cache_resource
def build_vectorstore():
    with open("knowledge_base.json") as f:
        kb = json.load(f)

    texts = []

    for plan in kb["plans"]:
        texts.append(f"{plan['name']} Plan: {plan['price']} - {', '.join(plan['features'])}")

    for policy in kb["policies"]:
        texts.append(policy)

    splitter = RecursiveCharacterTextSplitter(chunk_size=200)
    docs = splitter.create_documents(texts)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=API_KEY
    )

    return FAISS.from_documents(docs, embeddings)

VECTOR_STORE = build_vectorstore()

# ───────────────────────────────
# LLM (Cached)
# ───────────────────────────────
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-flash-lite-latest",
        temperature=0.2,
        google_api_key=API_KEY,
        streaming=True
    )

llm = get_llm()

# ───────────────────────────────
# STATE
# ───────────────────────────────
class AgentState(TypedDict):
    messages: list
    intent: str
    lead_stage: str
    name: str
    email: str
    platform: str
    context: str
    reply: str
    lead_captured: bool

# ───────────────────────────────
# TOOL
# ───────────────────────────────
def mock_lead_capture(name, email, platform):
    print(f"Lead captured successfully: {name}, {email}, {platform}")

# ───────────────────────────────
# NODE 1: INTENT
# ───────────────────────────────
def classify(state: AgentState):
    last = state["messages"][-1].content

    prompt = f"""
Classify intent into one word:
greeting, inquiry, high_intent

Message: {last}
"""

    res = llm.invoke([HumanMessage(content=prompt)]).content
    if isinstance(res, list):
        res = ' '.join([str(part) if isinstance(part, str) else str(part.get('text', part)) for part in res])
    res = res.strip().lower()

    return {**state, "intent": res}

# ───────────────────────────────
# NODE 2: RAG
# ───────────────────────────────
def retrieve(state: AgentState):
    query = state["messages"][-1].content
    docs = VECTOR_STORE.similarity_search(query, k=4)
    context = "\n".join([d.page_content for d in docs])
    return {**state, "context": context}

# ───────────────────────────────
# NODE 3: RESPONSE
# ───────────────────────────────
def generate(state: AgentState):
    stage = state["lead_stage"]
    context = state.get("context", "")

    last = state["messages"][-1].content

    system = f"""
You are Inflx AI sales agent.

Context:
{context}

Lead Stage: {stage}

Rules:
- Answer using context
- Be short
"""

    reply = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=last)
    ]).content
    if isinstance(reply, list):
        reply = ' '.join([str(part) if isinstance(part, str) else str(part.get('text', part)) for part in reply])
    reply = reply.strip()

    # ───── Lead Logic ─────
    if state["intent"] == "high_intent" and stage == "idle":
        reply = "Great! Let's get you started. What's your name?"
        state["lead_stage"] = "name"

    elif stage == "name":
        state["name"] = last
        state["lead_stage"] = "email"
        reply = f"Nice to meet you {last}! What's your email?"

    elif stage == "email":
        if "@" not in last:
            reply = "❌ Please enter a valid email address (must contain @)"
        else:
            state["email"] = last
            state["lead_stage"] = "platform"
            reply = "Which platform do you create content on?"

    elif stage == "platform":
        state["platform"] = last
        state["lead_stage"] = "done"

        mock_lead_capture(state["name"], state["email"], state["platform"])

        reply = f"🎉 You're all set {state['name']}! We'll contact you soon."
        state["lead_captured"] = True

    return {
        **state,
        "reply": reply,
        "messages": state["messages"] + [AIMessage(content=reply)]
    }

# ───────────────────────────────
# GRAPH
# ───────────────────────────────
builder = StateGraph(AgentState)

builder.add_node("classify", classify)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)

builder.set_entry_point("classify")

builder.add_edge("classify", "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

graph = builder.compile()

# ───────────────────────────────
# SESSION
# ───────────────────────────────
if "conversations" not in st.session_state:
    st.session_state.conversations = {
        "Chat 1": {
            "messages": [],
            "intent": "",
            "lead_stage": "idle",
            "name": "",
            "email": "",
            "platform": "",
            "context": "",
            "reply": "",
            "lead_captured": False
        }
    }
    st.session_state.current_chat = "Chat 1"
    st.session_state.chat_counter = 1

# ───────────────────────────────
# SIDEBAR
# ───────────────────────────────
with st.sidebar:
    st.title("💬 Chats")
    
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.chat_counter += 1
        new_chat_name = f"Chat {st.session_state.chat_counter}"
        st.session_state.conversations[new_chat_name] = {
            "messages": [],
            "intent": "",
            "lead_stage": "idle",
            "name": "",
            "email": "",
            "platform": "",
            "context": "",
            "reply": "",
            "lead_captured": False
        }
        st.session_state.current_chat = new_chat_name
        st.rerun()
    
    st.divider()
    
    # Chat history
    for chat_name in st.session_state.conversations:
        if st.button(chat_name, use_container_width=True, 
                    key=f"chat_{chat_name}",
                    type="primary" if chat_name == st.session_state.current_chat else "secondary"):
            st.session_state.current_chat = chat_name
            st.rerun()
    
    st.divider()
    
    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.conversations = {
            "Chat 1": {
                "messages": [],
                "intent": "",
                "lead_stage": "idle",
                "name": "",
                "email": "",
                "platform": "",
                "context": "",
                "reply": "",
                "lead_captured": False
            }
        }
        st.session_state.current_chat = "Chat 1"
        st.session_state.chat_counter = 1
        st.rerun()

# Get current conversation
state = st.session_state.conversations[st.session_state.current_chat]

# ───────────────────────────────
# CUSTOM CSS
# ───────────────────────────────
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 12px;
        padding: 12px;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    [data-testid="stAlert"] {
        border-radius: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────
# UI
# ───────────────────────────────
st.title("🔥 Inflx AI Agent")

# Display current chat name
st.caption(f"📌 {st.session_state.current_chat}")

# Welcome message
if len(state["messages"]) == 0:
    st.info("👋 **Welcome to Inflx AI Agent!** Ask me about our plans, features, or policies. I'm here to help!")

# Display messages
for msg in state["messages"]:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

user_input = st.chat_input("Message Inflx...")

if user_input:
    # Add user message
    state["messages"].append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)
    
    # Get response
    with st.spinner("✍️ Agent is typing..."):
        new_state = graph.invoke(state)
        state.update(new_state)
    
    # Display assistant response
    st.chat_message("assistant").write(state["reply"])
    
    # Update session
    st.session_state.conversations[st.session_state.current_chat] = state