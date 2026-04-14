import streamlit as st
from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START
from mistralai import Mistral
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from tavily import TavilyClient
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import json
import os
import uuid
import re
import ast

# -------------------- Load Secrets --------------------
MODEL = "mistral-large-2512"
client = Mistral(api_key=st.secrets["MISTRAL_API_KEY"])
tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])
SENDGRID_API_KEY = st.secrets.get("SENDGRID_API_KEY", "")
FROM_EMAIL = st.secrets.get("FROM_EMAIL", "")

# -------------------- TypedDict --------------------
class MessagesState(TypedDict):
    messages: List[BaseMessage]

# -------------------- User Identification --------------------
if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(uuid.uuid4())
user_id = st.session_state["user_id"]
CHATS_FILE = f"multi_chat_history_{user_id}.json"

# -------------------- Multi-chat Memory --------------------
def save_chats(filename, chats):
    data = {}
    for chat_id, chat_data in chats.items():
        messages = chat_data["messages"]
        title = chat_data.get("title", f"Chat {chat_id[:6]}")
        data[chat_id] = {
            "title": title,
            "messages": [{"type": m.type, "content": m.content} for m in messages]
        }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_chats(filename):
    if not os.path.exists(filename):
        return {}
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    chats = {}
    for chat_id, c in data.items():
        messages = []
        for m in c.get("messages", []):
            if m["type"] == "human":
                messages.append(HumanMessage(content=m["content"]))
            elif m["type"] == "ai":
                messages.append(AIMessage(content=m["content"]))
            else:
                messages.append(SystemMessage(content=m["content"]))
        chats[chat_id] = {"messages": messages, "title": c.get("title", f"Chat {chat_id[:6]}")}
    return chats

if "chats" not in st.session_state:
    st.session_state.chats = load_chats(CHATS_FILE)

# Convert old list-based chats to dict
for chat_id, chat in st.session_state.chats.items():
    if isinstance(chat, list):
        st.session_state.chats[chat_id] = {"messages": chat, "title": f"Chat {chat_id[:6]}"}

if "current_chat_id" not in st.session_state:
    if st.session_state.chats:
        st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
    else:
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"messages": [], "title": f"Chat {new_id[:6]}"}
        st.session_state.current_chat_id = new_id

# ============================================================
#                     SENDGRID EMAIL TOOL
# ============================================================
def send_email_tool(to_email: str, subject: str, content: str) -> str:
    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        message = Mail(
            from_email=FROM_EMAIL,
            to_emails=to_email,
            subject=subject,
            html_content=content,
        )
        response = sg.send(message)
        return f"Email sent successfully (status {response.status_code})."
    except Exception as e:
        return f"Error sending email: {e}"

# ============================================================
#        NODE 0: Decide if user wants to send an email
# ============================================================
def decide_email_or_search(state: MessagesState) -> MessagesState:
    last_user = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    msgs = [
        {"role": "system", "content":
         """Classify the user's intent. Respond ONLY with:
         - SEND_EMAIL
         - SEARCH_REQUIRED
         - NO_SEARCH
         """},
        {"role": "user", "content": last_user}
    ]
    resp = client.chat.complete(model=MODEL, messages=msgs, temperature=0.0)
    decision = resp.choices[0].message.content.strip().upper()
    if decision not in ["SEND_EMAIL", "SEARCH_REQUIRED", "NO_SEARCH"]:
        decision = "NO_SEARCH"
    state["messages"].append(SystemMessage(content=decision))
    return state

# ============================================================
#        NODE 1: Extract email parameters (robust)
# ============================================================
def extract_email_parameters(state: MessagesState) -> MessagesState:
    last_user = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    
    msgs = [
        {"role": "system", "content":
         """You are an AI assistant. Extract email fields STRICTLY in valid JSON.
Respond ONLY with JSON, no explanations, no backticks.
Format:
{
  "to": "recipient@example.com",
  "subject": "Subject here",
  "content": "Body content here"
}
Ensure:
- Use double quotes
- Escape any quotes inside content
- Do not include extra text
"""}, 
        {"role": "user", "content": last_user}
    ]
    
    resp = client.chat.complete(model=MODEL, messages=msgs, temperature=0.0)
    llm_output = resp.choices[0].message.content.strip()

    match = re.search(r"\{.*\}", llm_output, re.DOTALL)
    if not match:
        state["messages"].append(SystemMessage(content="SEND_EMAIL_ERROR: Could not extract JSON from LLM."))
        return state

    json_text = match.group().strip()

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(json_text)
        except Exception as e:
            state["messages"].append(SystemMessage(content=f"SEND_EMAIL_ERROR: JSON parse error: {e}"))
            return state

    to_email = data.get("to")
    subject = data.get("subject", "No Subject")
    content = data.get("content", "")

    if not to_email or "@" not in to_email:
        state["messages"].append(SystemMessage(content="SEND_EMAIL_ERROR: Missing or invalid 'to' email address."))
        return state

    result_text = send_email_tool(to_email, subject, content)
    state["messages"].append(SystemMessage(content=result_text))
    return state

# ============================================================
#        SEARCH + LLM Nodes
# ============================================================
def tavily_search_node(state: MessagesState) -> MessagesState:
    question = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    result = tavily.search(query=question, max_results=3)
    clean_summary = "\n".join([r["content"] for r in result["results"]])
    state["messages"].append(SystemMessage(content=f"SEARCH_RESULT: {clean_summary}"))
    return state

def llm_call(state: MessagesState) -> MessagesState:
    clean_msgs = []
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            clean_msgs.append({"role": "user", "content": m.content})
        if m.content.startswith("SEARCH_RESULT:"):
            clean_msgs.append({
                "role": "system",
                "content": "Here is verified search info:\n" + m.content.replace("SEARCH_RESULT:", "").strip()
            })
    clean_msgs.insert(0, {"role": "system", "content": "You are a precise AI assistant. Do NOT hallucinate. If unsure, respond: 'I am not sure.'"})
    resp = client.chat.complete(model=MODEL, messages=clean_msgs, temperature=0.0, max_tokens=2048)
    state["messages"].append(AIMessage(content=resp.choices[0].message.content))
    return state

# ============================================================
#                     BUILD GRAPH
# ============================================================
builder = StateGraph(MessagesState)
builder.add_node("decide", decide_email_or_search)
builder.add_node("extract_email", extract_email_parameters)
builder.add_node("search_node", tavily_search_node)
builder.add_node("llm_call", llm_call)

builder.add_edge(START, "decide")
builder.add_conditional_edges(
    "decide",
    lambda s: s["messages"][-1].content,
    {
        "SEND_EMAIL": "extract_email",
        "SEARCH_REQUIRED": "search_node",
        "NO_SEARCH": "llm_call",
    },
)
builder.add_edge("search_node", "llm_call")
agent = builder.compile()

# ============================================================
#                     STREAMLIT UI
# ============================================================
st.set_page_config(page_title="BoudyPilot 1.3", page_icon="🤖")
st.title("🤖 BoudyPilot 1.3 — Multi-Chat AI Agent")

# -------------------- Sidebar --------------------
st.sidebar.title("Your Chats")

current_chat = st.session_state.chats[st.session_state.current_chat_id]
new_title = st.sidebar.text_input(
    "Rename Current Chat",
    value=current_chat.get("title", f"Chat {st.session_state.current_chat_id[:6]}")
)
if new_title != current_chat.get("title"):
    st.session_state.chats[st.session_state.current_chat_id]["title"] = new_title
    save_chats(CHATS_FILE, st.session_state.chats)

if st.sidebar.button("➕ New Chat"):
    new_id = str(uuid.uuid4())
    st.session_state.chats[new_id] = {"messages": [], "title": f"Chat {new_id[:6]}"}
    st.session_state.current_chat_id = new_id
    st.rerun()

for chat_id, chat_data in st.session_state.chats.items():
    label = chat_data.get("title", f"Chat {chat_id[:6]}")
    if st.sidebar.button(label):
        st.session_state.current_chat_id = chat_id
        st.rerun()

if st.sidebar.button("🗑️ Clear Current Chat"):
    st.session_state.chats[st.session_state.current_chat_id]["messages"] = []
    save_chats(CHATS_FILE, st.session_state.chats)
    st.rerun()

current_messages = st.session_state.chats[st.session_state.current_chat_id]["messages"]
if current_messages:
    json_data = json.dumps(
        [{"type": m.type, "content": m.content} for m in current_messages],
        ensure_ascii=False,
        indent=2,
    )
    st.sidebar.download_button(
        label="⬇️ Download Current Chat (JSON)",
        data=json_data,
        file_name=f"chat_{st.session_state.current_chat_id}.json",
        mime="application/json",
    )

# -------------------- Chat Container --------------------
chat_container = st.container()
with chat_container:
    for msg in current_messages:
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.write(msg.content)

# -------------------- User Input --------------------
user_input = st.chat_input("Ask me anything...")

if user_input:
    # 1️⃣ Show user message instantly
    user_msg = HumanMessage(content=user_input)
    st.session_state.chats[st.session_state.current_chat_id]["messages"].append(user_msg)
    with chat_container:
        with st.chat_message("user"):
            st.write(user_input)

    # 2️⃣ Update agent memory and invoke agent
    st.session_state.chat_memory = {"messages": st.session_state.chats[st.session_state.current_chat_id]["messages"]}
    st.session_state.chat_memory = agent.invoke(st.session_state.chat_memory)

    # 3️⃣ Show new assistant messages
    new_messages = st.session_state.chat_memory["messages"][len(st.session_state.chats[st.session_state.current_chat_id]["messages"])-1:]
    for msg in new_messages:
        if msg.type != "human":
            with chat_container:
                with st.chat_message("assistant"):
                    st.write(msg.content)

    # 4️⃣ Save updated chat
    st.session_state.chats[st.session_state.current_chat_id]["messages"] = st.session_state.chat_memory["messages"]
    save_chats(CHATS_FILE, st.session_state.chats)
