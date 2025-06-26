import streamlit as st
import json
import re
import pandas as pd
from datetime import datetime
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Page Setup ---
st.set_page_config(page_title="AI Chat Agent for Calibr LMS")
st.title("🤖 AI Chat Agent for Calibr LMS")

# --- Sidebar Role Selector & Reset ---
st.sidebar.title("🎭 Select Role")

if "messages_by_role" not in st.session_state:
    st.session_state.messages_by_role = {
        "Learner": [],
        "Author": [],
        "Admin": []
    }

if "prev_role" not in st.session_state:
    st.session_state.prev_role = "Learner"
if "clear_msg" not in st.session_state:
    st.session_state.clear_msg = None

selected_role = st.sidebar.selectbox("Role", ["Learner", "Author", "Admin"], key="current_role")

if selected_role != st.session_state.prev_role:
    st.session_state.messages_by_role[selected_role] = []
    st.session_state.clear_msg = f" Switched from **{st.session_state.prev_role}** to **{selected_role}**."
    st.session_state.prev_role = selected_role
    st.rerun()

if st.sidebar.button("🔄 Clear Chat"):
    st.session_state.messages_by_role[selected_role] = []
    st.session_state.clear_msg = "🧹 Chat manually cleared."
    st.rerun()

role = selected_role
messages = st.session_state.messages_by_role[role]

# --- Load FAISS Vectorstore ---
@st.cache_resource
def load_vectorstore():
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    return FAISS.load_local("faiss_index", embed_model, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()

# --- Retrieve Context ---
def retrieve_context(question, k=3):
    docs = vectorstore.similarity_search(question, k=k)
    return [doc.page_content for doc in docs]

# --- Admin CSV Analytics ---
def admin_analytics_csv(question):
    if "inactive users" in question.lower():
        try:
            df = pd.read_csv("data/users.csv", parse_dates=["last_login"])
            df = df.sort_values(by="last_login").head(5)
            return {
                "title": "Top 5 Inactive Users This Week",
                "data": [{"label": row["user"], "value": row["last_login"].strftime("%Y-%m-%d")} for _, row in df.iterrows()]
            }
        except Exception as e:
            return {"error": f"Could not read users.csv: {e}"}

    elif "completion rates" in question.lower():
        try:
            df = pd.read_csv("data/completions.csv")
            df = df.sort_values(by="completion_rate").head(5)
            return {
                "title": "Courses with Lowest Completion Rates",
                "data": [{"label": row["title"], "value": f"{row['completion_rate']}%"} for _, row in df.iterrows()]
            }
        except Exception as e:
            return {"error": f"Could not read completions.csv: {e}"}

    return None

# --- Prompt Builder ---
def build_prompt(role, question):
    context = retrieve_context(question)

    if role == "Learner":
        if "flashcard" in question.lower():
            instruction = """You are a flashcard generator.
Create exactly 3 flashcards using this format:

<flashcards>
Flashcard 1:
Q: ...
A: ...

Flashcard 2:
Q: ...
A: ...

Flashcard 3:
Q: ...
A: ...
</flashcards>

No greetings or extra text. Just output the flashcards in the exact format."""
        else:
            instruction = """Explain the topic in 1–2 short paragraphs. No greetings, no list, just clear explanation."""

        return f"""
You are a helpful assistant in an LMS.

Role: {role}
Instruction: {instruction}

Context:
{chr(10).join(context)}

Question: {question}
"""

    elif role == "Author":
        instruction = """You are an LMS assistant. Your task is to generate exactly 3 multiple choice questions (MCQs) on the given topic.

🧩 Each question must follow *exactly* this strict format:
1) [Question text]  
a) [Option A]  
b) [Option B]  
c) [Option C]  
d) [Option D]  
Answer: a) [Correct Option]

⚠️ STRICT RULES:
- No explanations, notes, extra comments, or markdown.
- DO NOT include anything other than the 3 MCQs.
- Each MCQ must start with 1), 2), 3) respectively and be on separate lines.
- Output must be plain text in the above format. No markdown or symbols like ❓ or ✅."""

        return f"""
Role: {role}
Instruction: {instruction}

Context:
{chr(10).join(context)}

Question: {question}
"""

    elif role == "Admin":
        return f"""
You are an analytics assistant for an LMS. You must return only JSON in the format:
{{
  "title": "...",
  "data": [{{"label": "...", "value": ...}}, ...]
}}

Context:
{chr(10).join(context)}

Question: {question}
"""

# --- Flashcard Renderer ---
def render_flashcards(raw):
    match = re.search(r"<flashcards>(.*?)</flashcards>", raw, re.DOTALL)
    content = match.group(1).strip() if match else raw.strip()

    cards = re.findall(r"Flashcard\s*\d+:\s*Q:\s*(.*?)\s*A:\s*(.*?)(?=\n\s*Flashcard|\Z)", content, re.DOTALL)

    if not cards:
        st.warning("⚠️ Could not parse flashcards.")
        st.code(raw)
        return

    for i, (q, a) in enumerate(cards, 1):
        st.markdown(f"""
        <div style='background-color:#f4f9ff;padding:16px;border-left:6px solid #1c7ed6;
                    border-radius:8px;margin:12px 0;box-shadow:0 1px 4px rgba(0,0,0,0.06);'>
            <strong>🧠 Flashcard {i}</strong><br><br>
            <strong>Q:</strong> {q.strip()}<br>
            <strong>A:</strong> {a.strip()}
        </div>
        """, unsafe_allow_html=True)

# --- MCQ Renderer ---
def render_mcqs(raw):
    pattern = r"(\d+\))\s+(.*?)\n\s*a\)\s+(.*?)\n\s*b\)\s+(.*?)\n\s*c\)\s+(.*?)\n\s*d\)\s+(.*?)\n\s*Answer:\s*([a-d]\))"
    mcqs = re.findall(pattern, raw, re.DOTALL)

    if not mcqs:
        st.warning("⚠️ Could not parse MCQs.")
        st.code(raw)
        return

    for q in mcqs:
        number, question, a, b, c, d, answer = q
        st.markdown(f"**❓ {number} {question}**")
        st.markdown(f"a) {a}")
        st.markdown(f"b) {b}")
        st.markdown(f"c) {c}")
        st.markdown(f"d) {d}")
        st.markdown(f"✅ **Answer: {answer}**")
        st.markdown("---")

# --- Analytics Renderer ---
def render_analytics(raw):
    try:
        data = json.loads(raw)
        if "error" in data:
            st.warning(data["error"])
            return
        st.subheader(data.get("title", "📊 Analytics Result"))
        for entry in data.get("data", []):
            st.markdown(f"**{entry.get('label', 'Label')}**: {entry.get('value', 'N/A')}")
    except Exception as e:
        st.warning("⚠️ Could not parse analytics JSON.")
        st.code(raw)

# --- LLM Setup ---
llm = Ollama(model="phi")

# --- Show Previous Messages ---
if st.session_state.clear_msg:
    st.info(st.session_state.clear_msg)
    st.session_state.clear_msg = None

for msg in messages:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "flashcards":
            render_flashcards(msg["content"])
        elif msg.get("type") == "mcq":
            render_mcqs(msg["content"])
        elif msg.get("type") == "analytics":
            render_analytics(msg["content"])
        else:
            st.markdown(msg["content"])

# --- Chat Input Handler ---
if prompt := st.chat_input("💬 Ask a question..."):
    st.chat_message("user").markdown(prompt)
    messages.append({"role": "user", "content": prompt})
    st.session_state.messages_by_role[role] = messages

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            full_prompt = build_prompt(role, prompt)
            response = llm.invoke(full_prompt).strip()

            if role == "Learner" and "flashcard" in prompt.lower():
                render_flashcards(response)
                messages.append({"role": "assistant", "type": "flashcards", "content": response})
            elif role == "Author":
                render_mcqs(response)
                messages.append({"role": "assistant", "type": "mcq", "content": response})
            elif role == "Admin":
                csv_response = admin_analytics_csv(prompt)
                if csv_response:
                    render_analytics(json.dumps(csv_response))
                    messages.append({"role": "assistant", "type": "analytics", "content": json.dumps(csv_response)})
                else:
                    render_analytics(response)
                    messages.append({"role": "assistant", "type": "analytics", "content": response})
            else:
                st.markdown(response)
                messages.append({"role": "assistant", "content": response})

            st.session_state.messages_by_role[role] = messages
