# 🤖 AI Chat Bot

A role-based AI-powered assistant prototype, built with:

- 💬 LangChain + Ollama (`phi` model)
- 🔍 FAISS Vector Store + MiniLM Embeddings
- 🖥️ Streamlit UI
- 📁 Mock Data (Lessons, Users, Completions)

## 🧠 Roles Supported

- 🎓 **Learner**: Content summaries, flashcards
- ✍️ **Author**: Generate MCQs
- 📊 **Admin**: Analytics from CSV

## 🛠️ Setup

**1. Clone this repo**

   git clone https://github.com/PratikshaPhukon/ai-chat-bot.git
   
  >>cd ai-chat-agent
   
**2. Create virtual environment**

>>python -m venv venv

>>    source venv/bin/activate  # or venv\Scripts\activate on Windows


**3. Install dependencies**

>>pip install -r requirements.txt

**4. Start Ollama locally**

Install from https://ollama.com

>>ollama run phi


**5. Build the vectorstore**

>>python setup_vectorstore.py


**6. Launch the app**

>>streamlit run app.py





**📷 Screenshots**

![image](https://github.com/user-attachments/assets/e43e3ad7-3214-409b-b130-d6f16e40bdc5)
![image](https://github.com/user-attachments/assets/ea4e4232-7c59-4983-9e48-358c2686a459)









