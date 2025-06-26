# 🤖 AI Chat Agent

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

   git clone https://github.com/pratikxaphukon/ai-chat-agent.git
   
  >>cd ai-chat-agent
   
**2. Create virtual environment**

>>python -m venv venv
>>source venv/bin/activate  # or venv\Scripts\activate on Windows


**3. Install dependencies**

>>pip install -r requirements.txt

**4. Start Ollama locally**

Install from https://ollama.com

>>ollama run phi


**5. Build the vectorstore**

>>python setup_vectorstore.py


**6. Launch the app**

>>streamlit run app.py


**📂 Files**

ai-chat-agent/
├── app.py
├── setup_vectorstore.py
├── requirements.txt
├── data/
│   ├── lessons.json
│   ├── users.csv
│   └── completions.csv
└── faiss_index/



**📷 Screenshots**

![image](https://github.com/user-attachments/assets/e43e3ad7-3214-409b-b130-d6f16e40bdc5)
![image](https://github.com/user-attachments/assets/ea4e4232-7c59-4983-9e48-358c2686a459)
![image](https://github.com/user-attachments/assets/65924edb-4248-4a01-8f86-f39c7d1a8460)








