# Mindhive-Assessment
This is my chatbot for Mindhive assessment test

# ☕ ZUS Coffee Chatbot

An intelligent, friendly chatbot that helps users interact with ZUS Coffee's outlet and product information — powered by **LangChain**, **FAISS**, **OpenAI**, and **Streamlit**.

🚀 **[Try it live here](https://mindhive-assessment-csznxabm3ts9kyj5rkynv7.streamlit.app/)**

---

## 📌 Features

- 🔍 Ask natural questions about ZUS Coffee outlets (e.g. opening hours, locations)
- 🧃 Search product info using vector embeddings (semantic search via FAISS)
- 🧮 Ask math-related questions (basic calculator)
- 💬 Conversational memory for a natural flow
- 📋 Logging shown in-chat for transparency and debugging

---

## 🧠 Powered By

- 💬 **LangChain** – LLM orchestration & memory
- 🗄️ **SQLite** – Fast and simple outlet database
- 📦 **FAISS + HuggingFace Embeddings** – Product search vectorstore
- 🌐 **OpenAI GPT-3.5 Turbo** – Language understanding & generation
- 🎛 **Streamlit** – Interactive UI and frontend

---

## 🏗 Architecture Overview

| Layer                  | Tool / Library                     | Purpose                                      |
|------------------------|------------------------------------|----------------------------------------------|
| **Frontend UI**        | `Streamlit`                        | Interactive web interface                    |
| **LLM Integration**    | `ChatOpenAI` (via LangChain)       | Handles natural chat & responses             |
| **Outlet Data**        | `SQLDatabaseChain` (LangChain + SQLite) | SQL-powered outlet lookup               |
| **Product Search**     | `FAISS + HuggingFace Embeddings`   | Semantic vector search over product catalog  |
| **Math Support**       | Custom-safe math parser            | Calculates expressions securely              |
| **Session Memory**     | `ConversationBufferMemory`         | Maintains context for back-and-forth chat    |

---

## ⚖️ Key Trade-offs & Limitations

- ✅ **Session Memory** is implemented using `ConversationBufferMemory` to allow conversational context across turns.
  
  ❗However, due to either:
  - the LangChain memory object not being properly updated inside the `conversation.predict()`, or  
  - `Streamlit`'s stateless rerun model interfering with memory persistence,

  **the memory doesn't persist as expected.** Only the visible chat history is shown using `st.session_state`.

- ✅ Vector search via FAISS is fast and flexible  
  ❗But the product database is **read-only** unless you manually rebuild the vectorstore.

- ✅ Uses simple SQLite for portability  
  ❗Not ideal for scalability or multi-user deployments
  
- ✅ FastAPI backend is modular and usable in local development  
  ❗**Streamlit Cloud does not support running FastAPI servers**, so the `/outlets` and `/products` REST API endpoints are not active in the public deployment.

---

## ✨ Example Prompts

> “What time does ZUS SS15 open?”

> “List all outlets in Kuala Lumpur.”

> “What drinks contain matcha?”

> “How much is 12.5% of RM37?”

---

## ⚙️ Local Development (Optional)

While not required, developers can run this app locally if desired.

### Requirements

- Python 3.9+
- `zus_outlets.db` (SQLite DB file)
- `faiss_zus_products/` (Vectorstore folder)

### Setup

```bash
git clone https://github.com/yourusername/zus-chatbot.git
cd zus-chatbot

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
