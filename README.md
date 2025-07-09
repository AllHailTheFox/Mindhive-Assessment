# Mindhive-Assessment
This is my chatbot for Mindhive assessment test

# ZUS Coffee Chatbot

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

- ✅ SQL + summarizer works for outlet data extraction
❗However, some LLM hallucinations still occur, especially when summarizing SQL results.
  **Due to limited time for thorough testing and prompt tuning, some outputs may include fabricated details)** (e.g., closing times or outlet names not in the DB).

---

## ✨ Example Prompts

> “What time does Kuala Lumpur outlets open?” Issue

> “what are the names out the outlet are in selangor?”

> “Do you have any BPA free product?”

> “How much is 12.5% of RM37?”

---
## 🛡️ Error Handling & Security Strategy

- Input validation for calculator queries using regex + `eval` in a restricted context
- SQL Database chain wraps queries with error logging and fallback messages
  > “What is 3 divided by coffee?”
- For missing slot inputs (e.g., no city/state), chatbot asks follow-up questions
  >"Bot: Could you please specify the location or outlet name?"
- All major query types (SQL, vector, calculator) are wrapped in try/except
- Malicious inputs (e.g., SQL injection) are not executed — they return a polite error
- >"Bot: Failed to read FAISS"
  
---
#### ✅ Flow Diagram / Screenshot

You can add a flow diagram (if you're able to generate one), or include screenshots of your chatbot in action.

Here's how you might write it in markdown:


## 🔄 Chatbot Architecture & Flow

📤 User Input (via Streamlit Chat UI)  
  ⬇  
📌 Router Logic  
  ├── 🏢 Outlet Query → 🧠 LangChain Text2SQL → 🗃 SQLite DB  
  ├── 🧃 Product Query → 🔍 FAISS Vector Search → 📦 Product Vectorstore  
  ├── ➗ Math Query → 📐 Safe Calculator Logic  
  └── 💬 General Chat → 🧠 LangChain ConversationChain → 🌐 OpenAI GPT-3.5 API


---
## 📘 API Specification (Conceptual)

Although the deployed Streamlit app does not expose public APIs (due to Streamlit Cloud limitations), the chatbot backend is fully designed to support API access using FastAPI.

The FastAPI backend is **fully working and available locally**, provided in a this GitHub repository:

> ✅ **See**: [`Locally with FastAPI`](https://github.com/your-username/Locally-with-FastAPI)  
> 🚀 To test locally, run: `uvicorn zus_api:app --reload`

### Available Endpoints

#### `/outlets` (GET)
- **Description**: Query ZUS Coffee outlets by city or state.
- **Query Parameters**:
  - `city`: Optional string (e.g., `Kuala Lumpur`)
  - `state`: Optional string (e.g., `Selangor`)
- **Returns**: JSON list of matching outlets with full metadata.

#### `/products` (POST)
- **Description**: Query product-related information using vector search (RAG).
- **Request Body**:
  ```json
  {
    "query": "Do you sell matcha drinks?"
  }

---




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
