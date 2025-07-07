import streamlit as st
import os
import re
import ast
import time
import math
import logging
import warnings
from dotenv import load_dotenv

# LangChain Core Modules
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate as CorePromptTemplate

# LangChain LLM & Model Integrations
#from langchain_community.chat_models import ChatOllama
from langchain.chat_models import ChatOpenAI


# LangChain SQL & Database Tools
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from sqlalchemy import text

# LangChain Vector Store & Embeddings
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
warnings.filterwarnings("ignore", category=DeprecationWarning)

last_outlet = {"name": None}

# === Helper Functions (Unchanged) ===
def is_outlet_query(user_input: str) -> bool:
    return any(kw in user_input.lower() for kw in ["outlet", "location", "shop", "address", "branch", "open", "opening", "close", "closing", "time", "city", "state", "days", "phone", "contact", "number"])

def get_outlet_keywords(db) -> list:
    try:
        with db._engine.connect() as conn:
            result = conn.execute(text("SELECT DISTINCT city FROM outlets")).fetchall()
            cities = [row[0].lower() for row in result]
            result = conn.execute(text("SELECT DISTINCT state FROM outlets")).fetchall()
            states = [row[0].lower() for row in result]
            result = conn.execute(text("SELECT DISTINCT shop_name FROM outlets")).fetchall()
            names = [row[0].lower() for row in result]
            return set(cities + states + names)
    except Exception as e:
        logging.error("Failed to load outlet keywords: %s", e)
        return set()

def load_faiss_vectorstore(folder_path: str) -> FAISS:
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(folder_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
        #logging.info("FAISS vectorstore loaded successfully.")
        return vectorstore
    except Exception as e:
        logging.error("Failed to load FAISS vectorstore: %s", e)
        return None

def is_product_query(text: str) -> bool:
    return any(kw in text.lower() for kw in [
        "product", "products", "items", "catalog", "what do you have", "available", 
        "how many", "price", "cost", "how much", "menu", "drink", "coffee", 
        "matcha", "chocolate", "beverage", "flavor", "sell", "buy"
    ])

def detect_missing_info(user_input: str, history: str, keywords: set) -> str:
    combined = (history + "\n" + user_input).lower()
    if re.search(r"(how many|which|list all|what.*state|what.*city|how many.*state|states.*have)", combined):
        return ""
    if not any(keyword in combined for keyword in keywords):
        return "location"
    return ""

def is_math_query(text: str) -> bool:
    return any(op in text for op in ["+", "-", "*", "/", "^", "square root", "cube root", "power", "raised to", "mod", "modulus", "remainder", "multiply", "divide", "quotient", "add", "sum", "plus", "subtract", "minus", "difference", "average", "mean", "median", "mode", "percentage", "percent of", "calculate", "how much is", "what is", "total of"])

def safe_calculate(query: str) -> str:
    try:
        query = query.lower().replace("square root of", "math.sqrt").replace("^", "**")
        expressions = re.findall(r"(math\.sqrt\(\d+\)|\d+(\.\d+)?\s*[\+\-\*/]\s*\d+(\.\d+)?)", query)
        if not expressions:
            return "Sorry, I couldn't find any valid math expression."
        results = []
        for expr_tuple in expressions:
            expr = expr_tuple[0]
            result = eval(expr, {"math": math, "__builtins__": {}})
            results.append(f"{expr.strip()} = {result}")
        return "\n".join(results)
    except Exception as e:
        logging.error("Calculator error: %s", e)
        return "Sorry, I couldn't calculate that."

# === Initialization ===
@st.cache_resource
def initialize_chatbot():
    load_dotenv(dotenv_path=".env")
    # Make sure your API key is set
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("Missing OPENAI_API_KEY in environment variables.")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    #llm = ChatOllama(model="llama3.2:3b",temperature=0.3) #This worked the best in this usecase i used 0.3 so it doesn't hallucinate badly
    #llm = ChatOllama(model="deepseek-r1:1.5b")
    #llm = ChatOllama(model="deepseek-r1:latest")

    db = SQLDatabase.from_uri("sqlite:///zus_outlets.db")
    db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=False, return_intermediate_steps=True)
    keywords = get_outlet_keywords(db)
    vectorstore = load_faiss_vectorstore("faiss_zus_products")

    summarizer_prompt = CorePromptTemplate.from_template("""
You are a helpful assistant for ZUS Coffee. A user asked:

{question}

Here is the raw result from the database:
{data}

If any data is missing, clearly state that. Otherwise, summarize the outlet names and their closing times accurately.
""")
    summarizer_chain = summarizer_prompt | llm | StrOutputParser()
    memory = ConversationBufferMemory()
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template="""
You are a helpful chatbot for ZUS Coffee. If the user asks something general, try to be helpful and polite.

Conversation history:
{history}
User: {input}
Bot:"""
    )
    conversation = ConversationChain(llm=llm, memory=memory, prompt=prompt)
    return llm, db_chain, vectorstore, keywords, summarizer_chain, memory, conversation

llm, db_chain, vectorstore, keywords, summarizer_chain, memory, conversation = initialize_chatbot()

# === Streamlit App ===
st.set_page_config(page_title="ZUS Chatbot", page_icon="☕")

# Setup a log display in Streamlit
info_log_placeholder = st.empty()

if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = True
    st.session_state.info_logs = []  # Clear previous logs

class StreamlitInfoLogHandler(logging.Handler):
    def emit(self, record):
        if record.levelno == logging.INFO:  # Only show INFO level
            msg = self.format(record)
            if "info_logs" not in st.session_state:
                st.session_state["info_logs"] = []
            st.session_state["info_logs"].append(msg)
            st.session_state["info_logs"] = st.session_state["info_logs"][-100:]  # Keep last 100 logs

            # Update the Streamlit UI
            with info_log_placeholder:
                st.markdown("### ℹ️ Info Logs")
                st.code("\n".join(st.session_state["info_logs"]), language="text")

# Attach handler
if not any(isinstance(h, StreamlitInfoLogHandler) for h in logging.getLogger().handlers):
    info_handler = StreamlitInfoLogHandler()
    info_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(info_handler)





st.title("☕ ZUS Coffee Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.session_state.chat_history:
    with st.container():
        st.markdown(
            """
            <div style='height: 0.1px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; background-color: #f9f9f9'>
            """,
            unsafe_allow_html=True
        )

        for role, msg in st.session_state.chat_history:
            if role == "user":
                st.markdown(f"<div style='margin-bottom: 10px'><strong>You:</strong> {msg}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='margin-bottom: 10px'><strong>Bot:</strong> {msg}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        
user_input = st.chat_input("Ask about outlets, drinks, prices, or math!")

if user_input:
    st.chat_message("user").markdown(user_input)

    try:
        if is_math_query(user_input):
            result = safe_calculate(user_input)

        elif is_outlet_query(user_input):
            missing = detect_missing_info(user_input, last_outlet["name"] or "", keywords)
            if missing == "location":
                result = "Could you please specify the location or outlet name?"
            else:
                outlet_match = re.search(r"\b(ss ?\d+|usj ?\d+|mont ?kiara|klcc|bangsar|subang|kuala ?lumpur)\b", user_input.lower())
                if outlet_match:
                    last_outlet["name"] = outlet_match.group(0)
                if last_outlet["name"] and last_outlet["name"] not in user_input.lower():
                    user_input += f" (referring to the outlet at {last_outlet['name']})"
                sql_input = f"{memory.buffer}\nUser: {user_input}"
                response = db_chain.invoke({"query": sql_input})
                steps = response.get("intermediate_steps", [])
                sql_result = None
                for step in steps:
                    if isinstance(step, dict) and "result" in step:
                        sql_result = step["result"]
                        break
                    if isinstance(step, str) and step.startswith("[("):
                        try:
                            sql_result = ast.literal_eval(step)
                        except Exception as e:
                            logging.warning("SQL parsing error: %s", e)
                        break
                if isinstance(sql_result, list) and sql_result and isinstance(sql_result[0], tuple):
                    flat_data = ", ".join(str(item[0]) for item in sql_result)
                    result = summarizer_chain.invoke({"question": user_input, "data": flat_data}).strip()
                else:
                    result = response.get("result", "").strip() or "Sorry, I couldn't find the answer."

        elif is_product_query(user_input):
            if not vectorstore:
                result = "Bot: Sorry, product information is not available."
            else:
                retriever = vectorstore.as_retriever()
                docs = retriever.get_relevant_documents(user_input)
                if not docs:
                    result = "Bot: I couldn't find any relevant product info for that query."
                else:
                    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
                    result = qa_chain.run(user_input).strip()
                    if result.lower() in ["", "none"]:
                        result = "Bot: I couldn't find any relevant product info for that query."

        else:
            result = conversation.predict(input=user_input)

    except Exception as e:
        logging.error("Error occurred: %s", e)
        result = "Sorry, something went wrong."

    st.chat_message("assistant").markdown(result)
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("assistant", result))
