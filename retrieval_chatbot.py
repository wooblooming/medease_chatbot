import dotenv
dotenv.load_dotenv()
from langchain.document_loaders import TextLoader

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document

class Utf8TextLoader(TextLoader):
    def __init__(self, file_path):
        self.file_path = file_path

    def lazy_load(self):
        try:
            with open(self.file_path, encoding='utf-8') as f:
                text = f.read()
            yield Document(page_content=text)
        except Exception as e:
            raise RuntimeError(f"Error loading {self.file_path}") from e

file_path = "./hidac_cleaned7_text.txt"
loader = Utf8TextLoader(file_path)

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)

retriever = db.as_retriever()

from langchain.agents.agent_toolkits import create_retriever_tool

tool = create_retriever_tool(
    retriever,
    "hidac_cleaned7_text",
    "Searches and returns documents regarding the hidac_cleaned7_text.",
)
tools = [tool]

from langchain.agents.agent_toolkits import create_conversational_retrieval_agent

from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0)

agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)

# result = agent_executor({"input": "hi, im bob"})

# result["output"]

import streamlit as st
import time

import dotenv
dotenv.load_dotenv()

st.title("ChatGPT")

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        result = agent_executor({"input": prompt})
        for chunk in result['output'].split():
            full_response += chunk + " "
            time.sleep(0.05)

            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})