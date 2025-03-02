import os
import json
import streamlit as st
import PyPDF2  # PDF handling
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
import requests
from bs4 import BeautifulSoup
import re

# Load environment variables
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HF_Token")

def apply_changes():
    with open("chatbot_config.json", "w") as f:
        json.dump(st.session_state.chatbot_config, f)
    st.success("Changes Applied! Refresh the page if needed.")




import json
import re

def analyze_content(llm):
    retriever = st.session_state.vectorstore.as_retriever()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=False
    )

    query = """
    Output your name, role, appearance, personality, interests, abilities, additional_info briefly in proper JSON format without unescaped quotations according to the person described in the document.

    {
        "name": "...",
        "role": "...",
        "appearance": "...",
        "personality": "...",
        "interests": "...",
        "abilities": "...",
        "additional_info": "..."
    }

    Briefly give all details for each attribute in the JSON. Make sure you cover every single attribute and only give the JSON in the format above.
    Do not output anything else apart from the JSON.
    Do not be verbose.
    """

    response = qa_chain.run(query)
    print("Raw Response:\n", response)  # Debugging
    
    # Extract only the JSON part using regex
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)  # Extract matched JSON
        try:
            extracted_data = json.loads(json_str)  # Validate JSON
        except json.JSONDecodeError:
            extracted_data = {}  # Return empty JSON if parsing fails
    else:
        extracted_data = {}

    return extracted_data



def process_uploaded_file(uploaded_file,llm):
    if uploaded_file is None:
        return
    
    file_extension = uploaded_file.name.split(".")[-1].lower()
    file_contents = ""
    
    if file_extension == "txt":
        file_contents = uploaded_file.read().decode("utf-8")

        file_contents = "\n".join(line for line in file_contents.splitlines() if line.strip())


    elif file_extension == "pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        file_contents = "\n".join(
            re.sub(r"[^\w\s.,!?;:()'-]", "", line)
            for page in pdf_reader.pages if page.extract_text()
            for line in page.extract_text().split("\n") if line.strip()
        )
    
    with open("uploaded_text.txt", "w", encoding="utf-8") as f:
        f.write(file_contents)
    
    st.sidebar.success("File uploaded successfully!")
    create_vectorstore("uploaded_text.txt")
    return analyze_content(llm)

def fetch_webpage_content(webpage_url,llm):
    response = requests.get(webpage_url)
    if response.status_code != 200:
        st.sidebar.error("Failed to fetch webpage content.")
        return None
    
    soup = BeautifulSoup(response.text, "html.parser")
    text = "\n".join([p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True).strip()])
    
    with open("webpage_text.txt", "w", encoding="utf-8") as f:
        f.write(text)
    
    st.sidebar.success("Webpage content fetched!")
    create_vectorstore("webpage_text.txt")
    return analyze_content(llm)

def create_vectorstore(file_path):
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    docs = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state.vectorstore = FAISS.from_documents(docs, embedding_model)



def generate_response(user_input,llm):
    history = st.session_state.memory.load_memory_variables({})["history"]
    character_details = f"""
        You are {st.session_state.chatbot_config['name']}, a {st.session_state.chatbot_config['role']}.
        Your appearance: {st.session_state.chatbot_config['appearance']}
        Your personality: {st.session_state.chatbot_config['personality']}
        Your interests: {st.session_state.chatbot_config['interests']}
        Your abilities: {st.session_state.chatbot_config['abilities']}
        Keep Note:{st.session_state.chatbot_config['additional_info']}
        Do not create extra questions.
        Give only text as answer.
        NO code in your responses.
        No weird symbols and extra characters, only alphabets and numbers.
        Refer to yourself in first person with I and my.
        Include verb actions where applicable in your responses in italic font.
        Refers to the user as you.
        DO NOT output code blocks.
        Make sure you progress the chat in a meaningful way rather than just repeating the same sentences over again.
        DO not repeat the same responses again and again.
        Use diverse vocabulary.
        Only provide single helpful human-like responses.
        Include your personality in your responses.
        DO not provide unhelpful or alternative answers.
        Chat History:
        {history}
        
        Give a single response concise paragraph to this query according to the role and persona given to you and do not include chat history in your responses: {user_input}
    """
    
    if st.session_state.vectorstore:
        retriever = st.session_state.vectorstore.as_retriever()
        rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
        response = rag_chain.run(character_details)
    else:
        response = llm(character_details)
    st.session_state.memory.save_context({"input": user_input}, {"output": response})
    print(response)
    response = re.sub(r'\n\s*\n+', '\n', response).strip()

    return response.replace("\n", " ").strip()

def main():
    st.set_page_config(page_title="Customizable Chatbot", layout="wide")
    st.title("🤖 Personalizable Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None  

    if "chatbot_config" not in st.session_state:
        if os.path.exists("chatbot_config.json"):
            with open("chatbot_config.json", "r") as f:
                st.session_state.chatbot_config = json.load(f)
        else:
            st.session_state.chatbot_config = {
                "name": "AI Assistant",
                "role": "Assistant",
                "appearance": "A sleek, futuristic digital entity.",
                "personality": "Helpful, intelligent, and friendly.",
                "interests": "Technology, science, and philosophy.",
                "abilities": "Natural language understanding, knowledge retrieval, and personalized interactions.",
                "additional_info": ""
            }

    st.sidebar.header("Customize Your Chatbot")
    for key in st.session_state.chatbot_config:
        st.session_state.chatbot_config[key] = st.sidebar.text_area(key.capitalize(), st.session_state.chatbot_config[key])
    
    if st.sidebar.button("Apply Changes"):
        apply_changes()
    
    st.sidebar.subheader("Choose a Model")
    model_options = {"Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.3"}
    selected_model = st.sidebar.selectbox("Select a Model", list(model_options.keys()))
    repo_id = model_options[selected_model]

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True)

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.8,
        top_k=50,
        huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        max_length=500
    )

    llm_analyze = HuggingFaceEndpoint(
        response_format = {
          "type": "json_object"
        },
        repo_id=repo_id,
        temperature=0.8,
        top_k=50,
        huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        max_length=500

    )



    top_container = st.container()
    bottom_container = st.container()
    retriever = st.session_state.vectorstore.as_retriever() if st.session_state.vectorstore else None

    with bottom_container:
        user_input = st.text_input("Type your message...", key="chat_input",autocomplete="off")
        if st.button("Send") and user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            response = generate_response(user_input,llm)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            for message in st.session_state.messages:
                with top_container:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

    st.sidebar.subheader("Upload a File")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["txt", "pdf"])

    # Ensure the file is processed only once
    if uploaded_file and "last_uploaded_file" not in st.session_state:
        extracted_fields = process_uploaded_file(uploaded_file, llm_analyze)
        if extracted_fields:
            st.session_state.chatbot_config.update(extracted_fields)
            st.session_state.last_uploaded_file = uploaded_file.name  # Track uploaded file
            st.sidebar.success("Chatbot fields autofilled! Review before applying.")
            st.experimental_rerun()

        else:
            st.sidebar.warning("Could not extract meaningful details.")

    
    st.sidebar.subheader("Retrieve from Webpage")
    webpage_url = st.sidebar.text_input("Enter webpage URL", autocomplete="off")
    if st.sidebar.button("Fetch Content") and webpage_url:
        extracted_fields = fetch_webpage_content(webpage_url,llm)
        if extracted_fields:
            st.session_state.chatbot_config.update(extracted_fields)
            st.sidebar.success("Chatbot fields autofilled! Review before applying.")
            st.experimental_rerun()
        else:
            st.sidebar.warning("Could not extract meaningful details.")

if __name__ == "__main__":
    main()
