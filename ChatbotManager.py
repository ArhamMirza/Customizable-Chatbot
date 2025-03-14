import os
import json
import streamlit as st
import PyPDF2
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
import requests
from bs4 import BeautifulSoup
import re
import time
import logging
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HF_TOKEN")

# Constants
CONFIG_FILE = "chatbot_config.json"
MAX_RETRIES = 1
RETRY_DELAY = 2

class ChatbotManager:
    """Manages the chatbot configuration, storage and interactions"""
    
    def __init__(self):
        self.load_config()
        self.vectorstore = None
        self.memory = ConversationBufferWindowMemory(k=10, return_messages=True, memory_key="chat_history")
        
    def load_config(self) -> None:
        """Load configuration from file or create default"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    self.config = json.load(f)
                    logger.info("Configuration loaded successfully")
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in {CONFIG_FILE}")
                self.create_default_config()
        else:
            self.create_default_config()
            
    def create_default_config(self) -> None:
        """Create default configuration"""
        self.config = {
            "name": "AI Assistant",
            "role": "Assistant",
            "appearance": "A sleek, futuristic digital entity.",
            "personality": "Helpful, intelligent, and friendly.",
            "interests": "Technology, science, and philosophy.",
            "abilities": "Natural language understanding, knowledge retrieval, and personalized interactions.",
            "additional_info": "",
            "temperature": 0.7,
            "response_length": 500
        }
        self.save_config()
        
    def save_config(self) -> None:
        """Save configuration to file"""
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(self.config, f, indent=2)
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        self.config.update(new_config)
        self.save_config()
        
    def create_vectorstore(self, file_path: str) -> None:
        """Create vector store from document"""
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
            
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            
            docs = text_splitter.split_documents(documents)
            
            # Print out chunks
            for i, doc in enumerate(docs):
                print(f"Chunk {i+1}:\n{doc.page_content}\n{'-'*50}")

            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            self.vectorstore = FAISS.from_documents(docs, embedding_model)
            logger.info(f"Vector store created with {len(docs)} documents")
        
        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}")
            raise

            
    def get_mistral_llm(self, temperature: float = None, json_mode: bool = False) -> Any:
        """Get Mistral language model with appropriate settings"""
        if temperature is None:
            temperature = self.config.get("temperature", 0.8)
            
        model_params = {
            "repo_id": "mistralai/Mistral-7B-Instruct-v0.3",
            "temperature": temperature,
            "huggingfacehub_api_token": HUGGINGFACE_API_KEY,
            "max_length": self.config.get("response_length", 500)

        }
        
        if json_mode:
            model_params["response_format"] = {"type": "json_object"}
            
        return HuggingFaceEndpoint(**model_params)
            
    def analyze_content(self) -> Dict[str, str]:
        """Analyze content to extract character details"""
        if not self.vectorstore:
            logger.warning("No vector store available for content analysis")
            return {}
            
        try:
            llm = self.get_mistral_llm(temperature=0.8, json_mode=True)
            retriever = self.vectorstore.as_retriever()
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm, 
                retriever=retriever, 
                return_source_documents=False,
                chain_type="stuff"
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

            for attempt in range(MAX_RETRIES):
                try:
                    response = qa_chain.run(query)
                    logger.info(f"Raw analysis response received, length: {len(response)}")
                    
                    # Extract JSON using regex
                    json_match = re.search(r"\{[\s\S]*\}", response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        extracted_data = json.loads(json_str)
                        
                        # Ensure all required fields exist
                        required_fields = ["name", "role", "appearance", "personality", "interests", "abilities", "additional_info"]
                        for field in required_fields:
                            if field not in extracted_data:
                                extracted_data[field] = ""
                                
                        return extracted_data
                    else:
                        logger.warning(f"No JSON found in response: {response[:100]}...")
                        time.sleep(RETRY_DELAY)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error: {str(e)}, attempt {attempt+1}/{MAX_RETRIES}")
                    time.sleep(RETRY_DELAY)
                    
            logger.error("Failed to extract valid JSON after multiple attempts")
            return {}
        except Exception as e:
            logger.error(f"Error analyzing content: {str(e)}")
            return {}
