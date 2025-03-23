import logging
import os
from typing import Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import re
import json
import getpass


# Configure logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
HUGGINGFACE_API_KEY = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CONFIG_FILE = "chatbot_config.json"


if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

CONFIG_FILE = "chatbot_config.json"



# Constants
MODEL = "llama3-70b-8192"

class ChatbotManager:
    def __init__(self):
        """Initialize the ChatbotManager with default configuration"""
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self.retriever = None
        self.load_config()
        self._initialize_llm()
    
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
    
    def _initialize_llm(self):
        """Initialize the LLM with current configuration"""
        try:
            self.llm = ChatGroq(
                model=MODEL,
                api_key=GROQ_API_KEY, 
                temperature=self.config.get("temperature", 0.7),
                max_tokens=self.config.get("response_length", 500)
            )
            logger.info(f"LLM initialized with model: {MODEL}")
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            self.llm = None

    def _initialize_qa_chain(self):
        """Initialize the QA chain if vectorstore is available"""
        if not self.vectorstore or not self.llm:
            logger.warning("Cannot initialize QA chain: vectorstore or LLM not available")
            return False
        
        try:
            self.retriever = self.vectorstore.as_retriever(search_type="similarity_score_threshold",search_kwargs={"score_threshold": 0.5,"k": 5})
            
            system_prompt = (
                "Refer to the documents to answer the question when needed. "
                "If you don't know the answer, say you don't know. "
                "{context}"
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
            )
            question_answer_chain = create_stuff_documents_chain(self.llm, prompt)

            self.qa_chain = create_retrieval_chain(self.retriever,question_answer_chain)


            logger.info("QA chain initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing QA chain: {str(e)}")
            self.qa_chain = None
            return False
    
    def update_llm_parameters(self):
        """Update LLM parameters based on current configuration"""
        if self.llm:
            self.llm.temperature = self.config.get("temperature", 0.7)
            self.llm.max_tokens = self.config.get("response_length", 500)
            logger.info("LLM parameters updated")
        else:
            self._initialize_llm()
    
    def save_config(self) -> bool:
        """Save the current configuration and update LLM parameters"""
        try:
            # In a real application, you might save to a file or database
            with open(CONFIG_FILE, "w") as f:
                json.dump(self.config, f, indent=2)
            logger.info("Configuration saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update the configuration with new values and update LLM parameters"""
        for key, value in new_config.items():
            if key in self.config and value:
                self.config[key] = value
        
        # Update LLM parameters if temperature or response_length changed
        self.update_llm_parameters()
        logger.info("Configuration updated")
    
    def create_vectorstore(self, file_path: str) -> None:
        """Create vector store from document"""
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
            logger.info(f"Document length: {len(documents[0].page_content)} characters")
  
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
            
            docs = text_splitter.split_documents(documents)

            logger.info("Chunks: "+str(len(docs)))
            

            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            self.vectorstore = FAISS.from_documents(docs, embedding_model)
            self._initialize_qa_chain()
            logger.info(f"Vector store created with {len(docs)} documents")
        
        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}")
            raise e
    
    def analyze_content(self) -> Dict[str, str]:
        """Analyze content to extract character details"""
        if not self.vectorstore:
            logger.warning("No vector store available for content analysis")
            return {}
            
        try:
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
            except json.JSONDecodeError as e:       
                logger.error("Failed to extract valid JSON after multiple attempts")
                return {}
        except Exception as e:
            logger.error(f"Error analyzing content: {str(e)}")
            return {}
