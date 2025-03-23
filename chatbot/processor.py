import streamlit as st
import pdfplumber
import requests
from bs4 import BeautifulSoup
import logging
from typing import Dict, Optional

# Import the ChatbotManager class using relative import
from .manager import ChatbotManager

# Configure logging
logger = logging.getLogger(__name__)

# Constants
TEMP_TEXT_FILE = "temp_text.txt"

def process_uploaded_file(file, chatbot_manager: ChatbotManager) -> Optional[Dict[str, str]]:
    """Process uploaded file and extract character details"""
    if file is None:
        return None
        
    try:
        file_extension = file.name.split(".")[-1].lower()
        logger.info(file_extension+" file uploaded")
        
        if file_extension == "txt":
            file_contents = file.read().decode("utf-8")
            file_contents = "\n".join(line for line in file_contents.splitlines() if line.strip())

        elif file_extension == "pdf":
            with pdfplumber.open(file) as pdf:
                file_contents = "\n".join(
                    page.extract_text() for page in pdf.pages 
                    if page.extract_text() and page.extract_text().strip()
                )

        elif file_extension in ["py", "java", "cpp", "js"]:  # Add more extensions as needed
            file_contents = file.read().decode("utf-8")
            file_contents = "\n".join(line for line in file_contents.splitlines() if line.strip())


        elif file_extension == "csv":
            import csv
            reader = csv.reader(file)
            file_contents = "\n".join(
                ",".join(row) for row in reader 
                if any(field.strip() for field in row)
            )
            
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return None
            
        # Save processed text to temporary file
        with open(TEMP_TEXT_FILE, "w", encoding="utf-8") as f:
            f.write(file_contents)
            
        # Create vector store and analyze content
        chatbot_manager.create_vectorstore(TEMP_TEXT_FILE)
        # return chatbot_manager.analyze_content()
        return
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        st.error(f"Error processing file: {str(e)}")
        return None

def fetch_webpage_content(url: str, chatbot_manager: ChatbotManager) -> Optional[Dict[str, str]]:
    """Fetch and process content from a webpage"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            st.error(f"Failed to fetch webpage: Status code {response.status_code}")
            return None
            
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script, style elements and comments
        for element in soup(["script", "style"]):
            element.decompose()
            
        # Extract paragraphs and headings
        text_elements = soup.find_all(["p", "h1", "h2", "h3", "h4", "article", "section", "div.content"])
        text = "\n\n".join([elem.get_text(strip=True) for elem in text_elements if elem.get_text(strip=True)])
        
        # Save processed text
        with open(TEMP_TEXT_FILE, "w", encoding="utf-8") as f:
            f.write(text)
            
        # Create vector store and analyze content
        chatbot_manager.create_vectorstore(TEMP_TEXT_FILE)
        # return chatbot_manager.analyze_content()
        return
        
    except Exception as e:
        logger.error(f"Error fetching webpage: {str(e)}")
        st.error(f"Error fetching webpage: {str(e)}")
        return None