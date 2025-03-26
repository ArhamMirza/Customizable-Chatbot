import streamlit as st
import pdfplumber
import requests
from bs4 import BeautifulSoup
import logging
from typing import Dict, Optional
from .manager import ChatbotManager
import urllib3
from urllib.parse import urlparse
import re
import bleach
import os
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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


class WebPageSecurityManager:
    # Comprehensive list of safe, reputable domains
    SAFE_DOMAINS = [
        # Academic and Educational
        'wikipedia.org', 'scholar.google.com', 'academia.edu', 'researchgate.net', 
        'mit.edu', 'harvard.edu', 'stanford.edu', 'berkeley.edu', 'yale.edu', 
        'ox.ac.uk', 'cam.ac.uk', 'imperial.ac.uk', 'ethz.ch', 'caltech.edu',
        
        # News and Media (Established, Reputable Sources)
        'bbc.com', 'bbc.co.uk', 'npr.org', 'reuters.com', 'apnews.com', 
        'economist.com', 'nationalgeographic.com', 'scientificamerican.com', 
        'nature.com', 'science.org', 'pbs.org', 'newsweek.com', 'time.com',
        
        # Scientific and Research Organizations
        'nasa.gov', 'nih.gov', 'cdc.gov', 'noaa.gov', 'who.int', 'un.org', 
        'world-exchanges.org', 'ipcc.ch', 'iaea.org', 'oecd.org',
        
        # Technology and Open Source
        'github.com', 'gitlab.com', 'stackoverflow.com', 'arxiv.org', 
        'w3.org', 'mozilla.org', 'apache.org', 'linux.org', 'python.org', 
        'jupyter.org', 'kde.org', 'gnome.org',
        
        # Government and Public Services
        'usa.gov', 'data.gov', 'census.gov', 'loc.gov', 'gao.gov', 
        'uk.gov', 'europa.eu', 'un.org',
        
        # Non-Profit and International Organizations
        'unicef.org', 'redcross.org', 'amnesty.org', 'greenpeace.org', 
        'worldbank.org', 'imf.org', 'unesco.org', 'who.int',
        
        # Health and Medical Resources
        'mayoclinic.org', 'cdc.gov', 'nih.gov', 'medlineplus.gov', 
        'health.harvard.edu', 'who.int', 'cancer.org',
        
        # Professional and Scholarly Associations
        'acm.org', 'ieee.org', 'apa.org', 'asa.org', 'mathematicalmindsets.com'
    ]

    @staticmethod
    def is_safe_url(url: str, strict: bool = True) -> bool:
        """
        Advanced URL safety validation
        
        Args:
            url (str): URL to validate
            strict (bool): Whether to apply strict domain validation
        
        Returns:
            bool: Whether the URL is considered safe
        """
        try:
            # Parse the URL
            parsed_url = urlparse(url)
            
            # Check for valid schemes
            if parsed_url.scheme not in ['http', 'https']:
                logging.warning(f"Invalid URL scheme: {parsed_url.scheme}")
                return False
            
            # Reject URLs with unusual characters
            if re.search(r'[<>"\'\x00-\x1F\x7F]', url):
                logging.warning("URL contains suspicious characters")
                return False
            
            # IP address check (optional, can be disabled)
            try:
                ip_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
                if ip_pattern.match(parsed_url.netloc):
                    logging.warning("Direct IP URLs are not allowed")
                    return False
            except Exception as ip_check_error:
                logging.error(f"IP check error: {ip_check_error}")
                return False
            
            # Domain validation
            # if strict:
            #     domain_match = any(
            #         safe_domain in parsed_url.netloc.lower() 
            #         for safe_domain in WebPageSecurityManager.SAFE_DOMAINS
            #     )
            #     if not domain_match:
            #         logging.warning(f"Domain not in safe list: {parsed_url.netloc}")
            #         return False
            
            return True
        
        except Exception as e:
            logging.error(f"URL validation error: {str(e)}")
            return False

    @staticmethod
    def sanitize_text(text: str, max_length: int = 500000) -> str:
        """
        Advanced text sanitization
        
        Args:
            text (str): Input text to sanitize
            max_length (int): Maximum allowed text length
        
        Returns:
            str: Sanitized text
        """
        # Remove control characters and potentially dangerous content
        sanitized_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Remove potential XSS and script injection attempts
        sanitized_text = bleach.clean(sanitized_text, tags=[], attributes={}, protocols=[], strip=True)
        sanitized_text = re.sub(r'javascript:', '', sanitized_text, flags=re.IGNORECASE)
        
        # Limit text length
        return sanitized_text[:max_length]

    @staticmethod
    def content_hash(content: str) -> str:
        """
        Generate a hash to detect duplicate or suspicious content
        
        Args:
            content (str): Content to hash
        
        Returns:
            str: SHA-256 hash of the content
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

def fetch_webpage_content(
    url: str, 
    chatbot_manager, 
    max_content_size: int = 10 * 1024 * 1024,  # 10 MB default limit
    strict_domain_check: bool = True
) -> Optional[Dict[str, str]]:
    """
    Advanced secure webpage content fetching
    
    Args:
        url (str): URL of the webpage to fetch
        chatbot_manager: Chatbot management object
        max_content_size (int): Maximum allowed content size
        strict_domain_check (bool): Whether to enforce strict domain validation
    
    Returns:
        Optional[Dict[str, str]]: Processed webpage content or None
    """
    # Validate URL safety
    if not WebPageSecurityManager.is_safe_url(url, strict=strict_domain_check):
        st.error("Invalid or potentially malicious URL")
        logging.warning(f"Blocked potentially unsafe URL: {url}")
        return None
    
    try:
        # Disable insecure request warnings
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Enhanced security headers with randomization
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Referer": "https://www.google.com"  # Add a plausible referer
        }
        
        # Rate limiting simulation
        time.sleep(1)  # Basic rate limiting
        
        # Secure request with advanced parameters
        with requests.Session() as session:
            # Use session for potential connection pooling and cookie management
            response = session.get(
                url, 
                headers=headers, 
                timeout=(5, 10),  # Connect timeout, read timeout
                verify=True,  # Enforce SSL certificate verification
                stream=True,
                allow_redirects=False  # Prevent unintended redirects
            )
            
            # Advanced response validation
            if response.status_code != 200:
                st.error(f"Webpage fetch failed: HTTP {response.status_code}")
                logging.warning(f"Unexpected status code for {url}: {response.status_code}")
                return None
            
            # Check content length
            content_length = int(response.headers.get('content-length', 0))
            if content_length > max_content_size:
                st.error(f"Content size exceeds {max_content_size/1024/1024} MB")
                return None
            
            # Read and limit response
            response.raw.decode_content = True
            response_text = response.text[:max_content_size]
            
            # Content hash for duplicate detection
            content_signature = WebPageSecurityManager.content_hash(response_text)
            logging.info(f"Content hash for {url}: {content_signature}")
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response_text, "html.parser")
        
        # Remove potentially dangerous elements
        for element in soup(["script", "style", "iframe", "object", "embed", "form"]):
            element.decompose()
        
        # Extract text from safe elements
        text_elements = soup.find_all([
            "p", "h1", "h2", "h3", "h4", 
            "article", "section", 
            "div.content", "main", "body"
        ])
        
        # Combine and sanitize text
        text = "\n\n".join([
            WebPageSecurityManager.sanitize_text(elem.get_text(strip=True)) 
            for elem in text_elements 
            if elem.get_text(strip=True)
        ])
        
        # Secure temporary file handling
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode='w', 
            delete=False, 
            encoding='utf-8', 
            suffix='.txt'
        ) as temp_file:
            temp_file.write(text)
            temp_file_path = temp_file.name
        
        try:
            # Process content
            chatbot_manager.create_vectorstore(temp_file_path)
            # result = chatbot_manager.analyze_content()
            return 
        
        finally:
            # Always clean up the temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logging.error(f"Error cleaning up temp file: {cleanup_error}")
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Secure fetch error: {str(e)}")
        st.error(f"Secure fetch error: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error in webpage fetching: {str(e)}")
        st.error(f"Unexpected error: {str(e)}")
        return None

# Additional security configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)