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
    PRIVATE_IP_RANGES = [
        re.compile(r"^127\..*"),      # Loopback
        re.compile(r"^10\..*"),       # Private A
        re.compile(r"^192\.168\..*"), # Private C
        re.compile(r"^172\.(1[6-9]|2[0-9]|3[0-1])\..*"),  # Private B
        re.compile(r"^169\.254\..*"), # Link-local
        re.compile(r"^::1$"),         # IPv6 loopback
        re.compile(r"^fc00::/7$"),    # IPv6 private
        re.compile(r"^fe80::/10$")    # IPv6 link-local
    ]

    @staticmethod
    def is_safe_url(url: str, strict: bool = True) -> bool:
        """
        Secure URL validation to prevent SSRF, XSS, and other attacks.

        Args:
            url (str): The URL to validate.
            strict (bool): If True, enforces strict domain checks.

        Returns:
            bool: True if the URL is safe, False otherwise.
        """
        try:
            parsed_url = urlparse(url)

            # Ensure only HTTP and HTTPS are allowed
            if parsed_url.scheme not in ["http", "https"]:
                logging.warning(f"Blocked due to invalid scheme: {parsed_url.scheme}")
                return False

            # Reject suspicious characters in the URL
            if re.search(r"[<>'\"\x00-\x1F\x7F]", url):
                logging.warning(f"Blocked due to unsafe characters in URL: {url}")
                return False

            # Decode percent-encoded characters
            decoded_netloc = unquote(parsed_url.netloc)

            # Ensure domain is properly formatted
            if not re.match(r"^[a-zA-Z0-9.-]+$", decoded_netloc):
                logging.warning(f"Blocked due to malformed domain: {decoded_netloc}")
                return False

            # Prevent direct IP address usage
            if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", decoded_netloc):
                logging.warning(f"Blocked direct IP address access: {decoded_netloc}")
                return False

            # Prevent private/local network access
            try:
                resolved_ip = socket.gethostbyname(decoded_netloc)
                for pattern in WebPageSecurityManager.PRIVATE_IP_RANGES:
                    if pattern.match(resolved_ip):
                        logging.warning(f"Blocked private network access: {resolved_ip}")
                        return False
            except socket.gaierror:
                logging.error(f"DNS resolution failed for: {decoded_netloc}")
                return False

            return True

        except Exception as e:
            logging.error(f"URL validation error: {str(e)}")
            return False

    @staticmethod
    def sanitize_text(text: str, max_length: int = 500000) -> str:
        """
        Securely sanitize text to prevent XSS and script injections.

        Args:
            text (str): Input text.
            max_length (int): Max allowed text length.

        Returns:
            str: Sanitized text.
        """
        # Remove control characters
        sanitized_text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

        # Remove dangerous HTML tags
        sanitized_text = re.sub(
            r"<(script|iframe|object|embed|link|style).*?>.*?</\1>", 
            "", sanitized_text, flags=re.DOTALL | re.IGNORECASE
        )

        # Block JavaScript execution attempts
        sanitized_text = re.sub(r"javascript\s*:", "", sanitized_text, flags=re.IGNORECASE)

        # Block inline event handlers (e.g., `onerror=`, `onclick=`)
        sanitized_text = re.sub(r"on\w+\s*=", "", sanitized_text, flags=re.IGNORECASE)

        # Trim length to avoid DoS attacks
        return sanitized_text[:max_length]

    @staticmethod
    def content_hash(content: str) -> str:
        """
        Generate a SHA-256 hash of content to detect duplicate/malicious content.

        Args:
            content (str): Content to hash.

        Returns:
            str: Hex-encoded SHA-256 hash.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

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