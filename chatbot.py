import streamlit as st
import pdfplumber
from langchain.chains import RetrievalQA
import requests
from bs4 import BeautifulSoup
from ChatbotManager import ChatbotManager
import logging
from typing import Dict, Any, Optional, List
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
TEMP_TEXT_FILE = "temp_text.txt"



def process_uploaded_file(file, chatbot_manager: ChatbotManager) -> Optional[Dict[str, str]]:
    """Process uploaded file and extract character details"""
    if file is None:
        return None
        
    try:
        file_extension = file.name.split(".")[-1].lower()
        
        if file_extension == "txt":
            file_contents = file.read().decode("utf-8")
            file_contents = "\n".join(line for line in file_contents.splitlines() if line.strip())
            
        elif file_extension == "pdf":
            with pdfplumber.open(file) as pdf:
                file_contents = "\n".join(
                    page.extract_text() for page in pdf.pages 
                    if page.extract_text() and page.extract_text().strip()
                )
            
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return None
            
        # Save processed text to temporary file
        with open(TEMP_TEXT_FILE, "w", encoding="utf-8") as f:
            f.write(file_contents)
            
        # Create vector store and analyze content
        chatbot_manager.create_vectorstore(TEMP_TEXT_FILE)
        return chatbot_manager.analyze_content()
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        st.error(f"Error processing file: {str(e)}")
        return None

def fetch_webpage_content(url: str, chatbot_manager: ChatbotManager) -> Optional[Dict[str, str]]:
    """Fetch and process content from a webpage"""
    try:
        with st.spinner("Fetching webpage content..."):
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
            return chatbot_manager.analyze_content()
            
    except Exception as e:
        logger.error(f"Error fetching webpage: {str(e)}")
        st.error(f"Error fetching webpage: {str(e)}")
        return None

def format_chat_history(messages: List[Dict[str, str]]) -> str:
    """Format chat history from messages for inclusion in prompt"""
    formatted_history = ""
    
    for msg in messages:
        role = "User" if msg["role"] == "user" else msg["role"].capitalize()
        content = msg["content"]
        formatted_history += f"{role}: {content}\n\n"
        
    return formatted_history.strip()

def generate_response(user_input: str, chatbot_manager: ChatbotManager, messages: List[Dict[str, str]]) -> str:
    """Generate response based on user input, character configuration and chat history"""
    # Extract the latest messages to include as context
    # Skip the current user message which is already being passed as user_input
    previous_messages = messages[:-1] if messages else []
    
    # Format chat history for inclusion in prompt
    chat_history = format_chat_history(previous_messages)
    
    if not chatbot_manager.vectorstore:
        # Simple response without RAG
        llm = chatbot_manager.get_mistral_llm()
        character_details = create_character_prompt(chatbot_manager.config, user_input, chat_history)
        
        try:
            response = llm(character_details)
            return clean_response(response)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I'm having trouble responding right now. Please try again."
    else:
        # RAG-enhanced response
        try:
            llm = chatbot_manager.get_mistral_llm()
            retriever = chatbot_manager.vectorstore.as_retriever()
            
            # Use RetrievalQA instead of ConversationalRetrievalChain to have more control
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff"
            )
            
            # Include chat history and character details in the prompt
            character_details = create_character_prompt(chatbot_manager.config, user_input, chat_history)
            response = qa_chain.run(character_details)
            return clean_response(response)
        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}")
            return "I apologize, but I'm having trouble accessing my knowledge to respond appropriately. Please try again."

def create_character_prompt(config: Dict[str, str], user_input: str, chat_history: str = "") -> str:
    """Create a concise, structured prompt for character-based responses."""
    
    prompt = f"""
    
    You are {config['name']}, a {config['role']}.
    CHARACTER DETAILS:
    - {config['personality']}
    - Appearance: {config['appearance']}
    - Interests: {config['interests']}
    - Abilities: {config['abilities']}
    {f"- Additional info: {config['additional_info']}" if 'additional_info' in config else ''}
    GUIDELINES:
    - Stay in character and use first-person perspective.
    - Use *italics* for actions (e.g., *smiles*).
    - Answer concisely but in character.
    - Adapt tone to match user context.
    - Reference retrieved knowledge when available, but do not make up facts.
    """
    if chat_history:
        prompt += f"\nCHAT HISTORY:\n{chat_history}\n"
    prompt += f"\nUSER QUERY: {user_input}\n\nRESPOND AS {config['name']}:"

    return prompt


def clean_response(response: str) -> str:
    """Clean and format the response."""
    
    # Remove everything from the first occurrence of an unwanted pattern onward
    unwanted_patterns = [
        r"Unhelpful Answer:.*",
        r"Unhelpful Alternative Answer:.*",
        r"User query:.*",
        r"Previous conversation:.*",
        r"Alternative Answer:.*",
        r"Extra Question:.*", 
        r"Irrelevant Answer:.*", 
        r"Not An Answer:.*", 
        r"Off-topic Answer:.*", 
        r"Incomplete Answer:.*", 
        r"Lengthy Response:.*", 
        r"Incorrect Answer:.*", 
        r"Unrelated Answer:.*", 
        r"Vague Answer:.*"
    ]
    
    for pattern in unwanted_patterns:
        response = re.sub(pattern, "", response, flags=re.IGNORECASE | re.DOTALL).strip()
    
    # Remove excessive newlines
    response = re.sub(r'\n\s*\n+', '\n', response).strip()
    
    # Convert markdown italics (*text*) to HTML italics (<i>text</i>)
    response = re.sub(r'\*(.*?)\*', r'<i>\1</i>', response)
    
    return response


def display_chat_interface(chatbot_manager: ChatbotManager):
    """Display the chat interface"""
    st.title(f"💬 Chat with {chatbot_manager.config['name']}")
    
    # Initialize messages if not in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
            
    # Chat input
    if user_input := st.chat_input("Type your message..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Pass the entire message history to ensure context is maintained
                response = generate_response(user_input, chatbot_manager, st.session_state.messages)
                st.markdown(response, unsafe_allow_html=True)
                
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def is_unwanted_response(response: str) -> bool:
    """Check if the response contains unwanted text patterns."""
    unwanted_patterns = [
        "Unhelpful Answer:", "Extra Question:", "Irrelevant Answer:", "Not An Answer:",
        "Off-topic Answer:", "Incomplete Answer:", "Lengthy Response:", "Incorrect Answer:",
        "Unrelated Answer:", "Vague Answer:"
    ]
    return any(pattern in response for pattern in unwanted_patterns)

def main():
    st.set_page_config(
        page_title="Character Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize chatbot manager
    if "chatbot_manager" not in st.session_state:
        st.session_state.chatbot_manager = ChatbotManager()
    
    chatbot_manager = st.session_state.chatbot_manager
    print(chatbot_manager.config["name"])
    
    # Sidebar configuration
    with st.sidebar:
        st.title("💫 Character Settings")
        
        # Basic settings with explanations
        with st.expander("Basic Character Information", expanded=True):
            chatbot_manager.config["name"] = st.text_input(
                "Name", 
                chatbot_manager.config.get("name"),
                help="The name of your character"
            )
            
            chatbot_manager.config["role"] = st.text_input(
                "Role/Occupation", 
                chatbot_manager.config.get("role"),
                help="What does your character do? (e.g., Wizard, Detective, Teacher)"
            )
            
            chatbot_manager.config["appearance"] = st.text_area(
                "Appearance", 
                chatbot_manager.config.get("appearance"),
                help="How does your character look? Be descriptive."
            )
            
        # Personality and interests
        with st.expander("Character Traits", expanded=False):
            chatbot_manager.config["personality"] = st.text_area(
                "Personality", 
                chatbot_manager.config.get("personality"),
                help="Describe your character's personality traits"
            )
            
            chatbot_manager.config["interests"] = st.text_area(
                "Interests", 
                chatbot_manager.config.get("interests"),
                help="What topics interest your character?"
            )
            
            chatbot_manager.config["abilities"] = st.text_area(
                "Abilities", 
                chatbot_manager.config.get("abilities"),
                help="What special abilities or skills does your character have?"
            )
            
        # Advanced settings
        with st.expander("Advanced Settings", expanded=False):
            chatbot_manager.config["additional_info"] = st.text_area(
                "Additional Notes", 
                chatbot_manager.config.get("additional_info", ""),
                help="Any additional information or special instructions for the character"
            )
            
            chatbot_manager.config["temperature"] = st.slider(
                "Creativity", 
                min_value=0.1, 
                max_value=1.0, 
                value=chatbot_manager.config.get("temperature", 0.7),
                step=0.1,
                help="Higher values make responses more creative but less predictable"
            )
            
            chatbot_manager.config["response_length"] = st.slider(
                "Response Length", 
                min_value=100, 
                max_value=1000, 
                value=chatbot_manager.config.get("response_length", 500),
                step=50,
                help="Maximum length of character responses"
            )
            
        # Save configuration
        if st.button("Save Character Settings", use_container_width=True):
            chatbot_manager.save_config()
            st.success("Character settings saved!")
            
        st.divider()
        
        # Content import section
        st.subheader("Import Character Content")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload a file with character information", 
            type=["txt", "pdf"],
            help="Upload a text or PDF file containing details about your character"
        )
        
        character_data = None

        if uploaded_file:
            with st.spinner("Processing file..."):
                character_data = process_uploaded_file(uploaded_file, chatbot_manager)
                if character_data:
                    st.success("Character information extracted!")
            
        if character_data:
            with st.expander("Preview Extracted Information", expanded=True):
                for key, value in character_data.items():
                    st.text_area(key.capitalize(), value, disabled=True, height=100)

            if st.button("Apply These Settings", use_container_width=True):
                chatbot_manager.update_config(character_data)
                print("updated")
                st.success("Settings applied! Refresh the page to see changes.")
                st.rerun()

        
        # Web content import
        st.subheader("Import from Web")
        web_url = st.text_input(
            "Webpage URL",
            placeholder="https://example.com/character-bio",
            help="Enter a URL containing character information"
        )
        
        if "web_character_data" not in st.session_state:
            st.session_state.web_character_data = None

        if web_url:
            if st.button("Process Webpage", use_container_width=True):
                with st.spinner("Fetching webpage content..."):
                    st.session_state.web_character_data = fetch_webpage_content(web_url, chatbot_manager)
                    if st.session_state.web_character_data and len(st.session_state.web_character_data) > 0:
                        st.success("Character information extracted from webpage!")

        if st.session_state.web_character_data:
            with st.expander("Preview Extracted Information", expanded=True):
                for key, value in st.session_state.web_character_data.items():
                    st.text_area(key.capitalize(), value, disabled=True, height=100)

            if st.button("Apply Web Data", use_container_width=True):
                chatbot_manager.update_config(st.session_state.web_character_data)
                st.success("Settings applied from web data!")
                st.rerun()

        
        # Debug viewer
        with st.expander("Debug - Chat History", expanded=False):
            if "messages" in st.session_state and st.session_state.messages:
                st.json(st.session_state.messages)
            else:
                st.info("No chat history available.")
        
        # Reset button
        if st.button("Reset Chat History", use_container_width=True):
            st.session_state.messages = []
            st.success("Chat history cleared!")
            st.rerun()
            
        # Instructions
        with st.expander("Instructions", expanded=False):
            st.markdown("""
            ## How to Use
            1. Configure your character's settings in the sidebar
            2. Optionally upload a file or provide a webpage to extract character information
            3. Chat with your character in the main chat window
            4. Save your settings when you're happy with your character
            
            ## Tips
            - Be specific about personality traits to get more engaging responses
            - Adjust the creativity slider to control response randomness
            - Click "Reset Chat" to start a fresh conversation
            - The "Additional Notes" field can be used for special character quirks or backstory elements
            """)
    
    # Main chat area
    display_chat_interface(chatbot_manager)

if __name__ == "__main__":
    main()