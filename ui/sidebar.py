import streamlit as st
from chatbot.processor import process_uploaded_file, fetch_webpage_content

def configure_sidebar(chatbot_manager):
    """Configure the sidebar with all settings and options"""
    with st.sidebar:
        st.title("💫 Character Settings")
        
        # Basic settings with explanations
        configure_basic_settings(chatbot_manager)
        
        # Character traits settings
        configure_character_traits(chatbot_manager)
        
        # Advanced settings
        configure_advanced_settings(chatbot_manager)
        
        # Save configuration button
        if st.button("Save Character Settings", use_container_width=True):
            chatbot_manager.save_config()
            st.success("Character settings saved!")
            
        st.divider()
        
        # Content import options
        configure_content_import(chatbot_manager)
        
        # Debug and reset options
        configure_debug_options()
        
        # Instructions section
        display_instructions()

def configure_basic_settings(chatbot_manager):
    """Configure basic character settings"""
    with st.expander("Basic Character Information", expanded=True):
        chatbot_manager.config["name"] = st.text_input(
            "Name", 
            chatbot_manager.config.get("name", ""),
            help="The name of your character"
        )
        
        chatbot_manager.config["role"] = st.text_input(
            "Role/Occupation", 
            chatbot_manager.config.get("role", ""),
            help="What does your character do? (e.g., Wizard, Detective, Teacher)"
        )
        
        chatbot_manager.config["appearance"] = st.text_area(
            "Appearance", 
            chatbot_manager.config.get("appearance", ""),
            help="How does your character look? Be descriptive."
        )

def configure_character_traits(chatbot_manager):
    """Configure character personality traits"""
    with st.expander("Character Traits", expanded=False):
        chatbot_manager.config["personality"] = st.text_area(
            "Personality", 
            chatbot_manager.config.get("personality", ""),
            help="Describe your character's personality traits"
        )
        
        chatbot_manager.config["interests"] = st.text_area(
            "Interests", 
            chatbot_manager.config.get("interests", ""),
            help="What topics interest your character?"
        )
        
        chatbot_manager.config["abilities"] = st.text_area(
            "Abilities", 
            chatbot_manager.config.get("abilities", ""),
            help="What special abilities or skills does your character have?"
        )

def configure_advanced_settings(chatbot_manager):
    """Configure advanced character settings"""
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

def configure_content_import(chatbot_manager):
    """Configure content import options"""
    st.subheader("Import Character Content")

    # File uploader section
    uploaded_file = st.file_uploader(
        "Upload a file with character information",
        type=["txt", "pdf","py","csv"],
        help="Upload a text or PDF file containing details about your character"
    )

    # Ensure session state exists
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
        st.session_state.character_data = None

    # Store uploaded file in session state but don't process yet
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.success("File uploaded. Click 'Add Documents' to process.")

    # Only process when the user clicks 'Add Documents'
    if st.session_state.uploaded_file and st.button("Add Documents", use_container_width=True):
        with st.spinner("Processing file..."):
            st.session_state.character_data = process_uploaded_file(st.session_state.uploaded_file, chatbot_manager)
            st.success("Character information extracted!")

    # Show extracted information if available
    if st.session_state.character_data:
        with st.expander("Preview Extracted Information", expanded=True):
            for key, value in st.session_state.character_data.items():
                st.text_area(key.capitalize(), value, disabled=True, height=100)

        if st.button("Apply These Settings", use_container_width=True):
            chatbot_manager.update_config(st.session_state.character_data)
            st.success("Settings applied! Refresh the page to see changes.")
            st.rerun()

    # Web content import section
    st.subheader("Import from Web")
    web_url = st.text_input(
        "Webpage URL",
        placeholder="https://example.com/character-bio",
        help="Enter a URL containing character information"
    )

    if "web_character_data" not in st.session_state:
        st.session_state.web_character_data = None

    if web_url and st.button("Process Webpage", use_container_width=True):
        with st.spinner("Fetching webpage content..."):
            st.session_state.web_character_data = fetch_webpage_content(web_url, chatbot_manager)
            if st.session_state.web_character_data:
                st.success("Character information extracted from webpage!")

    if st.session_state.web_character_data:
        with st.expander("Preview Extracted Information", expanded=True):
            for key, value in st.session_state.web_character_data.items():
                st.text_area(key.capitalize(), value, disabled=True, height=100)

        if st.button("Apply Web Data", use_container_width=True):
            chatbot_manager.update_config(st.session_state.web_character_data)
            st.success("Settings applied from web data!")
            st.rerun()


def configure_debug_options():
    """Configure debug and reset options"""
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

def display_instructions():
    """Display usage instructions"""
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