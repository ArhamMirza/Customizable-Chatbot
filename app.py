import streamlit as st
from chatbot.manager import ChatbotManager
from ui.chat import display_chat_interface
from ui.sidebar import configure_sidebar

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
    
    # Configure sidebar and chat interface
    configure_sidebar(chatbot_manager)
    display_chat_interface(chatbot_manager)

if __name__ == "__main__":
    main()