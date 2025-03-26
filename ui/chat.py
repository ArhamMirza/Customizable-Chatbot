import streamlit as st
from typing import List, Dict
from chatbot.response import generate_response

def display_chat_interface(chatbot_manager):
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