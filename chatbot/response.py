import re
import logging
from typing import Dict, List, Any
from transformers import pipeline
import streamlit as st
from tiktoken import encoding_for_model 
from langchain.prompts import ChatPromptTemplate


# Configure logging
logger = logging.getLogger(__name__)


def count_tokens(text: str, model_name="gpt-3.5-turbo") -> int:
    """Estimate the number of tokens in a given text."""
    enc = encoding_for_model(model_name)
    return len(enc.encode(text))

def log_token_usage(input_tokens: int, output_tokens: int):
    """Log token usage to a file."""
    log_entry = f"Input Tokens: {input_tokens}, Output Tokens: {output_tokens}\n"
    with open("token_log.txt", "a") as log_file:
        log_file.write(log_entry)

def format_chat_history(messages: List[Dict[str, str]]) -> str:
    """Format chat history from messages for inclusion in prompt"""
    formatted_history = ""
    
    for msg in messages:
        role = "User" if msg["role"] == "user" else msg["role"].capitalize()
        content = msg["content"]
        formatted_history += f"{role}: {content}\n\n"
        
    return formatted_history.strip()


def create_character_prompt(config: dict, user_input: str, chat_history: str = "") -> ChatPromptTemplate:
    """Create a structured ChatPromptTemplate for character-based responses with chat history as a separate message."""
    
    # System message with placeholders for character details
    system_message = """
    You are {name}, a {role}.
    
    CHARACTER DETAILS:
    - {personality}
    - Appearance: {appearance}
    - Interests: {interests}
    - Abilities: {abilities}
    {additional_info}
    
    GUIDELINES:
    - Stay in character and use first-person perspective.
    - Use *italics* for actions (e.g., *smiles*).
    - Answer concisely but in character.
    - Adapt tone to match user context.
    - Reference retrieved knowledge when available, but do not make up facts.
    - Do not cut off your sentences abruptly due to output token limits.
    """

    # Format additional_info properly (optional field)
    additional_info = f"- Additional info: {config.get('additional_info', '')}" if config.get('additional_info') else ""

    # Construct the final ChatPromptTemplate
    prompt_messages = [
        ("system", system_message.format(
            name=config["name"],
            role=config["role"],
            personality=config["personality"],
            appearance=config["appearance"],
            interests=config["interests"],
            abilities=config["abilities"],
            additional_info=additional_info
        )),
    ]

    # Add chat history if available
    if chat_history:
        prompt_messages.append(("user", "CHAT HISTORY:\n{chat_history}"))

    # Add user query separately
    prompt_messages.append(("user", "USER QUERY: {user_input}"))

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages(prompt_messages)

    return prompt.format(chat_history=chat_history, user_input=user_input)



def generate_response(user_input: str, chatbot_manager, messages: List[Dict[str, str]]) -> str:
    """Generate response based on user input, character configuration, and chat history"""

    # Limit chat history tokens
    MAX_HISTORY_TOKENS = 1500  
    previous_messages = []
    token_count = 0

    for msg in reversed(messages[:-1]):  
        tokens = count_tokens(msg["content"])
        if token_count + tokens > MAX_HISTORY_TOKENS:
            break
        previous_messages.insert(0, msg)
        token_count += tokens

    chat_history = format_chat_history(previous_messages)

    try:
        if not chatbot_manager.llm:
            return "I'm having trouble connecting to my language model. Please try again later."

        # 🔹 Create the full prompt for the LLM
        character_details = create_character_prompt(chatbot_manager.config, user_input, chat_history)

        # 🔹 Count input tokens
        input_tokens = count_tokens(character_details)

        retrieved_text = ""
        retrieved_token_count = 0

        if chatbot_manager.vectorstore and chatbot_manager.qa_chain:
            # 🔹 Retrieve context **using only the user query**
            retrieved_docs_with_scores = chatbot_manager.vectorstore.similarity_search_with_score(user_input, k=5)

            # Extract text and similarity scores
            retrieved_texts = []
            for doc, score in retrieved_docs_with_scores:
                retrieved_texts.append(f"[Score: {score:.2f}] {doc.page_content}")

            retrieved_text = "\n\n".join(retrieved_texts)

            # 🔹 Count tokens in retrieved text
            retrieved_token_count = count_tokens(retrieved_text)
            logger.info(f"Retrieved context tokens: {retrieved_token_count}")
            logger.info(f"Retrieved Text with Scores:\n{retrieved_text}")

        # 🔹 Pass full prompt + retrieved context to the LLM
        final_prompt = f"{character_details}\n\nRetrieved Context:\n{retrieved_text}"
        response = chatbot_manager.llm.invoke(final_prompt).content

        # 🔹 Count output tokens
        full_output_tokens = count_tokens(response)

        # 🔹 Log token usage
        log_entry = (
            f"Input Tokens: {input_tokens}, Retrieved Tokens: {retrieved_token_count}, "
            f"Full Output Tokens: {full_output_tokens}, Total Tokens: {input_tokens + retrieved_token_count + full_output_tokens}\n"
        )
        with open("token_log.txt", "a") as log_file:
            log_file.write(log_entry)

        # 🔹 Store response in chat history
        messages.append({"role": "assistant", "content": response})
        logger.info(response)

        return response

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "I'm having trouble responding right now. Please try again."

