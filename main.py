from langchain_mistralai import ChatMistralAI
from st_multimodal_chatinput import multimodal_chatinput
from streamlit_carousel import carousel
import streamlit as st
from PIL import Image
import base64
import io
import re

import utils

# Streamlit Configuration and Styling
st.set_page_config(page_title="Bhala Manus", page_icon="🌟")

# Define CSS styles
css = """
<style>
/* Make header background transparent */
.stApp > header {
    background-color: transparent !important;
}

/* Apply animated gradient background */
.stApp {
    background: linear-gradient(45deg, #3a5683 10%, #0E1117 45%, #0E1117 55%, #3a5683 90%);
    animation: gradientAnimation 20s ease infinite;
    background-size: 200% 200%;
    background-attachment: fixed;
}

/* Keyframes for smooth animation */
@keyframes gradientAnimation {
    0% {
        background-position: 0% 0%;
    }
    50% {
        background-position: 100% 100%;
    }
    100% {
        background-position: 0% 0%;
    }
}

.main {
    font-family: 'Arial', sans-serif;
    background-color: #454545;
    color: #fff;
}
.header {
    text-align: center;
    color: #47fffc;
    font-size: 36px;
    font-weight: bold;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    font-size: 16px;
}
.stTextInput>div>input {
    background-color: #ffffff;
    border-radius: 10px;
    border: 1px solid #4CAF50;
    padding: 10px;
    font-size: 16px;
}
.stCheckbox>div>label {
    font-size: 16px;
    color: #4CAF50;
}
.stChatInput>div>input {
    background-color: #e8f5e9;
    border: 1px solid #81c784;
}
.stMarkdown {
    font-size: 16px;
}

.stChatMessage > div {
    border-radius: 12px;
    padding: 12px;
    margin: 8px 0;
    font-family: "Arial", sans-serif;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
}

/* User messages */
.stChatMessage > div.user {
    background-color: rgba(13, 9, 10, 0.6);
    color: #EAF2EF;
    border-left: 4px solid #361F27;
    margin-left: 32px;
    margin-right: 64px;
}

/* Assistant messages */
.stChatMessage > div.assistant {
    background-color: rgba(70, 40, 90, 0.6);
    color: #F5D7FF;
    border-left: 4px solid #BB8FCE;
    margin-left: 64px;
    margin-right: 32px;
}

/* Hover effects */
.stChatMessage > div:hover {
    transform: scale(1.005);
    transition: transform 0.2s ease-in-out;
}
</style>
"""

# Apply CSS
st.markdown(css, unsafe_allow_html=True)

# Header and message
st.markdown(
    '<div class="header">🌟No Back Abhiyan </div>', unsafe_allow_html=True
)
st.markdown(
    '<p style="color: #dcfa2f; font-size: 18px; text-align: center;">Padh le yaar...</p>',
    unsafe_allow_html=True,
)

# Sidebar configuration
st.sidebar.markdown(
    """<h3 style="color: cyan;">Configuration</h3>""", unsafe_allow_html=True
)

# Sidebar inputs
index_name = st.sidebar.selectbox(
    "Doc Name", 
    options=["cc-docs", "ann-docs", "dbms-docs"], 
    index=0, 
    help="Select the name of the Documents to use."
)

groq_api_key = st.sidebar.text_input(
    "LLM API Key", 
    type="password", 
    help="Enter your groq API key."
)

model = st.sidebar.selectbox(
    "Select Model",
    options=[
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.2-90b-vision-preview",
    ],
    index=0,
    help="Select the model to use for LLM inference.",
)

# Check for API key
if not groq_api_key:
    st.sidebar.markdown(
        "<p style='color: #f44336;'>Please enter the LLM API key to proceed!</p>",
        unsafe_allow_html=True,
    )
    st.warning("Please enter the LLM API key to proceed!")
    st.write('''**Find your Key [Groq](https://console.groq.com/keys)**''')
    st.stop()  # Stop execution if no API key

# Configuration checkboxes
use_web = st.sidebar.checkbox("Allow Internet Access", value=True)
use_vector_store = st.sidebar.checkbox("Use Documents", value=True)
use_chat_history = st.sidebar.checkbox(
    "Use Chat History (Last 2 Chats)", value=False
)

if use_chat_history:
    use_vector_store, use_web = False, False

# Instructions
st.sidebar.markdown(
    """
---
**Instructions:**  
Get your *Free-API-Key*  
From **[Groq](https://console.groq.com/keys)**

--- 
Kheliye *meating - meeting*
"""
)

# API keys dictionary
api_keys = {
    "pinecone": "pcsk_6KAu86_9Zzepx9S1VcDmLRmBSUUNpPf4JRbE4BaoVmk36yW9R4nkjutPiZ3AjZvcyL4MVx",
    "google": "AIzaSyARa0MF9xC5YvKWnGCEVI4Rgp0LByvYpHw",
    "groq": groq_api_key,
}

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""

# Initialize vector store
@st.cache_resource
def initialize_vector_store(index_name, api_keys):
    try:
        return utils.get_vector_store(index_name, api_keys)
    except Exception as e:
        st.error(f"Error initializing vector store: {e}")
        return None

# Initialize LLM
@st.cache_resource
def initialize_llm(model, api_keys):
    try:
        return utils.get_llm(model, api_keys)
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None

# Get or initialize vector store
vector_store = initialize_vector_store(index_name, api_keys)
if vector_store:
    st.success(f"Successfully connected to the Vector Database: {index_name}!")

# Get or initialize LLM
llm = initialize_llm(model, api_keys)

# Fallback model
try:
    llmx = ChatMistralAI(
        model="mistral-large-latest",
        temperature=0.3,
        api_key="RScM7WQKY4RtCVOOj49MWYqRVQB3zl9Y",
    )
except Exception as e:
    st.error(f"Error initializing fallback model: {e}")
    llmx = None

# Function to display chat history
def display_chat_history():
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message(message["role"], avatar="👼"):
                st.write(message["content"])
        else:
            with st.chat_message(message["role"], avatar="🧑‍🏫"):
                st.write(message["content"])

# Display existing chat history
display_chat_history()

# Main chat interaction
if groq_api_key and llm:
    try:
        # Get user input
        user_inp = multimodal_chatinput()
        
        if user_inp:
            # Check for duplicate queries
            current_query = str(user_inp.get("text", "")) + str(len(user_inp.get("images", [])))
            if current_query == st.session_state["last_query"]:
                st.stop()
            else:
                st.session_state["last_query"] = current_query
            
            # Initialize variables
            video_id = ""
            question = ""
            
            # Process images if present
            if user_inp.get("images"):
                try:
                    b64_image = user_inp["images"][0].split(",")[-1]
                    image = Image.open(io.BytesIO(base64.b64decode(b64_image)))
                    
                    # Extract question from image
                    try:
                        question = utils.img_to_ques(image, user_inp.get("text", ""))
                    except Exception as e:
                        st.warning(f"Error processing image: {e}")
                        try:
                            question = utils.img_to_ques(image, user_inp.get("text", ""), "gemini-2.0-flash-exp")
                        except Exception as e2:
                            st.error(f"Failed to process image with fallback: {e2}")
                            question = user_inp.get("text", "")
                    
                    # Look for YouTube links in text
                    youtube_pattern = r"(?:https://www\.youtube\.com/watch\?v=([^&\n]+))?(?:https://youtu.be/([^\?\n]+))?"
                    matches = re.findall(youtube_pattern, user_inp.get("text", ""))
                    for match in matches:
                        video_id = match[0] or match[1]
                        if video_id:
                            break
                    
                    user_inp["text"] = ""
                except Exception as e:
                    st.error(f"Error processing image input: {e}")
                    question = user_inp.get("text", "")
            
            # Look for YouTube links if not found in image processing
            if not video_id and user_inp.get("text"):
                youtube_pattern = r"(?:https://www\.youtube\.com/watch\?v=([^&\n]+))?(?:https://youtu.be/([^\?\n]+))?"
                matches = re.findall(youtube_pattern, user_inp["text"])
                for match in matches:
                    video_id = match[0] or match[1]
                    if video_id:
                        break
            
            # Combine question and text
            full_query = question + user_inp.get("text", "")
            
            # Add user message to chat history
            st.session_state.messages.append(
                {"role": "user", "content": full_query}
            )
            
            # Check for diagram requirement
            if llmx:
                try:
                    with st.spinner("🔍 Checking if diagram is needed..."):
                        diagram_required = utils.check_for_diagram(full_query, llmx)
                    
                    if diagram_required.requires_diagram:
                        with st.spinner("📊 Generating diagram..."):
                            try:
                                images = utils.search_images(diagram_required.search_query, 5)
                                if images:
                                    carousel(images, fade=True, wrap=True, interval=999000)
                                else:
                                    st.info("No relevant diagrams found.")
                            except Exception as e:
                                st.warning(f"Unable to generate diagram: {e}")
                    else:
                        st.info("No diagram required for this query.")
                except Exception as e:
                    st.warning(f"Error checking diagram requirement: {e}")
            
            # Process YouTube video if found
            if video_id:
                with st.spinner("🎥 Processing YouTube video..."):
                    st.success(f"YouTube video found: {video_id}")
                    try:
                        if llmx:
                            yt_response = utils.process_youtube(video_id, full_query, llmx)
                        else:
                            yt_response = "Unable to process YouTube video: Fallback model not available."
                    except Exception as e:
                        yt_response = f"Unable to process YouTube video: {e}"
                    
                    # Add response to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": yt_response}
                    )
                    
                    # Display messages
                    with st.chat_message("user", avatar="👼"):
                        st.write(full_query)
                    with st.chat_message("assistant", avatar="🧑‍🏫"):
                        st.write(yt_response)
            
            # Process regular query (non-YouTube)
            else:
                try:
                    # Get context
                    context = utils.get_context(
                        full_query,
                        use_vector_store,
                        vector_store,
                        use_web,
                        use_chat_history,
                        llm,
                        llmx,
                        st.session_state.messages,
                    )
                    
                    # Generate response
                    with st.spinner("🤔 Thinking..."):
                        assistant_response = utils.respond_to_user(full_query, context, llm)
                    
                    # Add response to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": assistant_response}
                    )
                    
                    # Display messages
                    with st.chat_message("user", avatar="👼"):
                        st.write(full_query)
                    with st.chat_message("assistant", avatar="🧑‍🏫"):
                        st.write(assistant_response)
                        
                except Exception as e:
                    st.error(f"Error processing query: {e}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Sorry, I encountered an error: {e}"}
                    )
    
    except Exception as e:
        st.error(f"Unexpected error in main chat loop: {e}")
        st.write("Please refresh the page and try again.")

elif not groq_api_key:
    st.info("Please enter your Groq API key in the sidebar to start chatting.")
else:
    st.error("Failed to initialize the language model. Please check your API key and try again.")
