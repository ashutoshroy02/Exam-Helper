from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
import google.generativeai as genai
from PIL import Image
from st_multimodal_chatinput import multimodal_chatinput
import base64,io,re,html
from langchain_mistralai import ChatMistralAI
import requests as r
import streamlit as st


try:
    from langchain_pinecone import PineconeVectorStore
except:
    import os
    os.system("pip install langchain-pinecone -U")


st.set_page_config(page_title="Bhala Manus", page_icon="🌟")


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
</style>

"""
st.markdown(css, unsafe_allow_html=True)

# HTML content
html = """
<section id="up"></section>
<section id="down"></section>
<section id="left"></section>
<section id="right"></section>
"""


def render_frontend():
    st.markdown(css, unsafe_allow_html=True)
    st.markdown(html, unsafe_allow_html=True)

st.markdown("""
<style>
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
    margin: 8px 0; /* General spacing for top and bottom */
    font-family: "Arial", sans-serif;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15); /* Subtle shadow for depth */
}

/* User messages (Dark with 60% opacity, right indent) */
.stChatMessage > div.user {
    background-color: rgba(13, 9, 10, 0.6); /* Smoky Black with 60% opacity */
    color: #EAF2EF; /* Mint Cream for text contrast */
    border-left: 4px solid #361F27; /* Dark Purple accent */
    margin-left: 32px; /* General left margin */
    margin-right: 64px; /* Right indent for user messages */
}

/* Assistant messages (Light Lavender with 40% opacity, left indent) */
.stChatMessage > div.assistant {
    background-color: rgba(70, 40, 90, 0.6); /* Royal purple for assistant messages */
    color: #F5D7FF; /* Lavender text for a softer, friendly vibe */
    border-left: 4px solid #BB8FCE;
    margin-left: 64px; /* Left indent for assistant messages */
    margin-right: 32px; /* General right margin */
}

/* Hover effects for smooth interaction */
.stChatMessage > div:hover {
    transform: scale(1.02);
    transition: transform 0.2s ease-in-out;
    filter: brightness(1.1); /* Slight brightness boost */
}

</style>
""", unsafe_allow_html=True)

# Header and message below it
st.markdown('<div class="header">🌟No Back Abhiyan </div>', unsafe_allow_html=True)
st.markdown('<p style="color: #dcfa2f; font-size: 18px; text-align: center;">Padh le yaar...</p>', unsafe_allow_html=True)

# Sidebar and configuration
st.sidebar.markdown("""<h3 style="color: cyan;">Configuration</h3>""", unsafe_allow_html=True)
index_name = st.sidebar.selectbox( "Doc Name", options=["cns-docs","dbms-docs","pma-docs", "ml-docs"], index=0, help="Select the name of the Documents to use." )
groq_api_key = st.sidebar.text_input("LLM API Key", type="password", help="Enter your groq API key.")
model = st.sidebar.selectbox("Select Model",options=["llama-3.3-70b-versatile","llama-3.1-70b-versatile","llama-3.1-8b-instant","llama-3.2-90b-vision-preview"],
    index=0,help="Select the model to use for LLM inference.")
if not groq_api_key:
    st.sidebar.markdown("<p style='color: #f44336;'>Please enter the LLM API key to proceed!</p>", unsafe_allow_html=True)
    st.warning("Please enter the LLM API key to proceed!")

use_web = st.sidebar.checkbox("Allow Internet Access", value=True)
use_vector_store = st.sidebar.checkbox("Use Documents", value=True)
use_chat_history = st.sidebar.checkbox("Use Chat History (Last 2 Chats)", value=False)


if use_chat_history:
    use_vector_store, use_web = False, False




# Instructions
st.sidebar.markdown("""
---
**Instructions:**  
Get your *Free-API-Key*  
From **[Groq](https://console.groq.com/keys)**

--- 
Kheliye *meating - meeting*
""")

# API Key Connection to Vector Database with feedback
if "vector_store" not in st.session_state and groq_api_key:
    pc = Pinecone(api_key="pcsk_6KAu86_9Zzepx9S1VcDmLRmBSUUNpPf4JRbE4BaoVmk36yW9R4nkjutPiZ3AjZvcyL4MVx")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyARa0MF9xC5YvKWnGCEVI4Rgp0LByvYpHw")
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    st.session_state["index_name"]=index_name
    st.session_state["vector_store"] = vector_store
    st.success(f"Successfully connected to the Vector Database:- {index_name}! let's go...")
else:
    vector_store = st.session_state.get("vector_store", None)


if "index_name" in st.session_state and st.session_state["index_name"]!=index_name:
    pc = Pinecone(api_key="pcsk_6KAu86_9Zzepx9S1VcDmLRmBSUUNpPf4JRbE4BaoVmk36yW9R4nkjutPiZ3AjZvcyL4MVx")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyARa0MF9xC5YvKWnGCEVI4Rgp0LByvYpHw")
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    st.session_state["index_name"]=index_name
    st.session_state["vector_store"] = vector_store
    st.success(f"Successfully connected to the Vector Database:- {index_name}! let's go...")

# LLM API Key check
if groq_api_key:
    if "llm" not in st.session_state:
        llm = ChatGroq(temperature=0.2, model=model, api_key=groq_api_key)
        st.session_state["llm"] = llm
        st.session_state["model"]=model
        st.session_state["api_key"]=groq_api_key
    else:
        llm = st.session_state["llm"]
    
if "api_key" in st.session_state and "model" in st.session_state:
    if groq_api_key != st.session_state["api_key"] or model != st.session_state["model"]:
        llm = ChatGroq(temperature=0.2, model=model, api_key=groq_api_key)
        st.session_state["llm"] = llm
        st.session_state["model"]=model
        st.session_state["api_key"]=groq_api_key

llmx=ChatMistralAI(model="mistral-large-latest",temperature=0.3,api_key="r1u9jBlZye7QrH3ymxkJjAMVd4VLoSEA")


def display_chat_history():
    for message in st.session_state.messages:
        if message["role"]=="user":
            with st.chat_message(message["role"],avatar="👼"):
                st.write(message["content"])
        else:
            with st.chat_message(message["role"],avatar="🧑‍🏫"):
                st.write(message["content"])

# Function to Clean RAG Data
def clean_rag_data(query, context, llm):
    system = """
        You are a Highly capable Proffesor of understanding the value and context of both user queries and given data. 
        Your Task for Documents Data is to analyze the list of document's content and properties and find the most important information regarding user's query.
        Your Task for ChatHistory Data is to analyze the given ChatHistory and then provide a ChatHistory relevant to user's query.
        Your Task for Web Data is to analyze the web scraped data then summarize only useful data regarding user's query.
        You Must adhere to User's query before answering.
        
        Output:
            For Document Data
                Conclusion:
                    ...
            For ChatHistory Data
                    User: ...
                    ...
                    Assistant: ...
            For Web Data
                Web Scarped Data:
                ...
    """

    user = """{context}
            User's query is given below:
            {question}
    """

    filtering_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("user", user)
        ]
    )

    filtering_chain = filtering_prompt | llm | StrOutputParser()

    response = filtering_chain.invoke({"context": context, "question": query})

    return response

# Function to Get LLM Data
def get_llm_data(query, llm):
    system = """
        You are a knowledgeable and approachable Computer Science professor with expertise in a wide range of topics.
        Your role is to provide clear, easy, and engaging explanations to help students understand complex concepts.
        When answering:
        - Make it sure to provide the calculations, regarding the solution if there are any.
        - Start with a high-level overview, then dive into details as needed.
        - Use examples, analogies, or step-by-step explanations to clarify ideas.
        - Ensure your answers are accurate, well-structured, and easy to follow.
        - If you don’t know the answer, acknowledge it and suggest ways to explore or research further.
    """

    user = """{query}
    """

    filtering_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("user", user)
        ]
    )

    filtering_chain = filtering_prompt | llm | StrOutputParser()

    response = filtering_chain.invoke({"query": query})

    return response

# Function to Get Context Based on Input Query
def get_context(query):
    context = ""

    if use_vector_store:
        with st.spinner(":green[Extracting Data From VectorStore...]"):
            result = "\n\n".join([_.page_content for _ in vector_store.similarity_search(query, k=4)])
            clean_data = clean_rag_data(query, f"Documents Data \n\n{result}", llmx)
            context += f"Documents Data: \n\n{clean_data}"

    if use_chat_history:
        with st.spinner(":green[Extracting Data From ChatHistory...]"):
            last_messages = st.session_state.messages[:-3][-5:]
            chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in last_messages])
            clean_data = clean_rag_data(query, f"\n\nChat History \n\n{chat_history}", llmx)
            context += f"\n\nChat History: \n\n{clean_data}"

    try:
        if use_web:
            with st.spinner(":green[Extracting Data From web...]"):
                search = DuckDuckGoSearchRun()
                clean_data = clean_rag_data(query, search.invoke(query), llm)
                context += f"\n\nWeb Data:\n{clean_data}"
    except:
        pass

    if not use_chat_history:
        with st.spinner(":green[Extracting Data From ChatPPT...]"):
            context += f"\n\n LLM Data {get_llm_data(query, llm)}"

    return context

# Function to Respond to User Based on Query and Context
def respond_to_user(query, context, llm):
    system_prompt = """
    You are a specialized proffesor of Computer Science Engg. Your job is to answer the given question based on the following types of context: 

    1. **Web Data**: Information retrieved from web searches.
    2. **Documents Data**: Data extracted from documents (e.g., research papers, reports).
    3. **Chat History**: Previous interactions or discussions in the current session.
    4. **LLM Data**: Insights or completions provided by the language model.

    When answering:
    - When Answering include all important information , as well as key points
    - Make it sure to provide the calculations, regarding the solution if there are any.
    - Ensure your response is clear and easy to understand and remember even for a naive person.
    """
    user_prompt = """Question: {question} 
    Context: {context} """

    rag_chain_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", user_prompt)
        ]
    )

    rag_chain = rag_chain_prompt | llm | StrOutputParser()

    response = rag_chain.invoke({"question": query, "context": context})

    return response

def yT_transcript(link):
    url = 'https://youtubetotranscript.com/transcript'
    payload = {
        'youtube_url': link
    }
    response = r.post(url, data=payload).text
    return ' '.join([html.unescape(i) for i in re.findall(r'class="transcript-segment"[^>]*>\s*([\S ]*?\S)\s*<\/span>', response)])


def process_youtube(video_id, original_text):
    transcript = yT_transcript(f'https://www.youtube.com/watch?v={video_id}')
    if len(transcript) == 0:
        raise IndexError
    system_prompt = """
You are Explainer Bot, a highly intelligent and efficient assistant designed to analyze YouTube video transcripts and respond comprehensively to user queries. You excel at providing explanations tailored to the user’s needs, whether they seek examples, detailed elaboration, or specific insights.

**Persona:**
- You are approachable, insightful, and skilled at tailoring responses to diverse user requests.
- You aim to provide explanations that capture the essence of the video, ensuring a balance between clarity and depth.
- Your tone is clear, neutral, and professional, ensuring readability and understanding for a broad audience.

**Task:**
1. Analyze the provided video transcript, which may contain informal language, repetitions, or filler words. Your job is to:
   - Address the user’s specific query, such as providing examples, detailed explanations, or focused insights.
   - Retain the most critical information and adapt your response style accordingly.
2. If the video includes technical or specialized content, provide brief context or explanations where necessary to enhance comprehension.
3. Maintain an organized structure using bullet points, paragraphs, or sections based on the user’s query.

**Additional Inputs:**
- When answering:
  - If the user requests examples, include relevant examples or anecdotes from the transcript or generate illustrative examples.
  - If the user requests a detailed explanation, expand on the key points, ensuring no critical information is lost.
  - If the user’s query requires a summary, condense the content into a clear, concise explanation while retaining the key messages.
  - Always address the user’s specific needs while keeping the overall purpose of the video in focus.

**Output Style:**
- Always respond using **Markdown** format, avoiding LaTeX or any other non-Markdown formatting.
  - Avoid using any LaTeX symbols or complex formatting.
  - Ensure your response is easy to read and compatible with a frontend that supports Markdown.
- Tailor the response to the user’s request:
  - Provide examples when explicitly asked or when its available in transcript.
  - Offer detailed and comprehensive explanations if required.
  - Keep summaries Comprehensive and focused if brevity is requested.
- Use simple, clear sentences to cater to a broad audience.
- Avoid jargon unless it is crucial to the video's context, and provide a brief explanation if used.
- Always Answer in English only.

Act as a skilled Professor, ensuring accuracy, brevity, and clarity while retaining the original context and intent of the video. Adjust your tone and structure to match the user’s specific query and expectations.
"""

    user_prompt = """
Transcription:
{transcription}

User's Query:
{query}
"""

    rag_chain_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", user_prompt)
        ]
    )

    rag_chain = rag_chain_prompt | llmx | StrOutputParser()

    response = rag_chain.invoke({"transcription": transcript, "query": original_text})

    return response


def img_to_ques(img,query,model="gemini-1.5-flash"):
    genai.configure(api_key="AIzaSyBGMk5yhUdGv-Ph5P6Y5rq7F3G56GQJbaw")
    model = genai.GenerativeModel(model)
    prompt = f"""Analyze the provided image and the user's query: "{query}". Based on the content of the image:

1. Extract the question from the image, if user wants to asks more question add it to the Question Section.
2. For any tabular , structured data or mcq or anyother relevant information present in the image, provide it in the "Relevant Information" section.

Format your response as follows:

Question:  
[Generated question based on the image and query]  

Relevant Information:  
[Include any tabular data, key details, or insights relevant to solving the problem. Ensure structured data is presented in an easily readable format.]

"""
    return model.generate_content([prompt, img]).text

if "messages" not in st.session_state:
    st.session_state.messages = []

display_chat_history()
if groq_api_key:
    with st.container():
        user_inp = multimodal_chatinput()
    with st.container():
        if user_inp:
            video_id=""
            question=""
            if user_inp["images"]:
                b64_image=user_inp["images"][0].split(",")[-1]
                image = Image.open(io.BytesIO(base64.b64decode(b64_image)))
                try:
                    question = img_to_ques(image, user_inp["text"])
                except:
                    question = img_to_ques(image, user_inp["text"],"gemini-2.0-flash-exp")
                soln=re.findall(r"(?:https://www\.youtube\.com/watch\?v=([^&\n]+))?(?:https://youtu.be/([^\?\n]+))?",user_inp["text"])
                video_id=soln[0]
                if video_id != ["",""] :
                    if video_id[0] == "":
                        video_id=video_id[1]
                    else:
                        video_id=video_id[0]
                else:
                    video_id=""
                user_inp["text"]=""
            if not video_id:
                soln=re.findall(r"(?:https://www\.youtube\.com/watch\?v=([^&\n]+))?(?:https://youtu.be/([^\?\n]+))?",user_inp["text"])
                video_id=soln[0]
                if video_id != ["",""] :
                    if video_id[0] == "":
                        video_id=video_id[1]
                    else:
                        video_id=video_id[0]
                else:
                    video_id=""
            st.session_state.messages.append({"role": "user", "content": question+user_inp["text"]})
            with st.spinner(":green[Processing Youtube Video]"):
                if video_id:
                    st.success(f"!! Youtube Link Found:- {video_id} , Summarizing Video")
                    try:
                        yt_response=process_youtube(video_id,question+user_inp["text"])
                    except Exception as e:
                        yt_response=f"Unable to Process , Youtube Video Due to Transcript not available Error: {e}"
                    st.session_state.messages.append({"role": "assistant", "content": yt_response})
                    with st.chat_message("user",avatar="👼"):
                        st.write(question+user_inp["text"])
                    with st.chat_message("assistant",avatar="🧑‍🏫"):
                        st.write(yt_response)

            if not video_id:
                context = get_context(question+user_inp["text"])
                with st.spinner(":green[Combining jhol jhal...]"):
                    assistant_response = respond_to_user(question+user_inp["text"], context, llm)
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})

                with st.chat_message("user",avatar="👼"):
                    st.write(question+user_inp["text"])
                with st.chat_message("assistant",avatar="🧑‍🏫"):
                    st.write(assistant_response)
