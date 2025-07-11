
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
import google.generativeai as genai
from langchain_pinecone import PineconeVectorStore
from pydantic import BaseModel, Field
from duckduckgo_search import DDGS

import base64, io, re, html
from langchain_mistralai import ChatMistralAI
import requests as r
import streamlit as st
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_vector_store(index_name, api_keys):
    """Initializes and returns a Pinecone vector store."""
    try:
        pc = Pinecone(api_key=api_keys["pinecone"])
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=api_keys["google"]
        )
        index = pc.Index(index_name)
        return PineconeVectorStore(index=index, embedding=embeddings)
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        raise

def get_llm(model, api_keys):
    """Initializes and returns a ChatGroq language model."""
    try:
        return ChatGroq(temperature=0.2, model=model, api_key=api_keys["groq"])
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        raise

def clean_rag_data(query, context, llm):
    """Cleans and filters RAG data based on the query."""
    try:
        system = """
        You are a highly capable professor with expertise in understanding the value and context of both user queries and given data. 
        Your task is to analyze the provided data and find the most important information regarding the user's query.

        For Documents Data: Analyze the document content and extract the most relevant information.
        For ChatHistory Data: Analyze the chat history and provide relevant context.
        For Web Data: Analyze web scraped data and summarize only useful information.
        
        You must adhere to the user's query before answering.
        
        Output format:
        For Document Data:
            Conclusion: [relevant information]
        For ChatHistory Data:
            User: [relevant user messages]
            Assistant: [relevant assistant responses]
        For Web Data:
            Web Scraped Data: [summarized relevant information]
        """
        
        user = """Context: {context}
        User's query: {question}
        """
        
        filtering_prompt = ChatPromptTemplate.from_messages(
            [("system", system), ("user", user)]
        )
        filtering_chain = filtering_prompt | llm | StrOutputParser()
        return filtering_chain.invoke({"context": context, "question": query})
    except Exception as e:
        logger.error(f"Error cleaning RAG data: {e}")
        return f"Error processing context: {e}"

def get_llm_data(query, llm):
    """Gets a response from the LLM based on the query."""
    try:
        system = """
        You are a knowledgeable and approachable Computer Science professor with expertise in a wide range of topics.
        Your role is to provide clear, easy, and engaging explanations to help students understand complex concepts.
        
        When answering:
        - Provide calculations and solutions with step-by-step explanations if needed
        - Start with a high-level overview, then dive into details
        - Use examples, analogies, or step-by-step explanations to clarify ideas
        - Ensure your answers are accurate, well-structured, and easy to follow
        - If you don't know the answer, acknowledge it and suggest ways to explore further
        """
        
        user = "{query}"
        filtering_prompt = ChatPromptTemplate.from_messages(
            [("system", system), ("user", user)]
        )
        filtering_chain = filtering_prompt | llm | StrOutputParser()
        return filtering_chain.invoke({"query": query})
    except Exception as e:
        logger.error(f"Error getting LLM data: {e}")
        return f"Error generating response: {e}"

def get_context(query, use_vector_store, vector_store, use_web, use_chat_history, llm, llmx, messages):
    """Retrieves and processes context from various sources."""
    context = ""
    
    # Vector store data
    if use_vector_store and vector_store:
        try:
            with st.spinner("📚 Extracting data from vector store..."):
                search_results = vector_store.similarity_search(query, k=3)
                result = "\n\n".join([doc.page_content for doc in search_results])
                if result.strip():
                    clean_data = clean_rag_data(query, f"Documents Data \n\n{result}", llmx or llm)
                    context += f"Documents Data: \n\n{clean_data}"
        except Exception as e:
            logger.error(f"Error retrieving vector store data: {e}")
            st.warning(f"Error accessing document database: {e}")

    # Chat history data
    if use_chat_history and messages:
        try:
            with st.spinner("💬 Extracting data from chat history..."):
                last_messages = messages[:-1][-5:]  # Get last 5 messages excluding current
                if last_messages:
                    chat_history = "\n".join(
                        [f"{msg['role']}: {msg['content']}" for msg in last_messages]
                    )
                    clean_data = clean_rag_data(
                        query, f"\n\nChat History \n\n{chat_history}", llmx or llm
                    )
                    context += f"\n\nChat History: \n\n{clean_data}"
        except Exception as e:
            logger.error(f"Error processing chat history: {e}")
            st.warning(f"Error processing chat history: {e}")

    # Web search data
    if use_web:
        try:
            with st.spinner("🌐 Searching the web..."):
                search = DuckDuckGoSearchRun()
                search_results = search.invoke(query)
                if search_results.strip():
                    clean_data = clean_rag_data(query, search_results, llm)
                    context += f"\n\nWeb Data:\n{clean_data}"
        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            st.warning(f"Web search unavailable: {e}")

    # LLM generated data
    if not use_chat_history:
        try:
            with st.spinner("🧠 Generating additional insights..."):
                llm_data = get_llm_data(query, llm)
                context += f"\n\nLLM Data: {llm_data}"
        except Exception as e:
            logger.error(f"Error generating LLM data: {e}")
            st.warning(f"Error generating additional insights: {e}")

    return context

def respond_to_user(query, context, llm):
    """Generates a response to the user based on the query and context."""
    try:
        system_prompt = """
        You are a specialized professor of Computer Science Engineering. Your job is to answer the given question based on the provided context.

        The context may include:
        1. Web Data: Information retrieved from web searches
        2. Documents Data: Data extracted from documents and research papers
        3. Chat History: Previous interactions in the current session
        4. LLM Data: Insights provided by the language model

        When answering:
        - Include all important information and key points
        - Provide calculations and step-by-step solutions when needed
        - Ensure your response is clear and easy to understand
        - Make explanations accessible even for beginners
        - Structure your response logically with proper formatting
        """
        
        user_prompt = """Question: {question} 
        Context: {context}"""
        
        rag_chain_prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", user_prompt)]
        )
        rag_chain = rag_chain_prompt | llm | StrOutputParser()
        return rag_chain.invoke({"question": query, "context": context})
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"I apologize, but I encountered an error while generating a response: {e}"

def html_entity_cleanup(text):
    """Replace common HTML entities with their text equivalents."""
    try:
        # Replace common HTML entities
        text = html.unescape(text)
        # Additional cleanup for specific entities
        replacements = {
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#39;': "'",
            '&nbsp;': ' '
        }
        for entity, replacement in replacements.items():
            text = text.replace(entity, replacement)
        return text
    except Exception as e:
        logger.error(f"Error cleaning HTML entities: {e}")
        return text

def yT_transcript(link):
    """Fetches the transcript of a YouTube video."""
    try:
        url = "https://youtubetotranscript.com/transcript"
        payload = {"youtube_url": link}
        
        response = r.post(url, data=payload, timeout=30)
        response.raise_for_status()
        
        # Extract transcript segments
        segments = re.findall(
            r'class="transcript-segment"[^>]*>\s*([\S ]*?\S)\s*</span>', 
            response.text
        )
        
        if not segments:
            raise ValueError("No transcript segments found")
        
        # Clean and join segments
        cleaned_segments = [html_entity_cleanup(segment) for segment in segments]
        return " ".join(cleaned_segments)
    
    except Exception as e:
        logger.error(f"Error fetching YouTube transcript: {e}")
        raise

def process_youtube(video_id, original_text, llmx):
    """Processes a YouTube video transcript and answers a query."""
    try:
        transcript = yT_transcript(f"https://www.youtube.com/watch?v={video_id}")
        
        if not transcript or len(transcript.strip()) == 0:
            raise ValueError("Empty transcript received")
        
        system_prompt = """
        You are an Explainer Bot, a highly intelligent assistant designed to analyze YouTube video transcripts and respond comprehensively to user queries.

        **Your Role:**
        - Analyze video transcripts that may contain informal language, repetitions, or filler words
        - Provide explanations tailored to the user's specific needs
        - Maintain a clear, professional, and approachable tone

        **Task:**
        1. Address the user's specific query using the provided transcript
        2. If the query includes a YouTube link, focus on the video content while addressing other parts of the query
        3. Provide context for technical or specialized content when necessary
        4. Structure your response clearly using appropriate formatting

        **Guidelines:**
        - Use **Markdown** format only (no LaTeX)
        - Provide examples when requested or when they enhance understanding
        - Offer detailed explanations while maintaining clarity
        - Keep summaries comprehensive yet focused
        - Use simple, clear language accessible to a broad audience
        - Always respond in English

        Focus on accuracy, clarity, and relevance to the user's query.
        """

        user_prompt = """
        Transcription: {transcription}
        
        User's Query: {query}
        """
        
        rag_chain_prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", user_prompt)]
        )
        rag_chain = rag_chain_prompt | llmx | StrOutputParser()
        response = rag_chain.invoke({"transcription": transcript, "query": original_text})
        return response
    
    except Exception as e:
        logger.error(f"Error processing YouTube video: {e}")
        return f"I apologize, but I couldn't process the YouTube video. Error: {e}"

def img_to_ques(img, query, model="gemini-1.5-flash"):
    """Extracts a question and relevant information from an image."""
    try:
        genai.configure(api_key="AIzaSyBkssLWrVkGHVa8Z5eC2c8snijh_X8d8ho")
        model_instance = genai.GenerativeModel(model)
        
        prompt = f"""Analyze the provided image and the user's query: "{query}". 

        Based on the content of the image:
        1. Extract or formulate the main question from the image
        2. If the user wants to ask additional questions, include them in the Question section
        3. For any tabular data, structured information, MCQs, or other relevant details present in the image, provide them in the "Relevant Information" section

        Format your response as follows:

        Question:  
        [Generated question based on the image and query]  

        Relevant Information:  
        [Include any tabular data, key details, or structured information from the image that's relevant to solving the problem. Only include information that's actually visible in the image. If no relevant information is present, state "No additional structured information found in the image."]
        """
        
        response = model_instance.generate_content([prompt, img])
        return response.text
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return f"Question: {query}\n\nRelevant Information: Unable to process image due to error: {e}"

class DiagramCheck(BaseModel):
    requires_diagram: bool = Field(
        ...,
        description="True if the user's question needs a diagram or image for explanation or solution, False otherwise.",
    )
    search_query: str = Field(
        "",
        description="A relevant search query to find the required diagram or image, if needed.",
    )



class DiagramCheck(BaseModel):
    requires_diagram: bool = Field(
        ...,
        description="True if the user's question needs a diagram or image for explanation or solution, False otherwise.",
    )
    search_query: str = Field(
        "",
        description="A relevant Google search query to find the required diagram or image, if needed.",
    )

# --- Function to check for diagram requirement ---
def check_for_diagram(user_query: str, llm):
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful assistant that analyzes user questions to determine if they require a diagram or image for a better explanation or solution. Your primary goal is to assist with educational and informational queries, especially in the field of Computer Science (CSE).

                - If a diagram/image is needed, set 'requires_diagram' to True and provide a suitable 'search_query' for finding that image on a general search engine.
                - **Give special consideration to diagrams and flowcharts commonly used in Computer Science.** These are often essential for understanding algorithms, data structures, system architectures, and processes. Be lenient when identifying the need for CSE-related diagrams.
                - **The search_query should focus on finding educational, technical, or illustrative content, including relevant CSE diagrams and flowcharts.** It should never explicitly search for or suggest sexually suggestive, explicit, or NSFW (Not Safe For Work) imagery.
                - If a diagram/image is NOT needed, set 'requires_diagram' to False and leave 'search_query' empty.
                - Consider if the question involves:
                    - Visualizing structures (e.g., graphs, trees, networks, data structures)
                    - Understanding processes (e.g., flowcharts, algorithms, control flow)
                    - Comparing visual information
                    - Describing layouts, architecture, or designs (especially in a software or system context)
                    - Scientific or medical illustrations (e.g., anatomy diagrams, biological processes). These may include representations of the human body for educational purposes, but the focus must remain on the scientific or medical context.
                - **In cases where the user's query might relate to potentially sensitive topics (e.g., human anatomy) or complex CSE topics, be extremely cautious. Prioritize search queries that lead to reputable educational or scientific sources. Avoid any terms that could be interpreted as seeking explicit or inappropriate content.**
                - **Under no circumstances should the 'search_query' include terms like "nude," "naked," "sex," or any other sexually suggestive language.**

                **Examples of Acceptable Queries (for educational/scientific/CSE purposes):**
                    - "binary search tree diagram"
                    - "linked list vs array visualization"
                    - "OSI model flowchart"
                    - "CPU scheduling algorithm explained with diagram"
                    - "human heart anatomy diagram"
                    - "mitosis process illustration"
                    - "breast tissue cross-section" (in a medical/biological context)

                **Examples of Unacceptable Queries:**
                    - "nude human body"
                    - "sexy woman"
                    - "breast pictures" (without a clear medical/scientific context)

                Output JSON:
                {{
                  "requires_diagram": bool,
                  "search_query": str
                }}
                """,
            ),
            ("user", "{user_query}"),
        ]
    )

    chain = prompt_template | llm.with_structured_output(DiagramCheck)
    result = chain.invoke({"user_query": user_query})
    return result

# --- Function to perform DuckDuckGo image search ---
def search_images(query, num_images=5):
    with DDGS() as ddgs:
        results2 = [ dict(text="",title="",img=img['image'],link=img["url"]) for img in ddgs.images(query, safesearch='Off',region="en-us", max_results=num_images-2,type_image="gif") if 'image' in img]
        results = [ dict(text="",title="",img=img['image'],link=img["url"]) for img in ddgs.images(query, safesearch='Off',region="en-us", max_results=num_images) if 'image' in img]
        images = results + results2
        return images
