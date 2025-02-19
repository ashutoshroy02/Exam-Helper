# 🌟 Bhala Manus: No BackLog Abhiyan 🌟

<p align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT81LIjm080VFPYpMizkeSMnjfNENB0poYT8Q&s" alt="Project Logo">
</p>

> Padh le yaar... (Study hard, buddy!)

### [**Hosted App**](https://good-boy.streamlit.app/)

**Bhala Manus** is your AI-powered study companion, meticulously crafted to help you conquer your computer science courses and banish the fear of backlogs. This powerful tool integrates cutting-edge language models, a robust vector database, and real-time web search to deliver comprehensive, digestible explanations that make learning a joyful experience.

## ✨ Key Features

*   **Multi-Source Contextual Understanding:** Bhala Manus intelligently gathers information from various sources to provide the most relevant and accurate answers:
    *   **Web Data:** Real-time information retrieval from the internet.
    *   **Documents Data:** Extracts key information from your chosen document sets.
    *   **Chat History:** Maintains context by remembering previous interactions (optional).
    *   **LLM Data:** Harnesses the power of large language models for insightful responses.
*   **Document Versatility:** Choose between different Vectorstore(Documents) for focused learning on specific topics like *cns, dbms, pma and ml*
*   **Advanced LLM Selection:** Choose between various versions of the Llama 3 Models, each optimized for different needs.
*   **Enhanced Customization:**
    *   **Internet Access:** Decide whether to allow web search during the session.
    *   **Chat History:** Choose to enable or disable the use of previous chat history.
*   **Multimedia Query Support:**
    *  **Image-Based Question Answering:** Upload images of questions, diagrams, or even complex tables and get analyzed, detailed responses.
    *  **YouTube Video Summarization:** Simply input a YouTube link and the tool will provide a comprehensive summary of the video, tailored to your queries, and give transcriptions as well.
*   **Streamlined User Experience:**
    *   Intuitive and visually appealing Streamlit-based user interface.
    *   Engaging animated gradient background to make learning fun.

## ⚙️ Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/IsNoobgrammer/Exam-Helper.git
    cd Exam-Helper
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Acquire API Keys:**
    *   **Groq API Key:** Sign up for a free account on [Groq](https://console.groq.com/keys) to obtain your unique API key.

## 🚀 Usage

1.  **Launch the Streamlit app:**

    ```bash
    streamlit run main.py
    ```

2.  **Configure Your Environment:**
    *   Enter your **Groq API key** in the sidebar.
    *   Select your preferred **Llama 3 model** for LLM inference.
    *   Choose your specific **document set** from the dropdown menu.
    *   Toggle **Internet access** and **chat history** according to your preference.

3.  **Dive into Learning!**
    *   Type in your questions, upload images, or input YouTube links.
    *   Experience the power of AI as it delivers tailored responses.
    *   Make sure to ask *meating-meeting* Questions.

## 🛠️ Dependencies

*   **LangChain:** A versatile framework for building sophisticated LLM-powered applications.
*   **Groq:** For accessing cutting-edge language models like the Llama 3 series.
*   **Pinecone:** For robust vector database management (if using document sets).
*   **Google Generative AI:** For high-performance embedding services.
*   **DuckDuckGo Search:** For fetching real-time data from the web.
*   **Streamlit:** For building a responsive and engaging user interface.
*   **Pillow (PIL):** For advanced image processing capabilities.
*   **YouTube Transcript API:** To extract transcriptions from the YouTube video for summaries.
*  **st-multimodal-chatinput:** For handling multi-modal inputs like images.


## 🙏 Acknowledgements

*   We extend our sincere gratitude to the creators of the fantastic libraries and services that power this project.
*   To everyone who has ever faced a backlog - this app is for you!

## 🤝 Contributing

We welcome contributions! Feel free to submit issues or open pull requests to improve this project.

### Current Contributors

We appreciate the valuable contributions of the following developers:

<a href="https://github.com/IsNoobgrammer/Exam-Helper/contributors">
  <img src="https://contrib.rocks/image?repo=IsNoobgrammer/Exam-Helper" />
</a>

## 📄 License

This project is licensed under the Apache License - see the `LICENSE` file for details.

---

<p align="center">Made with ❤️ by IsNoobGrammer and Co.</p>
