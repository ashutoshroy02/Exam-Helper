# 🌟 Bhala Manus: No BackLog Abhiyan 🌟

<p align="center">
  <img src="" alt="Project Logo">
</p>

> Padh le yaar...


### [**Hosted App**](https://good-boy.streamlit.app/)

**Bhala Manus** is an AI-powered study assistant designed to help you ace your computer science courses without the fear of backlogs. It leverages advanced language models, vector databases, and web search capabilities to provide you with comprehensive and easy-to-understand explanations, making learning a breeze.

## ✨ Features

*   **Context-Aware Responses:** Gets information from multiple sources:
    *   **Web Data:** Fetches relevant information from the internet.
    *   **Documents Data:** Extracts and summarizes data from provided documents.
    *   **Chat History:** Remembers previous interactions to maintain context.
    *   **LLM Data:** Utilizes the power of large language models for insightful answers.
*   **Document Support:** Choose between different Vectorstore(Documents) for focused learning.
*   **Multiple LLMs:** Choose between the different versions of Llama3 Models.
*   **Customizable:**
    *   **Internet Access:** Enable or disable web search.
    *   **Chat History:** Toggle the use of previous chat history.
*   **Image-Based Question Answering:** Upload images of questions or diagrams, and the app will analyze them to provide relevant answers.
*   **User-Friendly Interface:** Beautiful and intuitive Streamlit-based UI.
*   **Animated Gradient Background:** Because who doesn't love a bit of flair?

## ⚙️ Installation

1. **Clone the repository:**

    ```bash
    https://github.com/IsNoobgrammer/Exam-Helper.git
    cd Exam-Helper
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Get Your API Keys:**
    *   **Groq API Key:** Create a free account on [Groq](https://console.groq.com/keys) to get your API key.

## 🚀 Usage

1. **Run the Streamlit app:**

    ```bash
    streamlit run main.py
    ```

2. **Configure:**
    *   Enter your **Groq API key** in the sidebar.
    *   Select the **model** you want to use for LLM inference.
    *   Choose your preferred **document set** (if using vector store).
    *   Enable/disable **Internet access** and **chat history**.

3. **Start Learning!**
    *   Type your questions in the chat input.
    *   Upload images containing questions or diagrams.
    *   Get comprehensive answers powered by AI.

## 🛠️ Dependencies

*   **LangChain:** Framework for building LLM applications.
*   **Groq:** For accessing powerful language models like Llama-3.
*   **Pinecone:** For vector database functionalities (if enabled).
*   **Google Generative AI:** For embeddings.
*   **DuckDuckGo Search:** For web search capabilities.
*   **Streamlit:** For creating the user interface.
*   **Pillow (PIL):** For image processing.

## 🙏 Acknowledgements

*   The creators of the awesome libraries and services used in this project.
*   Anyone who has ever struggled with backlogs (you are not alone!).

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

<p align="center">Made with ❤️ by IsNoobGrammer and Co.</p>
