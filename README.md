## Introduction 📚

PDFandWebsiteRAG is a Retrieval-Augmented Generation (RAG) project designed to answer questions based on content from both PDF documents and websites. It leverages technologies like `sentence-transformers` for embedding generation, `faiss` for efficient similarity search, and `gradio` for creating a user-friendly web interface.  This project allows you to easily combine information from different sources to provide more comprehensive and accurate answers.

## Features ✨

*   **PDF Support:**  Extracts text from PDF documents and indexes the content for querying.
*   **Website Content Extraction:** Scrapes text content from specified websites.
*   **RAG Pipeline:**  Combines retrieval and generation to provide context-aware answers.
*   **Semantic Search:** Uses sentence embeddings to find the most relevant information.
*   **Gradio Interface:**  Offers a simple and intuitive web interface for asking questions.
*   **Dockerized:**  Comes with a Dockerfile for easy deployment and reproducibility.
*   **Efficient Indexing:** Utilizes Faiss for fast similarity search on large datasets.

## Installation ⚙️

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Pranesh-2005/PDFandWebsiteRAG.git
    cd PDFandWebsiteRAG
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Docker (Optional):**

    To build and run the project using Docker:

    ```bash
    docker build -t pdf-website-rag .
    docker run -d -p 5000:5000 pdf-website-rag
    ```
    Then access the Gradio interface at `http://localhost:5000`.

## Usage 🚀

1.  **Prepare your data:**
    *   Place your PDF files in a designated directory (e.g., `data/pdfs`).
    *   Provide a list of website URLs to scrape.

2.  **Run the application:**

    ```bash
    python app.py
    ```

3.  **Access the Gradio interface:**

    Open your web browser and go to `http://localhost:5000` (or the port specified in `start.sh`).

4.  **Ask questions:**

    Enter your question in the text box and submit. The application will retrieve relevant information from the PDFs and websites and generate an answer.

**Example:**

Let's say you have a PDF about "Quantum Computing" and you're scraping a website about "Artificial Intelligence". You could ask:

"What are the potential applications of quantum computing in artificial intelligence?"

The application will search both sources and provide a combined answer.

## Contributing 🤝

We welcome contributions to PDFandWebsiteRAG!  Here's how you can get involved:

1.  **Fork the repository.**
2.  **Create a new branch** for your feature or bug fix.
3.  **Make your changes** and ensure they are well-documented.
4.  **Test your changes** thoroughly.
5.  **Submit a pull request** with a clear description of your changes.

Please follow these guidelines:

*   **Code Style:**  Maintain consistent code style with the existing codebase.
*   **Documentation:**  Update the documentation to reflect your changes.
*   **Testing:**  Include unit tests for any new functionality.

## License 📜

This project is licensed under the [MIT License](https://github.com/pranesh-2005/PDFandWebsiteRAG/blob/main/LICENSE) - see the [LICENSE](https://github.com/pranesh-2005/PDFandWebsiteRAG/blob/main/LICENSE) file for details.

---

**Project Files:**

*   `Dockerfile`:  Defines the Docker image for the application.
*   `app.py`:  Contains the main application logic, including data loading, embedding generation, similarity search, and answer generation.
*   `start.sh`:  A shell script to start the application using `uvicorn`.

## License
This project is licensed under the **MIT** License.

---
🔗 GitHub Repo: https://github.com/Pranesh-2005/PDFandWebsiteRAG
