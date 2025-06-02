# YouTube Transcript Chatbot

A conversational AI chatbot that answers user queries based on the transcript of any YouTube video. It fetches the video transcript, embeds it into a vector store, and uses large language models (LLMs) to generate context-aware answers.

---

## Features

- Extracts and caches YouTube video transcripts (supports only English transcripts currently).
- Splits transcripts into manageable chunks using `langchain`.
- Embeds chunks using HuggingFace sentence transformers.
- Stores embeddings in a vector store (FAISS).
- Uses contextual compression retriever to enhance document retrieval.
- Generates responses using powerful LLMs (e.g., NVIDIA Llama or HuggingFace models).
- Interactive web UI built with Gradio for easy querying.

---

## Demo

Enter a YouTube video URL and ask any question about the video content. The chatbot will answer based on the transcript.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/youtube-transcript-chatbot.git
cd youtube-transcript-chatbot


## Project Structure 

├── app.py                  # Main application and Gradio UI
├── indexing.py             # Transcript extraction, splitting, vector store logic
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API keys)
└── README.md               # Project documentation

## Libraries Used

* youtube-transcript-api — For fetching YouTube video transcripts.

* langchain — Text splitting, retrieval, and prompt chaining.

* faiss — Vector similarity search.

* huggingface_hub and transformers — Embeddings and LLMs.

* gradio — Web UI.

* python-dotenv — Environment variable management.

## Contributing

Contributions, bug reports, and feature requests are welcome. Please open issues or pull requests with clear descriptions.Contributions, bug reports, and feature requests are welcome. Please open issues or pull requests with clear descriptions.