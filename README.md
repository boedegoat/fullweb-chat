# FullWebChat

FullWebChat is an AI-powered application that enables you to have intelligent conversations about any website's content. By crawling and processing entire websites, it creates a knowledge base that you can query through natural language. Built on RAG (Retrieval-Augmented Generation) technology, FullWebChat delivers accurate, contextually relevant answers based on the actual content of crawled sites.

## Features

-   **Entire Website Crawling**: Automatically extract and process content from any website using sitemap URLs
-   **Vector Search**: Find relevant information quickly using semantic search powered by pgvector
-   **Interactive Chat**: Have natural conversations about website content with AI-powered responses
-   **User-friendly Interface**: Clean Streamlit UI for easy website management and querying
-   **Multi-website Support**: Manage and query multiple websites from a single interface

## Installation

### Prerequisites

-   Python 3.9+
-   Conda
-   Supabase account
-   OpenAI API key

### Setting Up Conda Environment

```bash
# Create a new conda environment
conda create -n fullwebchat python=3.10
conda activate fullwebchat

# Clone the repository
git clone https://github.com/boedegoat/fullweb-chat.git
cd fullweb-chat

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

1. Copy the example environment file:

    ```bash
    cp .env.example .env
    ```

2. Edit the `.env` file with your credentials:

    ```bash
    OPENAI_API_KEY=your_openai_api_key
    SUPABASE_URL=your_supabase_project_url
    SUPABASE_SERVICE_KEY=your_supabase_service_key
    LLM_MODEL=gpt-4o-mini  # Or your preferred OpenAI model
    ```

### Supabase Setup

1. Copy the contents of `site_pages.sql`
2. Execute the SQL in the SQL Editor

### Running the Application

Start the Streamlit application:

```bash
streamlit run streamlit_ui.py
```

## Using FullWebChat

1. Use the sidebar to add a new website by providing:
    - A name for the website (e.g., "python_docs")
    - The sitemap URL (e.g., "https://docs.python.org/sitemap.xml")
2. Click "Crawl Website" and wait for the crawling process to complete
3. Select the website from the dropdown menu
4. Start asking questions about the website content!

## Project Structure

-   `streamlit_ui.py`: Main application UI built with Streamlit
-   `website_expert.py`: Agent definition and tools for searching website content
-   `crawl_website.py`: Website crawling and content processing utilities
-   `site_pages.sql`: Supabase database setup
-   `.env.example`: Example environment configuration
-   `requirements.txt`: Python dependencies

## How It Works

1. **Crawling**: The system uses Crawl4AI to fetch website content and convert it to markdown
2. **Processing**: Content is split into manageable chunks with title and summary extraction
3. **Embedding**: OpenAI's embedding models convert text chunks to vector representations
4. **Storage**: Chunks and embeddings are stored in Supabase with pgvector for similarity search
5. **Retrieval**: When a question is asked, relevant chunks are retrieved using vector similarity
6. **Generation**: An LLM combines retrieved content with the question to generate an accurate answer

## Credits

-   [ottomator-agents](https://github.com/coleam00/ottomator-agents/tree/main/crawl4AI-agent): Inspired the agent design and tooling
-   [Crawl4AI](https://github.com/crawl4ai/crawl4ai): Used for website crawling and content extraction
-   [pydantic-ai](https://github.com/pydantic/pydantic-ai): Used for structured agent interaction
-   [Supabase](https://supabase.com): Provides the vector database capabilities
-   [OpenAI](https://openai.com): Provides the embedding and language models
