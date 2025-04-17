# FullWebChat

FullWebChat is an AI-powered web application that allows you to crawl any website, process its content, and have interactive conversations about that content using RAG (Retrieval-Augmented Generation). The system crawls web pages, chunks the content, generates embeddings, and stores everything in a Supabase database for efficient semantic search. You can then ask questions about any crawled website, and the system will provide accurate answers based on the site's content.

## Features

-   **Entire Website Crawling**: Automatically extract content from any entire websites using sitemap URLs
-   **Vector Search**: Find relevant content quickly using semantic search with pgvector
-   **Interactive UI**: User-friendly Streamlit interface for website management and chat

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
