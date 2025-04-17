from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os

import streamlit as st
import logfire
from supabase import Client, create_client
from openai import AsyncOpenAI

# Import all the message part classes
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ModelRequestPart,
    ModelResponsePart,
    UserPromptPart,
    TextPart,
)
from website_expert import website_expert, WebsiteDeps

# Import crawler functions
from crawl_website import get_website_urls, crawl_parallel, process_and_store_document

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""
    role: Literal['user', 'model']
    timestamp: str
    content: str

def display_message_part(part: ModelRequestPart | ModelResponsePart):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)


async def run_agent_with_streaming(user_input: str, website_source: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # Prepare dependencies
    deps = WebsiteDeps(
        supabase=supabase,
        openai_client=openai_client,
        website_source=website_source
    )

    # Run the agent in a stream
    async with website_expert.run_stream(
        user_input,
        deps=deps,
        message_history=st.session_state.messages[:-1],  # pass entire conversation so far
    ) as result:
        # We'll gather partial text to show incrementally
        partial_text = ""
        message_placeholder = st.empty()

        # Render partial text as it arrives
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        # Now that the stream is finished, we have a final result.
        # Add new messages from this run, excluding user-prompt messages
        filtered_messages = [msg for msg in result.new_messages() 
                            if not (hasattr(msg, 'parts') and 
                                    any(part.part_kind == 'user-prompt' for part in msg.parts))]
        st.session_state.messages.extend(filtered_messages)

        # Add the final response to the messages
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )

async def crawl_website(sitemap_url: str, website_name: str):
    """Crawl a website and store its content in the database."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Get URLs from sitemap
    status_text.text("Getting URLs from sitemap...")
    urls = await get_website_urls(sitemap_url)
    if not urls:
        status_text.text("No URLs found to crawl")
        return False
    
    # Show URLs being crawled
    total_urls = len(urls)
    status_text.text(f"Found {total_urls} URLs to crawl")
    
    # Start crawling
    crawled = 0
    
    async def update_progress(url):
        nonlocal crawled
        crawled += 1
        progress = crawled / total_urls
        progress_bar.progress(progress)
        status_text.text(f"Crawled {crawled}/{total_urls}: {url}")
        
    await crawl_parallel(urls, website_name, max_concurrent=5, progress_callback=update_progress)
    
    status_text.text(f"Completed! Crawled {crawled}/{total_urls} URLs")
    return True

async def main():
    st.title("Website Knowledge RAG System")
    
    # Initialize state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_website" not in st.session_state:
        st.session_state.current_website = "Select a website or add a new one"
    if "websites" not in st.session_state:
        try:
            result = supabase.rpc('get_distinct_sources').execute()            
            st.session_state.websites = result.data
        except Exception as e:
            st.error(f"Error loading websites: {str(e)}")
            st.session_state.websites = []
    
    # Sidebar for website selection and crawling
    with st.sidebar:
        st.header("Website Management")
        
        # Website selection
        available_websites = ["Select a website"] + st.session_state.websites
        selected_website = st.selectbox(
            "Choose a website to query:",
            available_websites,
            index=0
        )
        
        if selected_website != "Select a website" and selected_website != st.session_state.current_website:
            st.session_state.current_website = selected_website
            st.session_state.messages = []  # Reset chat when changing websites
            st.rerun()
        
        # Add new website section
        st.subheader("Add New Website")
        new_website_name = st.text_input("Website Name (e.g., python_docs):")
        sitemap_url = st.text_input("Sitemap URL (e.g., https://docs.python.org/sitemap.xml):")
        
        if st.button("Crawl Website") and new_website_name and sitemap_url:
            st.write("Starting crawler...")
            success = await crawl_website(sitemap_url, new_website_name)
            if success:
                if new_website_name not in st.session_state.websites:
                    st.session_state.websites.append(new_website_name)
                st.session_state.current_website = new_website_name
                st.session_state.messages = []
                st.rerun()
    
    # Main chat area
    if st.session_state.current_website == "Select a website or add a new one":
        st.info("Please select a website from the sidebar or add a new one to start chatting.")
    else:
        st.write(f"Currently querying: **{st.session_state.current_website}**")
        st.write("Ask any questions about this website's content!")
        
        # Display all messages from the conversation so far
        for msg in st.session_state.messages:
            if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
                for part in msg.parts:
                    display_message_part(part)
        
        # Chat input for the user
        user_input = st.chat_input(f"Ask about {st.session_state.current_website}...")
        
        if user_input:
            # We append a new request to the conversation explicitly
            st.session_state.messages.append(
                ModelRequest(parts=[UserPromptPart(content=user_input)])
            )
            
            # Display user prompt in the UI
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Display the assistant's partial response while streaming
            with st.chat_message("assistant"):
                # Actually run the agent now, streaming the text
                await run_agent_with_streaming(user_input, st.session_state.current_website)


if __name__ == "__main__":
    asyncio.run(main())