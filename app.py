import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from main import graph_agent_supervisor  # zakładamy, że kod LangGraph masz w tym module
import uuid
import asyncio
from ChromaDbManager import ChromaDbManager
import traceback

def streamlit_to_langchain(messages):
    result = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, str):
            content = str(content)  # awaryjna konwersja, jeśli coś jest nie tak
        if msg["role"] == "user":
            result.append(HumanMessage(content=content))
        elif msg["role"] == "assistant":
            result.append(AIMessage(content=content))
    return result

def extract_content_safely(response_content):
    """Safely extract content from various response formats."""
    try:
        # If it's already a string, use it
        if isinstance(response_content, str):
            content = response_content
        # If it's an AIMessage object, extract the content
        elif hasattr(response_content, 'content'):
            content = response_content.content
        # If it's a dict with content key
        elif isinstance(response_content, dict) and 'content' in response_content:
            content = response_content['content']
        # Try to convert to string as fallback
        else:
            content = str(response_content)
        
        # Clean up the content - remove "response: " prefix and quotes if present
        if content and isinstance(content, str):
            if content.startswith("response: '") and content.endswith("'"):
                content = content[11:-1]
            elif content.startswith("response: "):
                content = content[10:]
        
        return content
    except Exception as e:
        st.error(f"Error processing response content: {e}")
        return "Sorry, there was an error processing the response."

# Workaround for "RuntimeError: no running event loop"
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

st.title("Chatbot")

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'thread_id' not in st.session_state:
    # Generuj unikalny thread_id dla tej sesji
    st.session_state.thread_id = str(uuid.uuid4())

st.sidebar.title("Model Parameters")
temp = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
max_token = st.sidebar.slider("Max tokens", min_value=1, max_value=4096, value=256)

# Clear chat button
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

file_uploader = st.sidebar.file_uploader("Upload a file", type=["pdf", "txt", "docx"])

if file_uploader:
    file_details = {"filename": file_uploader.name,
                    "filetype": file_uploader.type,
                    "filesize": file_uploader.size}
    st.sidebar.write(file_details)
    with open(file_uploader.name, "wb") as f:
        f.write(file_uploader.getbuffer())
    st.sidebar.success("File uploaded successfully!")
    chromaDbManager = ChromaDbManager()
    chromaDbManager.save_to_chromadb_async(file_uploader.name)

# Update the UI with previous messages
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# UI chat
if prompt := st.chat_input("Ask a question:"):
    # Add user message first
    user_message = {"role": "user", "content": prompt}
    st.session_state['messages'].append(user_message)
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Response
    with st.chat_message("assistant"):
        # Show thinking message
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("Thinking...")
        
        try:
            # Convert messages to LangGraph format
            messages = streamlit_to_langchain(st.session_state['messages'])
            print("messages from app.py: ", messages)
            
            # Get response from LangGraph
            response = graph_agent_supervisor.invoke(
                {
                    "question": HumanMessage(content=prompt),
                    "messages": messages,
                },
                config={
                    "configurable": {"thread_id": st.session_state.thread_id}
                })
            
            print("response from app.py: ", response)
            
            # Process the answer safely
            raw_content = response.get("answer", "")
            content = extract_content_safely(raw_content)
            
            # Ensure we have some content
            if not content or content.strip() == "":
                content = "I apologize, but I wasn't able to generate a proper response. Please try rephrasing your question."
            
        except Exception as e:
            print(f"Error in LangGraph processing: {e}")
            print(f"Full traceback: {traceback.format_exc()}")
            content = f"Sorry, there was an error processing your request: {str(e)}"
        
        # Clear thinking message and show actual response
        thinking_placeholder.empty()
        st.markdown(content)
        
        # Add assistant's response to session state
        assistant_message = {"role": "assistant", "content": content}
        st.session_state['messages'].append(assistant_message)
        
        # Debug: Print session state messages
        #print(f"Session state messages count: {len(st.session_state.messages)}")
        for i, msg in enumerate(st.session_state.messages):
            print(f"Message {i}: {msg['role']} - {msg['content'][:50]}...")

# Debug panel (optional - can be removed in production)
with st.expander("Debug - Session Messages"):
    st.write(f"Total messages in session: {len(st.session_state.messages)}")
    for i, msg in enumerate(st.session_state.messages):
        st.write(f"{i}: {msg['role']} - {msg['content']}")
    
    # Show last response details
    if st.session_state.messages:
        st.write("Last response debugging:")
        try:
            # This will help debug what the actual response structure looks like
            messages = streamlit_to_langchain(st.session_state['messages'])
            st.write(f"Converted messages: {len(messages)}")
        except Exception as e:
            st.error(f"Error in message conversion: {e}")