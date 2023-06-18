import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain

from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils import get_max_token_for_model, get_file_type, loader_from_file_from_extension

#create a temporary directory to store the uploaded files
import tempfile
import os
temp_dir = tempfile.TemporaryDirectory()

st.title("📄📒📰🦜 Multi-Docs Summarizer")
st.markdown("## Welcome to the Multi-Docs Summarizer")
st.markdown("#### Please enter your OPENAI_API_KEY to access the service.")
OPENAI_API_KEY = st.text_input(label="OPENAI_API_KEY", type="password")
# Add OPENAI_API_KEY to env
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
model = st.radio("Model", options=["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"], index=0)




llm = ChatOpenAI(temperature=0,
                 model=model,
                 openai_api_key=OPENAI_API_KEY,
                 max_tokens=get_max_token_for_model(model=None)
                 )

st.markdown("## Upload your documents")


## Ask for data and read it
uploaded_files = st.file_uploader("Choose a set of documents", accept_multiple_files=True)

files_data = dict()

for uploaded_file in uploaded_files:
    # write the file to the temp directory
    file_type = get_file_type(uploaded_file)
    with open(os.path.join(temp_dir.name, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = loader_from_file_from_extension(os.path.join(temp_dir.name, uploaded_file.name), file_type)    
    files_data[uploaded_file.name] = loader.load()
    bytes_data = uploaded_file.read()

## Chunk the documents - so that we can summarize them in parallel
docs_chunks = dict()
for file_name, file_data in files_data.items():
    st.write(f"Chunking {file_name}...")
    splitter = RecursiveCharacterTextSplitter(chunk_size = get_max_token_for_model(model)*4,
                                              chunk_overlap  = 0.1*get_max_token_for_model(model)*4,
                                              length_function = len
                                              )
    docs_chunks[file_name] =  splitter.create_documents([file_data[0].page_content, ])


## merge the chunks
docs = []
for docs_chunk in docs_chunks.values():
    docs.extend(docs_chunk)

st.write(f"Number of document chunks: {len(docs)}")

## Summarize the documents
st.markdown(f"## Summary")
if st.button('Summarize'):
    numbers = st.empty()
    with numbers.container():
        st.write('Summarizing...\nThis may take a couple of minutes...')
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        resp = chain.run(docs)
        print(resp)
        st.write(resp)
else:
    st.write('No summarization performed yet')








