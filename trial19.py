import os
import io
import streamlit as st
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import base64

load_dotenv()

# Set the API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

def get_word_text(word_bytes):
    """Extracts text from uploaded Word documents."""
    text = ""
    word_stream = io.BytesIO(word_bytes)
    doc = Document(word_stream)
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def get_text_chunks(text):
    """Splits text into chunks for vectorization."""
    if not text:
        return []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Creates or loads the FAISS vector store."""
    if not text_chunks:
        return None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def get_conversational_chain():
    """Creates the question-answering chain."""
    try:
        prompt_template = """
        You are an AI assistant specialized in extracting detailed information from documents. Your goal is to provide accurate and comprehensive answers to the questions based on the context provided from these documents. Follow these guidelines while answering:

        1. Begin with a concise summary of the context relevant to the question.
        2. Provide a detailed answer, including all pertinent details and explanations.
        3. If the answer cannot be derived from the provided context, try to answer from your knowledge but if it beyond your knowledge clearly state, "The answer is not in the context."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3, google_api_key=GOOGLE_API_KEY)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", 'question'])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error loading conversational chain: {str(e)}")
        return None

def user_input(user_question):
    """Processes user question and retrieves answer from Word documents."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        new_db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        if chain:
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            st.write("Reply: ", response["output_text"])
        else:
            st.error("Error processing user input.")
    except Exception as e:
        st.error(f"Error processing user input: {str(e)}")

def handle_upload(text):
    # Process the uploaded text
    text_chunks = get_text_chunks(text)
    get_vector_store(text_chunks)

@st.cache(allow_output_mutation=True)
def get_static_store():
    return {}

# Handle file uploads from Google Docs
# Handle file uploads from Google Docs
@st.cache_data(hash_funcs={str: lambda _: None})
def upload():
    uploaded_file = st.file_uploader("Upload File", type=["txt"])
    if uploaded_file:
        handle_upload(uploaded_file.read().decode("utf-8"))

def functmain():
    try:
        st.set_page_config(page_title="MyThoughts.AI", layout='wide', initial_sidebar_state='auto')
        st.title("Welcome to [MyThoughts.AI](https://github.com/vibhansh/Langchain_Gemini_Test_App)")
        st.write("Interact with your data")
        
        # Empty placeholder for user response
        user_response = st.empty()

        # Sidebar for Word document upload and processing
        with st.sidebar:
            st.title("Menu:")
            word_docs = st.file_uploader("Upload your Word docs and click Submit", accept_multiple_files=True, type=['docx'])

            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    all_text_chunks = []
                    if word_docs:
                        for word_file in word_docs:
                            raw_text = get_word_text(word_file.getvalue())
                            text_chunks = get_text_chunks(raw_text)
                            all_text_chunks.extend(text_chunks)
                        get_vector_store(all_text_chunks)
                    st.success("Done")

        # Handle file uploads from Google Docs
        upload()

        # Text input for user question
        user_question = st.text_input("Ask a Question about the uploaded data")

        # Update user response based on user input
        if user_question:
            user_input(user_question)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    functmain()