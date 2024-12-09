import streamlit as st
from src.retrieval import user_input

st.title("Document Search with RAG")
st.write("Upload your documents, and ask questions to retrieve relevant information.")

# Input field for user questions
user_question = st.text_input("Enter your question:")

if st.button("Submit"):
    if user_question:
        st.write("Searching for relevant information...")
        # Query the retrieval function
        response = user_input(user_question)
        st.write("Answer:")
        answer = ""
        if isinstance(response,dict):
            answer = response['output_text']
        else:
            answer = response
        st.write(answer)
    else:
        st.write("Please enter a question.")

st.sidebar.title("About")
st.sidebar.info("This application allows users to upload various document types and perform a question-answering search over them using Google's Gemini AI.")
