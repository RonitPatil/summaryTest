import os
import streamlit as st
import chardet
import pdfplumber
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

def read_file(uploaded_file):
    file_type = uploaded_file.type
    if file_type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text
    else:
        raw_data = uploaded_file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding'] if result['encoding'] is not None else 'utf-8'
        return raw_data.decode(encoding, errors='replace')

def generate_summary(text):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    message = HumanMessage(content=f"Summarize the following content:\n\n{text}\n\nSummary:")
    response = llm([message])
    response_content = response.content.replace('\n', '<br>')
    return response_content

def main():
    st.title("File Summary Generator with Gemini LLM")

    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"])
    if uploaded_file is not None:
        try:
            content = read_file(uploaded_file)
            summary = generate_summary(content)
            st.chat_message("assistant").markdown(AIMessage(summary), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error reading file: {e}")

if __name__ == "__main__":
    main()
