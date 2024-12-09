import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader, TextLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.document_loaders import DataFrameLoader
import pandas as pd

# import pytesseract
# from pdf2image import convert_from_path
from PIL import Image

# Specify the path to Tesseract if not in your PATH (for Windows users)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def load_documents_from_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if filename.endswith(".txt"):
            loader = TextLoader(file_path,encoding='utf8')
            documents.extend(loader.load())
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif filename.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(file_path)
            documents.extend(loader.load())
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())

        elif filename.endswith(".xlsx"):
            # loader = UnstructuredExcelLoader(file_path)
            df = pd.read_excel(file_path)
            loader = DataFrameLoader(df)
            documents.extend(loader.lazy_load())
    return documents

# def load_documents_from_directory_v2(file_path_list):
#     documents = []
#     for file_path in file_path_list:
#         print(file_path)
#         filename = os.path.basename(file_path)
#         # for filename in os.listdir(directory_path):
#         #     file_path = os.path.join(directory_path, filename)
#         if filename.endswith(".txt"):
#             loader = TextLoader(file_path,encoding='utf8')
#             documents.extend(loader.load())
#         elif filename.endswith(".pdf"):
#             loader = PyPDFLoader(file_path)
#             documents.extend(loader.load())
#         elif filename.endswith(".pptx"):
#             loader = UnstructuredPowerPointLoader(file_path)
#             documents.extend(loader.load())
#         elif filename.endswith(".docx"):
#             loader = Docx2txtLoader(file_path)
#             documents.extend(loader.load())
#         elif filename.endswith(".xlsx"):
#             # loader = UnstructuredExcelLoader(file_path)
#             df = pd.read_excel(file_path)
#             loader = DataFrameLoader(df)
#             documents.extend(loader.load())
        
#         ts = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
#         sdocs = ts.split_documents(documents)   
#         # print(sdocs)
#         # if sdocs == []:
#         #     documents = []
#         #     # Convert PDF pages to images
#         #     images = convert_from_path(file_path, dpi=300)

#         #     # Loop through images and extract text
#         #     text = ""
#         #     for page_num, image in enumerate(images, start=1):
#         #         _text = pytesseract.image_to_string(image, lang='eng')
#         #         text = f'{text}\n{_text}'
#         #     #dump text in a .txt file and store it read the file with textloader and delete the file
#         #     temp_file_name = f'{filename.replace('.',"_")}.txt'
#         #     with open(temp_file_name, 'w', encoding='utf8') as file:
#         #         file.write(text)
#         #     loader = TextLoader(temp_file_name,encoding='utf8')
#         #     documents.extend(loader.load())
#         #     os.remove(temp_file_name)

#     return documents

def create_faiss_index(directory_path, index_path = 'faiss_index'):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)

    # Load and split documents
    documents = load_documents_from_directory(directory_path)
    split_docs = text_splitter.split_documents(documents)

    # Create FAISS index
    try:
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        print("Loading exsisting FAISS index")
        new_texts = [doc.page_content for doc in split_docs]
        vector_store.add_texts(new_texts)
        vector_store.save_local(index_path)
        print(f"FAISS index saved at {index_path}")
    except Exception as e:
        print(f"No FIASS index found")
        print("Creating new FAISS index")
        vector_store = FAISS.from_documents(split_docs, embedding=embeddings)
        vector_store.save_local(index_path)
        print(f"FAISS index created and saved at {index_path}")

if __name__ == "__main__":
    create_faiss_index(directory_path=r"C:\Users\akshi\Documents\Varush\RAG GEMINI\Data")