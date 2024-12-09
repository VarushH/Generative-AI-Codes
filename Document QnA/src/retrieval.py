import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_core.documents import Document
from ContextualCompression import contextual_compression
from langchain_google_genai import GoogleGenerativeAI

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
def get_conversational_chain():
    prompt_template = """
    Question: \n{question}\n
    Answer the above question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-exp-0827", temperature=0.3)
    # model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-002", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, index_path=r"faiss_index"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if not os.path.exists(index_path):
        return "FAISS index does not exist. Please run the data ingestion script first."
    else:
        new_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        # docs = new_db.similarity_search(user_question)
        # docs = [Document(metadata={'source': 'C:\\Users\\akshi\\Documents\\Varush\\RAG GEMINI\\Data\\helpx_adobe_com_tips-and-tricks.txt'}, page_content='Design and create prototypes faster in Adobe XD using these tips and tricks. Adobe Discover Mobile Apps Adobe Express Photoshop Express Lightroom Acrobat PDF Reader Adobe Scan Creativity & Design Creativity Creativity & Design Quick Actions Remove Background Resize Image Convert image to SVG Convert video to GIF Create a QR code See all quick actions Create Now Resume Posters Card Instagram Post YouTube Video Create now Featured products Creative Cloud Over 20 Creative Cloud Apps Adobe Express Social graphics and more Photoshop Image editing and design Premiere Pro Video editing and production Illustrator Vector graphics and illustration See plans and pricing Artificial Intelligence AI Overview About AI at Adobe Adobe Firefly AI-powered content creation Find the perfect app in about 60')]
        docs = contextual_compression(user_question)
        print('What is the type of docs ',docs)
        chain = get_conversational_chain()
        response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response

# if __name__ == "__main__":

#     print(user_input("what is the Value for declared tarrif 950 ", index_path=r"faiss_index"))