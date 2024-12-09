from langchain.retrievers.document_compressors import LLMChainFilter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import FAISS
from langchain.retrievers.document_compressors import LLMListwiseRerank
from langchain.retrievers.document_compressors import EmbeddingsFilter



def contextual_compression(query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True).as_retriever()

    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    _filter = LLMChainFilter.from_llm(llm)
    # _filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=_filter, base_retriever=new_db
    )

    compressed_docs = compression_retriever.invoke(query)
    return compressed_docs