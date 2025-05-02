import os
import re
from typing import List, Dict, Optional

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DestinationQA:
    def __init__(self, pdf_path: str, persist_dir: str = "./chroma_db"):
        self.pdf_path = pdf_path
        self.persist_dir = persist_dir
        self.vector_db = None
        self.qa_chain = None
        self.embedding_model = "nomic-embed-text"
        self.llm_model = "llama3.2:3b"
        
        # Configure text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self._initialize_prompt_template()
    
    def _initialize_prompt_template(self):
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a knowledgeable transportation assistant. Your task is to help users find 
            transportation routes based on the provided information.

            Context Information:
            - Each route has a ROUTE number, description, and sequence of locations
            - Locations may have alternative names or spellings
            
            Guidelines:
            1. For location queries, return all routes including:
               - Route number
               - Description
               - Complete sequence of locations
            2. Be flexible with location name matching (consider abbreviations, nearby areas)
            3. If multiple routes serve the location, list them all
            4. If no matching routes found, respond: "I couldn't find routes for that location."
            5. NEVER invent information - only use what's in the context
            6. If the user is chatting and not asking for destinations, be yourself and chat back.

            Context:
            {context}

            Question:
            {question}

            Please provide a helpful response:
            """
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Normalize common transportation terms
        text = re.sub(r'\b(Route|Rt|Rte)\b', 'ROUTE', text, flags=re.IGNORECASE)
        return text
    
    def load_and_process_pdf(self) -> List[Document]:
        """Load PDF and process into destination documents."""
        try:
            loader = UnstructuredPDFLoader(file_path=self.pdf_path)
            docs = loader.load()
            
            # Combine and clean text
            full_text = self._clean_text("\n".join(doc.page_content for doc in docs))
            
            # Split by destinations
            dest_chunks = {}
            matches = re.split(r"(?=\[DESTINATION:.*?\])", full_text)
            
            for chunk in matches:
                if chunk.strip():
                    dest_match = re.search(r"\[DESTINATION:\s*(.*?)\]", chunk)
                    if dest_match:
                        destination = dest_match.group(1).strip().lower()
                        cleaned_chunk = self._clean_text(chunk)
                        dest_chunks.setdefault(destination, "")
                        dest_chunks[destination] += cleaned_chunk + "\n"
            
            # Further split large chunks
            final_docs = []
            for dest, text in dest_chunks.items():
                splits = self.text_splitter.split_text(text)
                for i, split in enumerate(splits):
                    metadata = {
                        "destination": dest,
                        "chunk_num": i+1,
                        "total_chunks": len(splits)
                    }
                    final_docs.append(Document(page_content=split, metadata=metadata))
            
            return final_docs
        
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return []
    
    def initialize_qa_system(self):
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF file not found at {self.pdf_path}")
        os.makedirs(self.persist_dir, exist_ok=True)
        print("Loading and processing PDF...")
        chunks = self.load_and_process_pdf()
        print(f"Processed {len(chunks)} document chunks from PDF.")
        print("Building vector database...")
        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(model=self.embedding_model),
            collection_name="destination_routes",
            persist_directory=self.persist_dir
        )
        self.vector_db.persist()
        
        # Configure retriever with MMR for diverse results
        retriever = self.vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,  
                "fetch_k": 20,
                "lambda_mult": 0.5  
            }
        )
        
        # Initialize QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOllama(model=self.llm_model, temperature=0.1),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt_template}
        )
    
    def query_destinations(self, question: str) -> Dict:
        """Query the destination knowledge base."""
        if not self.qa_chain:
            raise RuntimeError("QA system not initialized.")
        
        try:
            result = self.qa_chain.invoke({"query": question})
            return {
                "answer": result["result"],
                "sources": [doc.metadata.get("destination", "unknown") for doc in result["source_documents"]]
            }
        except Exception as e:
            return {
                "answer": f"Sorry, an error occurred while processing your request: {str(e)}",
                "sources": []
            }
    
    def interactive_session(self):
        if not self.qa_chain:
            self.initialize_qa_system()
        
        print("\nDestination Route Information System")
        print("Type your question (e.g., 'How do I get to German Hospital?') or 'exit' to quit\n")
        
        while True:
            try:
                query = input("\u00bb ").strip()
                if query.lower() in ("exit", "quit", "q"):
                    break
                
                if not query:
                    continue
                
                result = self.query_destinations(query)
                print(result["answer"])
                
                # if result["sources"]:
                #     print("\nSOURCE LOCATIONS:", ", ".join(set(result["sources"])))
                # print("\n" + "="*50 + "\n")
            
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}\n")

if __name__ == "__main__":
    # Configuration
    PDF_PATH = r"D:\GRAD\salka_routes.pdf"
    DB_DIR = "./chroma_db"
    
    # Initialize and run the system
    try:
        qa_system = DestinationQA(PDF_PATH, DB_DIR)
        qa_system.interactive_session()
    except Exception as e:
        print(f"Failed to initialize system: {e}")
