import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from dotenv import load_dotenv
import together
import logging
import re
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# Fix NumPy compatibility
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'int'):
    np.int = np.int64
if not hasattr(np, 'float_'):
    np.float_ = np.float64

# Get API keys
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# Document processing imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredWordDocumentLoader, TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

logger = logging.getLogger(__name__)

class HFInferenceEmbeddings:
    """Custom embedding class using HuggingFace Inference API"""
    def __init__(self, client, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.client = client
        self.model_name = model_name
    
    def embed_documents(self, texts):
        """Embed a list of documents"""
        embeddings = []
        for text in texts:
            try:
                result = self.client.feature_extraction(
                    text,
                    model=self.model_name
                )
                # Ensure result is a list/array
                if isinstance(result, list):
                    embeddings.append(result)
                else:
                    embeddings.append(result.tolist())
            except Exception as e:
                logger.warning(f"Failed to embed text: {str(e)[:100]}...")
                # Use zero vector as fallback (384 dimensions for MiniLM)
                embeddings.append([0.0] * 384)
        return embeddings
    
    def embed_query(self, text):
        """Embed a single query"""
        try:
            result = self.client.feature_extraction(
                text,
                model=self.model_name
            )
            if isinstance(result, list):
                return result
            else:
                return result.tolist()
        except Exception as e:
            logger.warning(f"Failed to embed query: {e}")
            return [0.0] * 384

class RagEngine:
    def __init__(self, doc_folder: str = None, persist_directory: str = "./chroma_langchain_db"):
        self.doc_folder = doc_folder or os.path.join(os.path.dirname(__file__), "knowledge_base")
        self.persist_directory = persist_directory
        self.vectordb = None
        self.together_client = None
        self.documents_cache = []  # Fallback document storage
        self.embedding_model = None
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the RAG system"""
        try:
            # Initialize Together client
            if TOGETHER_API_KEY:
                self.together_client = together.Together(api_key=TOGETHER_API_KEY)
                logger.info("Together client initialized successfully")
            else:
                logger.warning("TOGETHER_API_KEY not found, some features will be limited")
            
            # Initialize vector store with documents
            self.initialize_vector_store()
            
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            raise
    
    def initialize_vector_store(self):
        """Initialize the vector store with documents"""
        try:
            # Load documents
            docs = self.load_documents()
            
            if not docs:
                logger.warning("No documents loaded, creating minimal knowledge base")
                docs = self.create_minimal_knowledge_base()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            split_docs = text_splitter.split_documents(docs)
            logger.info(f"Created {len(split_docs)} chunks from documents")
            
            # Try different embedding approaches
            embedding = None
            
            # First try: HuggingFace Inference API (memory-efficient)
            try:
                # Initialize HF Inference client
                if HF_TOKEN:
                    hf_client = InferenceClient(api_key=HF_TOKEN)
                    logger.info("Using authenticated HuggingFace client")
                else:
                    logger.warning("HF_TOKEN not found, trying without authentication")
                    hf_client = InferenceClient()
                
                # Test the API with a simple call
                try:
                    test_result = hf_client.feature_extraction(
                        "test sentence",
                        model="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    logger.info("HuggingFace API test successful")
                except Exception as test_e:
                    logger.warning(f"HF API test failed: {test_e}")
                    raise Exception(f"HF API test failed: {test_e}")
                
                # Use the HF API embedding
                embedding = HFInferenceEmbeddings(hf_client)
                self.embedding_model = embedding
                logger.info("Using HuggingFace Inference API for embeddings (memory-efficient)")
                
            except Exception as e:
                logger.warning(f"HuggingFace Inference API failed: {e}")
                
            # Second try: Use simple keyword matching as fallback
            if embedding is None:
                logger.warning("Using simple keyword matching (no embeddings)")
                # Store documents for keyword-based search
                self.documents_cache = split_docs
                return
            
            # Create/Load Chroma VectorDB
            try:
                self.vectordb = Chroma.from_documents(
                    documents=split_docs,
                    embedding=embedding,
                    persist_directory=self.persist_directory
                )
                self.vectordb.persist()
                logger.info(f"VectorDB ready with {len(split_docs)} chunks")
            except Exception as chroma_e:
                logger.error(f"Chroma VectorDB creation failed: {chroma_e}")
                # Fallback to keyword search
                self.documents_cache = split_docs
                logger.info(f"Using fallback keyword search with {len(split_docs)} chunks")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            # Store documents for fallback keyword search
            try:
                docs = self.load_documents()
                if not docs:
                    docs = self.create_minimal_knowledge_base()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                self.documents_cache = text_splitter.split_documents(docs)
                logger.info(f"Using fallback keyword search with {len(self.documents_cache)} chunks")
            except Exception as fallback_e:
                logger.error(f"Even fallback initialization failed: {fallback_e}")
                # Create minimal cache
                self.documents_cache = self.create_minimal_knowledge_base()
    
    def load_documents(self):
        """Load documents from the specified folder"""
        docs = []
        
        try:
            doc_folder_path = Path(self.doc_folder)
            
            if not doc_folder_path.exists():
                logger.warning(f"Document folder {self.doc_folder} does not exist")
                return docs
            
            # Load different file types
            for file in doc_folder_path.glob("*.*"):
                if file.suffix.lower() in ['.docx', '.pdf', '.txt']:
                    try:
                        logger.info(f"Loading: {file.name}")
                        
                        if file.suffix.lower() == '.docx':
                            try:
                                loader = Docx2txtLoader(str(file))
                            except:
                                loader = UnstructuredWordDocumentLoader(str(file))
                        elif file.suffix.lower() == '.pdf':
                            loader = PyPDFLoader(str(file))
                        elif file.suffix.lower() == '.txt':
                            loader = TextLoader(str(file))
                        else:
                            continue
                            
                        file_docs = loader.load()
                        for doc in file_docs:
                            doc.metadata['source_file'] = file.name
                        docs.extend(file_docs)
                        logger.info(f"Loaded {len(file_docs)} pages from {file.name}")
                    except Exception as e:
                        logger.error(f"Error loading {file.name}: {str(e)}")
            
            return docs
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []
    
    def create_minimal_knowledge_base(self):
        """Create a minimal knowledge base for fallback"""
        minimal_knowledge = [
            Document(
                page_content="PingUs is a freelancing team of five developers from India specializing in AI solutions and web applications.",
                metadata={"source_file": "default_knowledge.docx"}
            ),
            Document(
                page_content="Our AI Integration services include custom chatbots, RAG systems, and LLM solutions for businesses.",
                metadata={"source_file": "default_knowledge.docx"}
            ),
            Document(
                page_content="We provide web development services including responsive websites and scalable web applications with modern technologies.",
                metadata={"source_file": "default_knowledge.docx"}
            ),
            Document(
                page_content="API development and integration services help connect your business systems and automate workflows.",
                metadata={"source_file": "default_knowledge.docx"}
            ),
            Document(
                page_content="Contact PingUs through email, LinkedIn, or our website contact form. We respond quickly and work in IST timezone.",
                metadata={"source_file": "default_knowledge.docx"}
            ),
            Document(
                page_content="Pricing for our services is customized based on project scope, complexity, and specific requirements.",
                metadata={"source_file": "default_knowledge.docx"}
            )
        ]
        return minimal_knowledge
    
    def search_documents(self, query: str, k: int = 3):
        """Search for relevant documents in the vector store or use fallback"""
        # Try vector store first
        if self.vectordb:
            try:
                results = self.vectordb.similarity_search(query, k=k)
                return results
            except Exception as e:
                logger.error(f"Error searching vector store: {e}")
        
        # Fallback to keyword-based search
        if self.documents_cache:
            return self.keyword_search(query, k)
        
        return []
    
    def keyword_search(self, query: str, k: int = 3):
        """Simple keyword-based document search fallback"""
        query_words = [word.lower() for word in re.findall(r'\w+', query)]
        scored_docs = []
        
        for doc in self.documents_cache:
            content = doc.page_content.lower()
            score = sum(1 for word in query_words if word in content)
            if score > 0:
                scored_docs.append((doc, score))
        
        # Sort by score and return top k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:k]]
    
    def query(self, question: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Process a question using the RAG system
        
        Args:
            question: The user's question
            chat_history: Previous conversation history for context
            
        Returns:
            Dictionary with answer and source documents
        """
        try:
            if not self.together_client:
                return {
                    "response": "AI service is not configured. Please check your API key.",
                    "sources": [],
                    "success": False
                }
            
            # Search for relevant documents
            relevant_docs = self.search_documents(question, k=4)
            
            # Build context from relevant documents
            context = self.build_context(relevant_docs)
            
            # Prepare messages for the API
            messages = self.prepare_messages(question, context, chat_history)
            
            # Call Together API
            response = self.together_client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                messages=messages,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9
            )
            
            # Format source documents
            sources = []
            for doc in relevant_docs:
                source_file = doc.metadata.get("source_file", "Unknown Document")
                sources.append({
                    "title": source_file,
                    "url": f"/docs/{source_file}",
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
            
            return {
                "response": response.choices[0].message.content,
                "sources": sources,
                "success": True
            }
            
        except together.error.AuthenticationError:
            logger.error("Authentication failed - check your Together API key")
            return {
                "response": "Authentication error. Please check your API configuration.",
                "sources": [],
                "success": False
            }
        except together.error.RateLimitError:
            logger.error("Rate limit exceeded")
            return {
                "response": "I'm receiving too many requests. Please try again in a moment.",
                "sources": [],
                "success": False
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again later.",
                "sources": [],
                "success": False
            }
    
    def build_context(self, documents: List[Document]) -> str:
        """Build context string from relevant documents"""
        if not documents:
            return "No relevant documents found."
        
        context_parts = ["Relevant information from knowledge base:"]
        for i, doc in enumerate(documents, 1):
            source_file = doc.metadata.get('source_file', 'Unknown')
            context_parts.append(f"\n--- Document {i} ({source_file}) ---")
            context_parts.append(doc.page_content)
        
        return "\n".join(context_parts)
    
    def prepare_messages(self, question: str, context: str, chat_history: List[Dict] = None) -> List[Dict]:
        """Prepare messages for the Together API"""
        messages = []
        
        # System message with instructions
        system_message = """You are an AI-powered assistant representing PingUs ✨ – a freelancing team of five developers from India 🇮🇳.
You provide answers strictly based on PingUs' portfolio documents 📂.

🎯 Goals:

Give crisp, clear, and confident answers ✅

Highlight key points with bold text ✨

Use structured lists 📌 for readability

Never say "I don't know" 🙅 — instead, guide users toward services, expertise, or contact options

Maintain a friendly + professional tone 🤝

End with a call to action 📞 when suitable

💡 Example Styles:

Q: Who are you?
👉 "We are PingUs – a team of five freelance developers 👨‍💻👩‍💻 from India.
Our expertise lies in:

🤖 AI Solutions (Chatbots, RAG systems, AI Agents)

🌐 Web Applications (Scalable, real-time, secure)
Together, we help businesses automate workflows, boost engagement, and scale online 🚀."

Q: Do you build mobile apps?
👉 "At PingUs, we specialize in Web Applications 🌐 and AI Chatbots 🤖.
Mobile app development 📱 is not part of our services at this time."

Q: How can I contact you?
👉 "You can reach PingUs through:

📧 Email

🌍 Contact Page

💼 LinkedIn

📞 Phone
We respond quickly ⏱ and are available globally 🌎 in IST (UTC +5:30)"""
        
        messages.append({"role": "system", "content": system_message})
        
        # Add conversation history if available
        if chat_history:
            for msg in chat_history[-6:]:  # Last 6 messages for context
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add context and current question
        prompt = f"""Context information:
{context}

Based on the above context, please answer the following question:

Question: {question}

Answer:"""
        
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    def add_document(self, file_path: str):
        """Add a new document to the knowledge base"""
        try:
            if not Path(file_path).exists():
                return False, "File does not exist"
            
            # Determine loader based on file extension
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.docx':
                try:
                    loader = Docx2txtLoader(file_path)
                except:
                    loader = UnstructuredWordDocumentLoader(file_path)
            elif file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_ext == '.txt':
                loader = TextLoader(file_path)
            else:
                return False, f"Unsupported file type: {file_ext}"
                
            new_docs = loader.load()
            
            # Add source file metadata
            for doc in new_docs:
                doc.metadata['source_file'] = Path(file_path).name
            
            # Split the document
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            split_new_docs = text_splitter.split_documents(new_docs)
            
            # Add to vector store if available, otherwise to cache
            if self.vectordb:
                self.vectordb.add_documents(split_new_docs)
                self.vectordb.persist()
            else:
                self.documents_cache.extend(split_new_docs)
            
            logger.info(f"Added {len(split_new_docs)} chunks from {Path(file_path).name}")
            return True, f"Added {len(split_new_docs)} chunks from {Path(file_path).name}"
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False, str(e)

# Test function
def test_together_api():
    """Test the Together API connection"""
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("ERROR: TOGETHER_API_KEY not found in environment variables")
        return False
    
    try:
        client = together.Together(api_key=api_key)
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            max_tokens=50
        )
        print("SUCCESS: Together API is working!")
        print(f"Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"ERROR: Together API test failed: {e}")
        return False

def test_hf_api():
    """Test the HuggingFace API connection"""
    hf_token = os.getenv("HF_TOKEN")
    try:
        if hf_token:
            client = InferenceClient(api_key=hf_token)
        else:
            client = InferenceClient()
            
        result = client.feature_extraction(
            "test sentence",
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("SUCCESS: HuggingFace API is working!")
        print(f"Embedding dimension: {len(result)}")
        return True
    except Exception as e:
        print(f"ERROR: HuggingFace API test failed: {e}")
        return False

if __name__ == "__main__":
    # Test the APIs
    print("Testing Together API...")
    test_together_api()
    
    print("\nTesting HuggingFace API...")
    test_hf_api()
    
    # Test the RAG engine
    print("\nTesting RAG Engine...")
    engine = RagEngine()
    result = engine.query("What services does PingUs offer?")
    print("RAG Test Result:")
    print(f"Response: {result['response']}")
    print(f"Success: {result['success']}")
    if result['sources']:
        print("Sources:")
        for source in result['sources']:
            print(f"  - {source['title']}")
