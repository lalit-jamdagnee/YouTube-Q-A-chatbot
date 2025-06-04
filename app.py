# Required Libraries

from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi
from youtube_transcript_api.proxies import GenericProxyConfig
from youtube_transcript_api._errors import NoTranscriptFound, VideoUnavailable
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
import gradio as gr 
import re 
import os

load_dotenv()

class Indexing:

    def __init__(self, url):
        self.url = url 
        self._transcript = None

    def extract_video_id(self):
        pattern = (
            r"(?:https?://)?(?:www\.)?"
            r"(?:youtube\.com/(?:watch\?v=|embed/|shorts/)|youtu\.be/)"
            r"([0-9A-Za-z_-]{11})"
        )
        match = re.search(pattern, self.url)
        if not match:
            print(f"[Error] Could not extract video ID from URL: {self.url}")
            return None

        video_id = match.group(1)
        print(f"[Info] Extracted video ID: {video_id}")
        return video_id

    def transcript(self):

        proxies = GenericProxyConfig(http_url= 'http://103.87.169.243:3128',
                                     https_url= 'http://103.87.169.243:3128')
        if self._transcript is not None:
            return self._transcript
        try:
            id = self.extract_video_id()
            yt_api = YouTubeTranscriptApi(proxy_config=proxies)
            transcript_list = yt_api.get_transcript(video_id=id, languages=['en'])
            self._transcript = ' '.join(doc['text'] for doc in transcript_list)
            return self._transcript
        
        except TranscriptsDisabled:
            print("Transcripts are disabled")
            return None
        except NoTranscriptFound:
            print("No Transcripts found")
            return None
        except VideoUnavailable:
            print("Video not available")
            return None
        except Exception as e:
            print(f"Unexpected error while fetching the transcript: {e}")
            return None
    
    def splitter(self):
        transcript = self.transcript()
        if not transcript:
            print("Transcript is None")
            return []
        
        print(f"Length of the transcript {len(transcript)}")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        chunks = splitter.create_documents([transcript])
        return chunks
    
    def vectorStore(self):
        cache_path = f"vectorstore_cache_{self.extract_video_id()}" # unique cache folder per video
        
        # Initialize the Huggingface embedding model
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device':'cpu'}
        )

        # Try loading from the cache first 
        if os.path.exists(cache_path):
            try:
                print("[Info] Loading vector store from cache... ")
                vectorstore = FAISS.load_local(
                    folder_path=cache_path,
                    embeddings=embedding_model,
                    allow_dangerous_deserialization=True
                )
                return vectorstore
            except Exception as e:
                print(f"[Warning] failed to load cached vectorstore: {e}")


        docs = self.splitter()
        if not docs:
            print("No Document to embed")
            return []
        

        # create a Chroma vector store from documents

        vectorstore = FAISS.from_documents(
            embedding=embedding_model,
            documents=docs,
        )
        
        # save to the cache folder 
        try: 
            vectorstore.save_local(cache_path)
            print(f"[Info] Vectorstore save to cache at: {cache_path}")
        except Exception as e:
            print(f"[Warning] Failed to save the vector store cache: {e}")
        
        return vectorstore

def model():
    # llm = HuggingFaceEndpoint(
    #     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    #     temperature=0.4
    # )
    # model = ChatHuggingFace(llm = llm)

    # get the api key
    secret_key = os.getenv('NVIDIA_API_KEY')
    llm = ChatNVIDIA(
        model="meta/llama-4-maverick-17b-128e-instruct",
        api_key=secret_key, 
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        )
    return llm

def retrieval(url):
    obj = Indexing(url)
    try:
        vectorstore = obj.vectorStore()
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k':6, 'fetch_k':30})
        return retriever
    except Exception as e:
        print(f"Unexpected error occured while retrieval: {e}")
        return RunnablePassthrough()

def contextualCompressor(url, llm_model):
    try: 
        compressor = LLMChainExtractor.from_llm(llm_model)
        retriever = retrieval(url)
        
        contextual_retriever = ContextualCompressionRetriever(
            base_retriever = retriever,
            base_compressor = compressor
        )
        return contextual_retriever
    except Exception as e:
        print(f"Unexpected error occured while contextual compression:{e}")
        return RunnablePassthrough()

def format_docs(retrieved_docs):
    context = ' '.join(doc.page_content for doc in retrieved_docs)
    return context 


def improveQuery(query, llm_model):
    prompt = PromptTemplate.from_template(
        template="""
            You are an expert assistant that rewrites and improves user queries to be clearer, more specific, and optimized for accurate information retrieval.

            - Use only the information contained in the original query.

            - Do not request or rely on any additional input or clarification.

            - Maintain the original intent while eliminating ambiguity.

            - Add context only if it is logically implied or necessary for clarity.

            Original user query: "{query}"
            Improved query (more precise and retrieval-optimized):
            """)
    parser = StrOutputParser() # Output parser
    chain = prompt | llm_model | parser # LLM chain
    result = chain.invoke({'query':query})

    return result

def augmentation(url, query, llm_model):
    prompt = PromptTemplate(
                template = """
                    You are a helpful assistant.
                    Answer ONLY from the provided transcript context.
                    if the context is insufficient, just say I don't know as the context is insufficient

                    {context}
                    Question: {question}
                """,
                input_variables=['context', 'question']
                )
    
    # Retrieve the vector from vector DB

    retriever = contextualCompressor(url, llm_model) # Initialize retriever object
    retrieved_docs = retriever.invoke(query) #get retrieved docs
    if not len(retrieved_docs): 
        retriever = retrieval(url)
        retrieved_docs = retriever.invoke(query)

    print(f"[debug]Retrieved {len(retrieved_docs)} Documents") # length of retrieved document

    # Context 

    context = format_docs(retrieved_docs) # combine all docs content into single variable
    print(f"[debug]Context Length: {len(context)}") # length of context

    # Get Query

    improved_query = improveQuery(query, llm_model) # enhance the query
    print(f"[debug]Original query: {query}")
    print(f"[debug]Improved query: {improved_query}")

    # Final Prompt 

    final_prompt = prompt.invoke({'context':context, 'question':improved_query})
    
    return final_prompt

def generation(url, query):
    
    llm_model = model() # Initialize the model object
    prompt = augmentation(url, query, llm_model) # Getting the prompt 
    parser = StrOutputParser() # Parse the output

    chain = llm_model | parser 
    try:
        result = chain.invoke(prompt)
    except Exception as e:
        result = f"Error during generation {e}"
    
    return result

def chat_with_youtube(url, query):
    if not url or not query:
        return "Please enter both a valid YouTube URL and a query."
    try:
        answer = generation(url, query)
        return answer
    except Exception as e:
        return f"Error: {e}"
    
if __name__ == "__main__" :
    with gr.Blocks() as demo:
        gr.Markdown("# YouTube Transcript Chatbot")
        gr.Markdown("Enter a YouTube video URL and your question to get answers based on the transcript.")
        
        with gr.Row():
            url_input = gr.Textbox(label="YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
            query_input = gr.Textbox(label="Your Question", placeholder="Ask something about the video")
        
        output = gr.Textbox(label="Answer", lines=10)
        
        submit_btn = gr.Button("Ask")
        submit_btn.click(chat_with_youtube, inputs=[url_input, query_input], outputs=output)

    # This makes the app work on Render.com / Hugging Face Spaces
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 8080)))
