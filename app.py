from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
import gradio as gr
import os

def document_loader(file):
    loader = PyPDFLoader(file)
    loaded_doc = loader.load()
    return loaded_doc

def text_splitter(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    return splitter.split_documents(text)

def embeddings():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embedding_model

def vector_database(chunks):
    embedding_model = embeddings()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever

def get_llm():
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.7,
        api_key=os.environ.get("GROQ_API_KEY")
    )
    return llm

def retriever_qa(file, query, chat_history=[]):
    llm = get_llm()
    retriever_object = retriever(file)
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever_object,
    )
    response = qa.invoke({
        "question": query,
        "chat_history": []
    })
    return response['answer']

with gr.Blocks() as Rag_application:
    gr.HTML("""
        <div>
            <h1 style='font-size: 2em; color: #6366f1;'>RAG Q&A Assistant</h1>
            <p style='color: #6b7280;'>Upload a PDF and chat about your document!</p>
        </div>
    """)
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload PDF",
                file_types=['.pdf'],
                type="filepath",
            )
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=450,
            )
            with gr.Row():
                query_input = gr.Textbox(
                    placeholder="Ask something about your PDF...",
                    show_label=False,
                    lines=1,
                    scale=4
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear History", variant="secondary")


    def respond(file, query, history):
        answer = retriever_qa(file, query, history)
        history = history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer}
        ]
        return history, ""

    submit_btn.click(fn=respond, inputs=[file_input, query_input, chatbot], outputs=[chatbot, query_input])
    query_input.submit(fn=respond, inputs=[file_input, query_input, chatbot], outputs=[chatbot, query_input])
    clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, query_input])

Rag_application.launch(theme=gr.themes.Soft())