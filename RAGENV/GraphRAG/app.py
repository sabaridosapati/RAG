"""
Gradio UI Application for RAG System.
Provides interface for uploading PDFs and querying the knowledge base.
"""

import gradio as gr
import os
from rag_orchestrator import RAGOrchestrator
from config import Config

# Initialize RAG System
print("Starting RAG System...")
rag = RAGOrchestrator()

def upload_and_process_pdf(pdf_file):
    """
    Handle PDF upload and processing.
    
    Args:
        pdf_file: Uploaded PDF file object
        
    Returns:
        Status message and database info
    """
    if pdf_file is None:
        return "❌ Please upload a PDF file", get_database_status()
    
    try:
        # Get the file path from the uploaded file
        pdf_path = pdf_file.name
        
        # Process the PDF
        result = rag.process_and_store_pdf(pdf_path)
        
        if result["success"]:
            stats = result["statistics"]
            message = f"""✅ **PDF Processed Successfully!**
            
📄 **File:** {os.path.basename(pdf_path)}
📊 **Chunks Created:** {result['chunks_processed']}
📏 **Average Chunk Size:** {stats['avg_chunk_size']} characters
📈 **Total Characters:** {stats['total_chars']}

Your document has been indexed and is ready for querying!"""
        else:
            message = f"❌ **Processing Failed:** {result['message']}"
        
        return message, get_database_status()
        
    except Exception as e:
        error_msg = f"❌ **Error processing PDF:** {str(e)}"
        return error_msg, get_database_status()

def query_rag_system(question, top_k):
    """
    Handle user queries to the RAG system.
    
    Args:
        question: User question
        top_k: Number of chunks to retrieve
        
    Returns:
        Answer and formatted sources
    """
    if not question or not question.strip():
        return "❌ Please enter a question", ""
    
    try:
        # Query the RAG system
        result = rag.query(question, top_k=int(top_k))
        
        # Format the answer
        answer = f"**Answer:**\n\n{result['answer']}"
        
        # Format the sources
        sources_text = "**Sources:**\n\n"
        for i, source in enumerate(result['sources'], 1):
            sources_text += f"**{i}. {source['source']}** (Chunk {source['chunk_index']})\n"
            sources_text += f"   - Similarity Score: {source['similarity']:.4f}\n"
            sources_text += f"   - Preview: {source['text'][:200]}...\n\n"
        
        return answer, sources_text
        
    except Exception as e:
        error_msg = f"❌ **Error processing query:** {str(e)}"
        return error_msg, ""

def get_database_status():
    """
    Get current database status.
    
    Returns:
        Formatted database information
    """
    try:
        info = rag.get_database_info()
        
        status = f"""📊 **Database Status**

📚 **Total Documents:** {info['total_documents']}
📄 **Total Chunks:** {info['total_chunks']}

**Indexed Documents:**
"""
        if info['documents']:
            for doc in info['documents']:
                status += f"  • {doc}\n"
        else:
            status += "  (No documents yet)\n"
        
        return status
        
    except Exception as e:
        return f"❌ Error retrieving database status: {str(e)}"

def delete_document_handler(document_name):
    """
    Handle document deletion.
    
    Args:
        document_name: Name of document to delete
        
    Returns:
        Status message and updated database info
    """
    if not document_name or not document_name.strip():
        return "❌ Please enter a document name", get_database_status()
    
    try:
        rag.delete_document(document_name.strip())
        message = f"✅ Successfully deleted all chunks from: {document_name}"
        return message, get_database_status()
        
    except Exception as e:
        error_msg = f"❌ Error deleting document: {str(e)}"
        return error_msg, get_database_status()

# Create Gradio Interface
# Note: theme parameter removed for compatibility with older Gradio versions
with gr.Blocks(title="RAG System - PDF Knowledge Base") as demo:
    
    gr.Markdown("""
    # 📚 RAG System - PDF Knowledge Base
    
    Upload PDF documents and query them using AI-powered semantic search with Gemini!
    """)
    
    with gr.Tabs():
        
        # Tab 1: Upload Documents
        with gr.Tab("📤 Upload Documents"):
            gr.Markdown("### Upload and Process PDF Documents")
            
            with gr.Row():
                with gr.Column(scale=2):
                    pdf_input = gr.File(
                        label="Upload PDF File",
                        file_types=[".pdf"],
                        type="filepath"
                    )
                    upload_btn = gr.Button("🚀 Process PDF", variant="primary", size="lg")
                    
                with gr.Column(scale=1):
                    db_status = gr.Markdown(get_database_status())
            
            upload_status = gr.Markdown()
            
            upload_btn.click(
                fn=upload_and_process_pdf,
                inputs=[pdf_input],
                outputs=[upload_status, db_status]
            )
        
        # Tab 2: Query System
        with gr.Tab("🔍 Query Knowledge Base"):
            gr.Markdown("### Ask Questions About Your Documents")
            
            with gr.Row():
                with gr.Column():
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="What would you like to know about the documents?",
                        lines=3
                    )
                    
                    with gr.Row():
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Number of chunks to retrieve"
                        )
                        query_btn = gr.Button("🔍 Search", variant="primary", size="lg")
            
            answer_output = gr.Markdown(label="Answer")
            sources_output = gr.Markdown(label="Sources")
            
            query_btn.click(
                fn=query_rag_system,
                inputs=[question_input, top_k_slider],
                outputs=[answer_output, sources_output]
            )
            
            # Add example questions
            gr.Examples(
                examples=[
                    ["What are the main topics discussed in the document?"],
                    ["Summarize the key findings"],
                    ["What conclusions are drawn?"],
                ],
                inputs=question_input
            )
        
        # Tab 3: Manage Database
        with gr.Tab("⚙️ Manage Database"):
            gr.Markdown("### Database Management")
            
            with gr.Row():
                with gr.Column():
                    current_db_status = gr.Markdown(get_database_status())
                    refresh_btn = gr.Button("🔄 Refresh Status")
                    
                with gr.Column():
                    gr.Markdown("### Delete Document")
                    delete_input = gr.Textbox(
                        label="Document Name",
                        placeholder="Enter exact document name to delete"
                    )
                    delete_btn = gr.Button("🗑️ Delete Document", variant="stop")
                    delete_status = gr.Markdown()
            
            refresh_btn.click(
                fn=get_database_status,
                outputs=[current_db_status]
            )
            
            delete_btn.click(
                fn=delete_document_handler,
                inputs=[delete_input],
                outputs=[delete_status, current_db_status]
            )
        
        # Tab 4: System Info
        with gr.Tab("ℹ️ System Info"):
            gr.Markdown(f"""
            ### RAG System Configuration
            
            **Models:**
            - Embedding Model: `{rag.config.embedding_model}`
            - Embedding Dimensions: `{rag.config.embedding_dimensions}`
            - Generation Model: `{rag.config.generation_model}` (Gemini 2.5 Flash-Lite)
            
            **Database:**
            - Neo4j URI: `{rag.config.neo4j_uri}`
            - Database: `{rag.config.neo4j_database}`
            
            **Settings:**
            - Minimum Chunk Length: {rag.config.chunk_min_length} characters
            - Default Top-K Results: {rag.config.top_k_results}
            - Max API Retries: {rag.config.max_retries}
            
            **Features:**
            - ✅ PDF extraction with Docling
            - ✅ Vector embeddings with Gemini
            - ✅ Neo4j graph database storage
            - ✅ Dynamic Neo4j Cypher retrieval (vector + graph expansion)
            - ✅ AI-powered responses with Gemini 2.5 Flash-Lite + dynamic thinking budget
            
            **Retrieval:**
            - Source-aware query filtering when document names are mentioned
            - Configurable graph hop expansion for GraphRAG context enrichment
            """)

# Launch the application
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Launching Gradio UI...")
    print("="*60)
    
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,        # Default Gradio port
        share=False,             # Set to True to create public link
        show_error=True
    )



