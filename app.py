from flask import Flask, request, jsonify
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.llms.ollama import Ollama
from torch.cuda import is_available
import os
import logging
import base64
from flask_cors import CORS
from paddleocr import PaddleOCR
import cv2
import numpy as np
import logging
logging.getLogger('ppocr').setLevel(logging.ERROR)

# Flask app setup
app = Flask(__name__)
CORS(app)
ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True)
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global settings
llm = Ollama(base_url="http://localhost:11434", model="llama3.1")
strict_conversation_history = {}  # Separate history for strict mode
flexible_conversation_history = {}  # Separate history for flexible mode

# # Prompts
# strict_prompt = ChatPromptTemplate.from_template(
#     """
#     Answer questions using only the provided context and conversation history.
#     If the question is unrelated to the context, respond with a brief apology text like I'm sorry, but I don't have information regarding that. and suggest exploring products/services. 
#     Keep off-topic responses at least 30 words, and avoid phrases like "documents" or "provided context" or "context".
#     Please provide the most relevant response in no more than 50 words.
#     For greetings or non-questions, give a short, friendly reply.
#     <context>
#     {context}
#     </context>
#     <history>
#     {history}
#     </history>
#     Question: {input}
#     """
# )

# flexible_prompt = ChatPromptTemplate.from_template(
#     """
#     Answer questions using only the provided context and conversation history:
#     - Please provide the most relevant response in no more than 50 words
#     If the question is unrelated to the context but can be answered with general knowledge:
#     - Present information assertively without hedging phrases
#     - Use formal but accessible language
#     - Include relevant dates/terms of service
#     - Maintain neutral tone
#     Please provide the most relevant response in no more than 50 words.
#     For greetings or non-questions, give a short, friendly reply.
#     <context>
#     {context}
#     </context>
#     <history>
#     {history}
#     </history>
#     Question: {input}
#     """
# )



strict_prompt = ChatPromptTemplate.from_template(
    """
    Answer questions using only the provided context and conversation history:
    - Please provide the most relevant response in no more than 50 words,encourage the user to ask further questions related to the context by adding: "Feel free to ask more about [specific topic/product/service]!"
    If the question is unrelated to the context, respond with: "I'm sorry, but I don't have information regarding that. Could you ask something related to [specific topic/product/service]?"
    - For greetings or non-questions, give a short, friendly reply.
    - After providing an answer, encourage the user to ask further questions related to the context by adding: "Feel free to ask more about [specific topic/product/service]!"
 
    <context>
    {context}
    </context>
    <history>
    {history}
    </history>
    Question: {input}
    """
)
 
flexible_prompt = ChatPromptTemplate.from_template(
    """
    Answer questions using only the provided context and conversation history:
    - Please provide the most relevant response in no more than 50 words,encourage the user to ask further questions related to the context by adding: "Feel free to ask more about [specific topic/product/service]!"
    If the question is unrelated to the context but can be answered with general knowledge:
    - Present information assertively without hedging phrases
    - Use formal but accessible language
    - Include relevant dates/terms of service
    - Maintain neutral tone
    -encourage the user to ask further questions related to the context by adding: "Feel free to ask more about [specific topic/product/service]!"
    For greetings or non-questions, give a short, friendly reply.
    <context>
    {context}
    </context>
    <history>
    {history}
    </history>
    Question: {input}
    """
)







class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L12-v2"):
        device = "cuda" if is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False, show_progress_bar=True)

    def embed_query(self, text):
        return self.model.encode(text, convert_to_tensor=False, show_progress_bar=True)


@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """
    Endpoint to upload a Base64-encoded PDF, save it, and generate embeddings.
    """
    data = request.get_json(force=True)
    user_code = data.get("User_Code")
    token = data.get("Token")
    base64_pdf = data.get("pdf_base64")
    filename = data.get("filename")
    file_extension = data.get("file_extension")

    if not user_code or not token:
        return jsonify({
            "status": "error",
            "message": "User_Code and Token are required"
        })

    if not base64_pdf:
        return jsonify({
            "status": "error",
            "message": "PDF Base64 string is required"
        })

    if not filename or not file_extension:
        return jsonify({
            "status": "error",
            "message": "Both filename and file_extension are required"
        })

    if not file_extension.startswith('.'):
        file_extension = f".{file_extension}"

    # Define the directory structure
    user_dir = os.path.join("./clients", user_code)
    token_dir = os.path.join(user_dir, token)
    pdf_directory = os.path.join(token_dir, "pdfs")
    faiss_path = os.path.join(token_dir, "faiss_index")

    os.makedirs(pdf_directory, exist_ok=True)
    os.makedirs(faiss_path, exist_ok=True)

    try:
        full_filename = f"{filename}{file_extension}"
        pdf_path = os.path.join(pdf_directory, full_filename)

        # Save the Base64 PDF
        with open(pdf_path, "wb") as pdf_file:
            pdf_file.write(base64.b64decode(base64_pdf))
        logging.info(f"PDF saved at: {pdf_path}")

        # Load and process PDF
        loader = PyPDFDirectoryLoader(pdf_directory)
        docs = loader.load()

        if not docs:
            raise ValueError(f"No documents found in directory: {pdf_directory}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
        split_docs = text_splitter.split_documents(docs)

        if not split_docs:
            raise ValueError(f"No valid split documents created from {pdf_path}")

        # Generate embeddings and save FAISS index
        embeddings = LocalEmbeddings(model_name="all-MiniLM-L12-v2")
        vectors = FAISS.from_documents(split_docs, embeddings)
        vectors.save_local(faiss_path)
        logging.info(f"FAISS index updated and saved at: {faiss_path}")

        return jsonify({
            "status": "success",
            "message": "PDF uploaded, processed, and embeddings updated successfully"
        })
    except Exception as e:
        logging.error(f"Error processing uploaded PDF: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to process PDF: {str(e)}"
        })


@app.route('/delete_pdf', methods=['POST'])
def delete_pdf():
    """
    Delete PDF by name and regenerate FAISS index with remaining PDFs if needed.
    """
    data = request.get_json()
    user_code = data.get('User_Code')
    token = data.get('Token')
    pdf_name = data.get('pdf_name')

    if not pdf_name:
        return jsonify({
            "status": "error",
            "message": "pdf_name is required"
        }), 400

    base_dir = "./clients"
    deleted_files = 0
    regenerated_indexes = 0
    deleted_indexes = 0

    try:
        # Determine search scope
        user_search_paths = []
        if user_code:
            user_search_paths.append(os.path.join(base_dir, user_code))
        else:
            user_search_paths = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
                               if os.path.isdir(os.path.join(base_dir, d))]

        for user_path in user_search_paths:
            if not os.path.isdir(user_path):
                continue

            # Get tokens to search
            token_search_paths = []
            if token:
                token_search_paths.append(os.path.join(user_path, token))
            else:
                token_search_paths = [os.path.join(user_path, d) for d in os.listdir(user_path) 
                                    if os.path.isdir(os.path.join(user_path, d))]

            for token_path in token_search_paths:
                if not os.path.isdir(token_path):
                    continue

                # Paths to check
                pdf_dir = os.path.join(token_path, "pdfs")
                faiss_dir = os.path.join(token_path, "faiss_index")
                target_pdf = os.path.join(pdf_dir, pdf_name)

                # Delete PDF if exists
                if os.path.isfile(target_pdf):
                    try:
                        os.remove(target_pdf)
                        deleted_files += 1
                        logging.info(f"Deleted PDF: {target_pdf}")
                    except Exception as e:
                        logging.error(f"Error deleting PDF {target_pdf}: {str(e)}")
                        continue

                    # Check remaining PDFs and regenerate index if needed
                    remaining_pdfs = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
                    if remaining_pdfs:
                        try:
                            # Reload and process remaining PDFs
                            loader = PyPDFDirectoryLoader(pdf_dir)
                            docs = loader.load()
                            if not docs:
                                logging.warning(f"No documents found in {pdf_dir} after deletion")
                                # Delete FAISS index as no valid documents
                                if os.path.exists(faiss_dir):
                                    for f in os.listdir(faiss_dir):
                                        os.remove(os.path.join(faiss_dir, f))
                                    os.rmdir(faiss_dir)
                                    deleted_indexes += 1
                                continue

                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
                            split_docs = text_splitter.split_documents(docs)
                            if not split_docs:
                                logging.warning(f"No split documents in {pdf_dir} after deletion")
                                # Delete FAISS index as no valid splits
                                if os.path.exists(faiss_dir):
                                    for f in os.listdir(faiss_dir):
                                        os.remove(os.path.join(faiss_dir, f))
                                    os.rmdir(faiss_dir)
                                    deleted_indexes += 1
                                continue

                            # Regenerate embeddings and save new FAISS index
                            embeddings = LocalEmbeddings(model_name="all-MiniLM-L6-v2")
                            vectors = FAISS.from_documents(split_docs, embeddings)
                            vectors.save_local(faiss_dir)
                            logging.info(f"Regenerated FAISS index at {faiss_dir} after deletion")
                            regenerated_indexes += 1
                        except Exception as e:
                            logging.error(f"Error regenerating FAISS index for {pdf_dir}: {str(e)}")
                    else:
                        # No remaining PDFs, delete FAISS index
                        if os.path.exists(faiss_dir):
                            try:
                                for f in os.listdir(faiss_dir):
                                    os.remove(os.path.join(faiss_dir, f))
                                os.rmdir(faiss_dir)
                                deleted_indexes += 1
                                logging.info(f"Deleted FAISS index at {faiss_dir} as no PDFs remain")
                            except Exception as e:
                                logging.error(f"Error deleting FAISS index: {str(e)}")

        return jsonify({
            "status": "success",
            "message": "PDF deletion completed",
            "deleted_files": deleted_files,
            "regenerated_indexes": regenerated_indexes,
            "deleted_indexes": deleted_indexes
        })

    except Exception as e:
        logging.error(f"PDF deletion failed: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Deletion process failed: {str(e)}"
        })


@app.route('/ask_question', methods=['POST'])
def ask_question():
    """
    Handle question-answering requests with isolated chat history per user_token.
    """
    data = request.get_json(force=True)
    user_code = data.get("User_Code")
    token = data.get("Token")
    user_token = data.get("User_Token")  # Unique identifier for user session
    use_external = data.get("Condition", False)
    question = data.get("Question")
    greeting = data.get("GreetingMessage")
    

    # Validate required fields
    if not all([user_code, token, user_token]):
        return jsonify({
            "status": "error",
            "message": "User_Code, Token, and user_token are required",
            "answer": ""
        })

    if not question:
        return jsonify({
            "status": "error",
            "message": "Question is required",
            "answer": ""
        })

    # Determine the FAISS path
    faiss_path = os.path.join("./clients", user_code, token, "faiss_index")
    if not os.path.exists(os.path.join(faiss_path, "index.faiss")):
        return jsonify({
            "status": "error",
            "message": f"No FAISS index found for user {user_code}",
            "answer": ""
        })

    try:
        # Create a unique session key
        session_key = f"{user_code}_{token}_{user_token}"

        # Select the appropriate conversation history based on the mode
        if use_external:
            history_dict = strict_conversation_history
            current_prompt = strict_prompt
        else:
            history_dict = flexible_conversation_history
            current_prompt = flexible_prompt

        # Initialize history if needed
        if session_key not in history_dict:
            history_dict[session_key] = []

        # Get the current conversation history (last 2 interactions)
        history = history_dict[session_key][-2:]

        # Load embeddings and FAISS index
        embeddings = LocalEmbeddings(model_name="all-MiniLM-L6-v2")
        vectors = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        retriever = vectors.as_retriever()

        # Retrieve relevant documents for the question
        retrieved_docs = retriever.invoke(question)
        context = "\n".join([doc.page_content for doc in retrieved_docs])  # Combine documents into a single context string

        # Format the input for the retrieval chain
        input_data = {
            "context": context,  # Use retrieved documents as context
            "history": "\n".join(history),  # Pass only the last 2 interactions as history
            "input": question
        }

        # Create and invoke the retrieval chain
        document_chain = create_stuff_documents_chain(llm, current_prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke(input_data)
        answer = response.get('answer', "No answer found")

        # Remove undesired phrases from the answer
        undesired_phrases = [
            "According to the provided context,",
            "Based on the provided context,",
            "According to the context,",
            "As per the context,",
            "According to the provided table,"
        ]
        for phrase in undesired_phrases:
            if answer.startswith(phrase):
                answer = answer[len(phrase):].strip()

        # Update the appropriate conversation history
        history_dict[session_key].append(f"User: {question}")
        history_dict[session_key].append(f"Assistant: {answer}")

        # Limit history to the last 10 interactions (optional)
        if len(history_dict[session_key]) > 10:
            history_dict[session_key] = history_dict[session_key][-10:]

        return jsonify({
            "status": "success",
            "message": "",
            "answer": answer
        })
    except Exception as e:
        logging.error(f"Error processing question for session {session_key}: {e}")
        return jsonify({
            "status": "error",
            "message": "Failed to process the question",
            "answer": ""
        })

def decode_base64_image(base64_string):
    """Convert base64 string to an OpenCV image."""
    try:
        image_data = base64.b64decode(base64_string)
        image_np = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        return None



def perform_ocr(img):
    """Perform OCR on the selected image and store bounding box information."""
    ocr_results = ocr.ocr(img)
    # print("ocr_Image:",ocr_results)
    data = []
    for line in ocr_results:
 
        for word_info in line:
            bbox = word_info[0]  # Bounding box coordinates
            text = word_info[1][0]  # Extracted text
            confidence = word_info[1][1]  # Confidence score
            x, y, w, h = cv2.boundingRect(np.array(bbox, dtype=np.int32))
            data.append((x, y, w, h, text, confidence))
    # print("Data:",data)
    return data


@app.route("/ocr", methods=["POST"])
def ocr_api():
    """API endpoint for OCR using Base64 image input."""
    data = request.json
    if "image" not in data:
        return jsonify({"error": "No image data provided"}), 400

    base64_image = data["image"]
    image = decode_base64_image(base64_image)

    if image is None:
        return jsonify({"error": "Invalid Base64 image"}), 400

    # Perform OCR
    ocr_results = perform_ocr(image)
    # print("ocr_results:",ocr_results)
    return ocr_results

if __name__ == '__main__':
    # Start Flask app
    app.run(host='0.0.0.0', port=5011)