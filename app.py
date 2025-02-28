from flask import Flask, request, jsonify
import base64
import pymupdf
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.ERROR)

@app.route('/extract_pdf_pages', methods=['POST'])
def extract_pdf_pages():
    data = request.get_json(force=True)
    base64_pdf = data.get("pdf_base64")

    if not base64_pdf:
        return jsonify({
            "status": "error",
            "message": "PDF Base64 string is required"
        }), 400

    try:
        # Decode the Base64 PDF
        pdf_bytes = base64.b64decode(base64_pdf)

        # Open the PDF using PyMuPDF
        pdf_document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        pages_dict = {}  # Dictionary to store pages

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            pages_dict[f"page_{page_num + 1}"] = img_base64  # Add page number as key

        pdf_document.close()

        return jsonify({
            "status": "success",
            "message": "PDF pages extracted successfully",
            "pages": pages_dict  # Return dictionary
        })
    except Exception as e:
        logging.error(f"Error processing PDF: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to process PDF: {str(e)}"
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501)