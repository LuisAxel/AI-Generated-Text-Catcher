from flask import Flask, request, jsonify
from flask_cors import CORS
from io import BytesIO
from model_utils import load_model_and_tokenizer, DistilBERTClass, test_essay, initialize_model
import PyPDF2
import torch

# Global variables to store the model and tokenizer
model = None
tokenizer = None
device = torch.device('cpu')

# Initialize model outside of main block for compatibility with production servers
model, tokenizer = initialize_model(model, tokenizer, device)

# Initialize Flask app
predict_app = Flask(__name__)
CORS(predict_app)

def extract_text_from_pdf(pdf_file: bytes) -> str:
    """
    Extracts text from a PDF file.

    Args:
        pdf_file (bytes): The PDF file content as bytes.

    Returns:
        str: Extracted text from the PDF, or None if extraction fails.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None


def extract_essay_from_request(request: request) -> tuple:
    """
    Extracts the essay content from a request, either from a file or JSON.

    Args:
        request (flask.Request): The HTTP request object.

    Returns:
        tuple: A tuple containing:
            - str: The extracted essay content, or None if extraction fails.
            - flask.Response: An error response, or None if no error occurred.
            - int: An HTTP status code, or None if no error occurred.
    """
    essay = None

    if 'file' in request.files:
        file = request.files['file']
        if not file.filename.endswith('.pdf'):
            return None, jsonify({"error": "Only PDF files are supported"}), 400

        pdf_text = extract_text_from_pdf(file.read())
        if not pdf_text:
            return None, jsonify({"error": "Unable to extract text from PDF"}), 400
        essay = pdf_text

    elif request.is_json:
        essay_data = request.get_json()
        essay = essay_data.get('essay', "").strip()
        if not essay:
            return None, jsonify({"error": "Essay content cannot be empty"}), 400

    else:
        return None, jsonify({"error": "No essay or PDF file provided"}), 400

    return essay, None, None


@predict_app.route('/predict', methods=['POST'])
def predict() -> tuple:
    """
    Handles the prediction API route, processing an essay or PDF file to generate predictions.

    Returns:
        tuple: A tuple containing:
            - flask.Response: JSON response with predictions or an error message.
            - int: HTTP status code.
    """
    if model is None or tokenizer is None:
        return jsonify({"error": "Model or tokenizer not loaded. Please try again later."}), 500

    essay, error_response, error_code = extract_essay_from_request(request)
    if error_code:
        return error_response, error_code

    try:
        predictions = test_essay(essay, model, tokenizer, 20, device)
        predictions = [tensor.item() for tensor in predictions]
        predictions_rounded = [round(prediction) for prediction in predictions]

        average_prediction = sum(predictions) / len(predictions)
        authentic_percentage = average_prediction * 100
        generated_percentage = 100 - authentic_percentage

        return jsonify({
            "essay": essay,
            "predictions": predictions,
            "predictions_rounded": predictions_rounded,
            "authentic_percentage": authentic_percentage,
            "generated_percentage": generated_percentage
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "An error occurred while processing the essay."}), 500


if __name__ == '__main__':
    predict_app.run(debug=False, host='0.0.0.0', port=8000)
