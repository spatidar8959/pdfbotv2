from flask import Flask, request, jsonify, session
import secrets
from deep_translator import GoogleTranslator
from PyPDF2 import PdfReader
from flask_cors import CORS  
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
from threading import Lock


load_dotenv()
lock = Lock()
app = Flask(__name__)
CORS(app)
app.secret_key = os.urandom(24)

api_key = os.environ["OPENAI_API_KEY"]

language = 'english'
embeddings = None
db = None
new_db = None
llm_chain = None
pdf_processing_complete = False
store_path = ""
user_id = None

template = """
You are helpful AI question answering Assistant.You read the context provide by user and generate answer of the user query.
If user ask question out of context tell the user "This is not given in the context" and make the answer by your own.And dont disclose 
internal information.explain every user query or question in brief details.

# Context: {context}
# Question: {query}
# Answer: """


prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template=template
)
language_model = ChatOpenAI(model_name="gpt-3.5-turbo-0125")

llm_chain = LLMChain(
    llm=language_model,
    prompt=prompt_template,
    verbose=False,
)

ALLOWED_EXTENSIONS = {'pdf','txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    global language, embeddings, db, pdf_processing_complete

    with lock:
        try:
            pdf_file = request.files.get('pdf_file')
            store_path = os.path.splitext(os.path.basename(pdf_file.filename))[0]
            if not pdf_file or pdf_file.filename == '':
                return jsonify({'error': 'Please provide a PDF file in the "pdf_file" form field.'}), 400
            
            if not allowed_file(pdf_file.filename):
                return jsonify({'error': 'Only PDF and TEXT files are allowed.'}), 400

            user_id = secrets.token_hex(16)
            session['user_id'] = user_id

            pdfreader = PdfReader(pdf_file)
            raw_text = ''
            for i, page in enumerate(pdfreader.pages):
                content = page.extract_text()
                if content:
                    raw_text += content

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=10,
                length_function=len,
                is_separator_regex=False,
            )
            texts = text_splitter.split_text(raw_text)

            embeddings = OpenAIEmbeddings()
            db = FAISS.from_texts(texts, embeddings)
            db.save_local(f"embeddings/{user_id}")
            pdf_processing_complete = True

            return jsonify({'message': 'Document processed successfully', 'user_id': user_id})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/ask_question', methods=['POST'])
def ask_question():
    global pdf_processing_complete ,language
    with lock:
        if not pdf_processing_complete:
            return jsonify({'error': 'Document processing is not complete. Please process the Document first.'}), 400

        user_question = request.form.get('question', '')
        user_id = request.form.get('user_id', '')
        language = request.form.get('language', '').lower()
        
        if language == "":
            language = 'english'

        # stored_user_id = session.get('user_id', '')
        # if user_id != stored_user_id:
        #     return jsonify({'error': 'Invalid user ID.'}), 400
        
        db = session.get('db')

        if not user_question:
            return jsonify({'error': 'Please provide a question in the "question" form field.'}), 400

        query = user_question
        new_db = FAISS.load_local(f"embeddings/{user_id}", embeddings)
        docs = new_db.similarity_search(query, k=3)
        response = llm_chain.predict(context=docs, query=query)
        translated = GoogleTranslator(source='auto', target=language).translate(response)
        return jsonify({'response': translated})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
