# Mistral API libraries
from mistralai import Mistral
from pathlib import Path
from mistralai import DocumentURLChunk

## langchain libraries
from langchain.docstore.document import Document
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore


## extra
import json
import os

os.environ["MISTRAL_API_KEY"] = "t9SeAwzKnmPA3UyEdlNONQsz5Vrj9IHr"
class MistralPDFProcessor:
    def __init__(self, emb_model, file_path, file_mapper):
        self.emb_model = emb_model
        self.file_path = file_path
        self.file_mapper = file_mapper  # Instance of FilePathMapper
        self.json_path, self.faiss_path = self.file_mapper.get_faiss_json_file_paths(file_path)

    def get_ocr_response(self):
        print(f"Calling Mistral OCR for: {self.file_path}")
        client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

        pdf_file = Path(self.file_path)
        assert pdf_file.is_file(), f"File not found: {pdf_file}"

        uploaded_file = client.files.upload(
            file={"file_name": pdf_file.stem, "content": pdf_file.read_bytes()},
            purpose="ocr",
        )

        signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

        response = client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-latest",
            include_image_base64=True
        )

        return json.loads(response.model_dump_json())

    def process_pdf(self):
        # Step 1: Load or generate OCR JSON
        if not os.path.exists(self.json_path):
            response_dict = self.get_ocr_response()
            os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
            with open(self.json_path, "w") as f:
                json.dump(response_dict, f)
        else:
            print("Loading OCR JSON from disk.")
            with open(self.json_path, "r") as f:
                response_dict = json.load(f)

        # Step 2: Convert OCR response to documents
        parent_documents = []
        for page in response_dict["pages"]:
            content = page["markdown"]
            metadata = {"page_index": page["index"], "source": f"page_{page['index']}"}
            parent_documents.append(Document(page_content=content, metadata=metadata))

        splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)

        child_documents = []
        for parent in parent_documents:
            children = splitter.split_documents([parent])
            for child in children:
                child.metadata["parent_source"] = parent.metadata["source"]
                child.metadata["page_index"] = parent.metadata["page_index"]
            child_documents.extend(children)

        # Step 3: Load or create FAISS index
        if os.path.exists(self.faiss_path):
            print("Loading FAISS index from disk.")
            vectorstore = FAISS.load_local(self.faiss_path, self.emb_model, allow_dangerous_deserialization=True)
        else:
            print("Creating new FAISS index.")
            vectorstore = FAISS.from_documents(child_documents, self.emb_model)
            os.makedirs(os.path.dirname(self.faiss_path), exist_ok=True)
            vectorstore.save_local(self.faiss_path)

        # Step 4: Create in-memory docstore and retriever
        docstore = InMemoryStore()
        docstore.mset([(doc.metadata["source"], doc) for doc in parent_documents])

        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            child_splitter=splitter,
            child_to_parent=lambda doc: doc.metadata["parent_source"],
            search_type="mmr",
            search_kwargs={"k": 10},
        )

        return retriever
