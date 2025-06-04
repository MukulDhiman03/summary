from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, HTTPException, Form

import tempfile
from filepath_mapper import *
from pdf_processor import *
from query_handler import *
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings


app = FastAPI()

class SummaryRequest(BaseModel):
    chapter: str

    
@app.post("/generate_chapter_summary/")
@app.post("/generate_chapter_summary/")
async def generate_summary(file: UploadFile = File(...), save_path: str = Form(...)):
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        uploaded_file_path = os.path.join(temp_dir, file.filename)
        
        # Save uploaded file to temp path
        with open(uploaded_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        try:
            # Initialize models
            llm = ChatMistralAI(model="mistral-large-latest", temperature=0.1)
            emb_model = MistralAIEmbeddings()
            
            # Initialize file mapper with temp dir
            file_mapper = FilePathMapper(base_dir=temp_dir)
            
            # Process PDF
            processor = MistralPDFProcessor(
                emb_model=emb_model,
                file_path=str(uploaded_file_path),
                file_mapper=file_mapper
            )
            
            retriever = processor.process_pdf()
            
            # Generate summary
            handler = PromptQueryHandler(llm=llm, retriever=retriever, templates=template_dict)
            summary = handler.get_summary()

            # Clean up temp files
            os.remove(uploaded_file_path)
            
            return summary["summary"]
            
        except Exception as processing_error:
            raise HTTPException(status_code=500, detail=f"Processing error: {str(processing_error)}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/")
def root():
    return {"message": "✅ FastAPI backend is running!"}

@app.get("/favicon.ico")
async def favicon():
    return {}


class FilePathMapper:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.db_dir = "db"
        self.raw_db_dir = "raw_db"
        self.json_db_dir = "json_db"
        self.faiss_db_dir = "faiss_db"
        self.ppt_db_dir = "ppt_db"

        # Construct full paths
        self.faiss_db_dir_path = os.path.join(self.base_dir, self.db_dir, self.faiss_db_dir)
        self.raw_db_dir_path = os.path.join(self.base_dir, self.db_dir, self.raw_db_dir)
        self.json_db_dir_path = os.path.join(self.base_dir, self.db_dir, self.json_db_dir)
        self.ppt_db_dir_path = os.path.join(self.base_dir, self.db_dir, self.ppt_db_dir)

        # Create ALL required directories
        for dir_path in [self.faiss_db_dir_path, 
                        self.json_db_dir_path,
                        self.raw_db_dir_path,
                        self.ppt_db_dir_path]:
            os.makedirs(dir_path, exist_ok=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
