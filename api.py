from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, HTTPException, Form


from filepath_mapper import *
from pdf_processor import *
from query_handler import *
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

app = FastAPI()

class SummaryRequest(BaseModel):
    chapter: str

    
@app.post("/generate_chapter_summary/")
async def generate_summary(file: UploadFile = File(...), save_path: str = Form(...)):
    """
    Takes a PDF file and database save path as input, returns summary as PPT
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        
        # Create permanent path for the uploaded file in the specified directory
        uploaded_file_path = os.path.join(os.getcwd(), "db", "raw_db", save_path)

        if not os.path.exists(uploaded_file_path):
            os.mkdir(uploaded_file_path)

        uploaded_file_path = os.path.join(uploaded_file_path,  file.filename)

        
        # Save uploaded file to the specified path
        with open(uploaded_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        try:
            # Initialize models
            llm = ChatMistralAI(model="mistral-large-latest", temperature=0.1)
            emb_model = MistralAIEmbeddings()
            
            # Initialize file mapper
            file_mapper = FilePathMapper(base_dir=os.getcwd())
            
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

            return(summary["summary"])
            
            
            
        except Exception as processing_error:
            raise processing_error
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/")
def root():
    return {"message": "✅ FastAPI backend is running!"}

@app.get("/favicon.ico")
async def favicon():
    return {}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
