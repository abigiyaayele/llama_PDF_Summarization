from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from llama_pdf import PDFProcessor

app = FastAPI()

@app.post("/summarize")
async def summarize_pdf(file: UploadFile = File(...)):
    """
    Endpoint to summarize the content of an uploaded PDF file.
    """
    #  file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException("Only PDF files are supported.")

    try:
        pdf_text = PDFProcessor.extract_text_from_pdf(file)
        chunks = PDFProcessor.chunk_text(pdf_text)
        summarized_chunks = []
        for chunk in chunks:
            summarized_chunk = PDFProcessor.summarize_with_ollama(chunk)
            summarized_chunks.append(summarized_chunk)

        summarized_text = " ".join(summarized_chunks)

        return JSONResponse(content={"summary": summarized_text})
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")