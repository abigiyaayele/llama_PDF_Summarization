from PyPDF2 import PdfReader
from fastapi import HTTPException
from ollama import ChatResponse,chat
class PDFProcessor:

    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        """
        Extracts text from an uploaded PDF file.
        :param pdf_file: Uploaded PDF file (FastAPI UploadFile object)
        :return: Extracted text as a string
        """
        try:
            reader = PdfReader(pdf_file.file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            return text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error extracting text from PDF: {str(e)}")

    @staticmethod
    def chunk_text(text: str, max_chunk_size: int = 2048) -> list:
        """
        Splits a large text into smaller chunks for processing.
        :param text: The input text to be chunked.
        :param max_chunk_size: Maximum size of each chunk.
        :return: A list of text chunks.
        """
        words = text.split()
        chunks = []
        current_chunk = []

        for word in words:
            if len(' '.join(current_chunk)) + len(word) + 1 <= max_chunk_size:
                current_chunk.append(word)
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    @staticmethod
    def summarize_with_ollama(text: str) -> str:
        """
        Uses the Ollama API to summarize the input text.
        :param input_text: The text to be summarized
        :return: Summarized text
        """
        SYSTEM_PROMPT = """
          You are an advanced AI PDF summarizer. Your goal is to summarize the <content> tags  of a PDF document and generate a concise, structured, and accurate summary. Follow these principles:

          1. Focus on the key points and important details relevant to the title and subtitles.
          2. Use bullet points to organize the summary for clarity and readability.
          3. Eliminate redundant or irrelevant information.
          4. Preserve the original context and meaning of the content.
          5. If the document has no clear title or subtitles, summarize based on the overall content.

          Your summaries should be professional, precise, and easy to understand.
        """
          
        USER_PROMPT = f"""
        Summarize it concisely based on the title and subtitles provided in pdf <content> tags, and organize the summary into bullet points. 
        Ensure the summary is aligned with the main ideas and structured clearly.
        <content>
        {text}
        </content>

        #### Instructions:
        - Summarize the content in 2-5 bullet points per subtitle or section.
        - Focus on key points and eliminate unnecessary details.
        - Use the title and subtitles as a guide to prioritize and organize the information.
        - If the content is fragmented or incomplete, summarize based on the most coherent information available.
        """

        try:
            response: ChatResponse = chat(
                model="llama3.1",  
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT},
                ],
            )
            
            summarized_text = response.get("message", {}).get("content", "")
            formatted_summary = summarized_text.replace("\n", "\n\n") 
            formatted_summary = formatted_summary.replace("•", "\n•") 
        
            return formatted_summary
       
        except Exception as e:
            # Handle errors and raise an HTTPException for API-related issues
            raise HTTPException(status_code=500, detail=f"Error summarizing text with Ollama: {str(e)}")
    
        