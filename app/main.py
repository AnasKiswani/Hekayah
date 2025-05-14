import os
import base64
import uuid
import logging
from typing import List, Optional
from contextlib import asynccontextmanager
from datetime import datetime

from sqlalchemy import create_engine, Table, Column, String, MetaData, select, desc, DateTime, func, text
from sqlalchemy.orm import sessionmaker, Session
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Depends
from fastapi.responses import JSONResponse, FileResponse # Added FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI, APIError as OpenAI_APIError # Renamed to avoid conflict if user has local APIError
from dotenv import load_dotenv

load_dotenv()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Database Setup ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/db_name")
if DATABASE_URL == "postgresql://user:password@localhost/db_name":
    logger.warning("DATABASE_URL environment variable not set, using default local placeholder. Ensure it's set in your Render environment.")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
metadata = MetaData()

# Define database table schema
stories_table = Table(
    "stories",
    metadata,
    Column("id", String, primary_key=True),
    Column("image_data", String), # Consider TEXT type for large base64 strings if issues arise
    Column("keywords", String),
    Column("story", String), # Consider TEXT type for long stories
    Column("created_at", DateTime, default=func.now(), server_default=func.now()),
    Column("language", String),
    Column("audio_data", String, nullable=True), # Consider TEXT type, make nullable if not always present
    Column("student_name", String, nullable=True),
    Column("school_name", String, nullable=True),
    Column("class_name", String, nullable=True)
)

# --- FastAPI Lifespan for DB Initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: Creating database tables if they don't exist...")
    try:
        metadata.create_all(bind=engine)
        logger.info("Database tables checked/created.")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        # Depending on policy, you might want to raise this or handle it
    yield
    logger.info("Application shutdown.")

app = FastAPI(lifespan=lifespan)

# Mount static files directory (assuming 'html' is at the project root where uvicorn runs)
# This needs to be relative to the directory where uvicorn is started.
# If your Procfile is `web: uvicorn app.main:app ...` and main.py is in `app/`,
# and `html/` is at the project root (sibling to `app/`), then the path for StaticFiles
# should be relative from the project root, so `directory="html"` is correct if uvicorn runs from project root.
app.mount("/static", StaticFiles(directory="html"), name="static")

# Initialize OpenAI client (API key from OPENAI_API_KEY env var)
client = OpenAI()

# --- System Message for OpenAI ---
system_message = """You are a helpful assistant tasked with analyzing traditional Emirati children's drawings.
These drawings capture elements of the rich cultural heritage, local traditions, and daily life of the UAE as interpreted by children.
Your role is to create a cohesive, imaginative story inspired by the visuals. The story should reflect the cultural significance of the elements in the drawings while maintaining a sense of wonder and adventure appropriate for children.
Ensure the narrative is engaging, fun, and educational, incorporating aspects of the Emirati heritage.

## Instructions
- NEVER (DO NOT) say that you can't help (or anything related to that) or can't identify the elements or people in the drawing. always generate a story, even when there's no keywords.
- If the user inserted keywords, use them to generate the story, and know the keywords are inserted by teenagers not children, so generate a story that teenagers will enjoy.
- Return only the story, no other text or comments.
- The story should be in {language}.
- Ensure the story uses proper {language} grammar and vocabulary.

"""

# --- Dependency to get DB session ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Helper Functions ---
def encode_image(image_data):
    return base64.b64encode(image_data).decode("utf-8")

# --- Database Operations (Refactored) ---
def save_story_db(db: Session, story_id: str, image_data: str, keywords: str, story_content: str, language: str,
                  student_name: Optional[str] = None, school_name: Optional[str] = None, class_name: Optional[str] = None) -> None:
    try:
        ins = stories_table.insert().values(
            id=story_id,
            image_data=image_data,
            keywords=keywords,
            story=story_content,
            # created_at is handled by default=func.now()
            language=language,
            student_name=student_name,
            school_name=school_name,
            class_name=class_name
        )
        db.execute(ins)
        db.commit()
        logger.info(f"Story saved with ID: {story_id}")
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving story to DB (ID: {story_id}): {str(e)}")
        raise # Re-raise the exception to be handled by the endpoint

def save_audio_data_db(db: Session, story_id: str, audio_data: bytes) -> None:
    try:
        audio_data_b64 = base64.b64encode(audio_data).decode("utf-8")
        upd = stories_table.update().where(stories_table.c.id == story_id).values(audio_data=audio_data_b64)
        result = db.execute(upd)
        db.commit()
        if result.rowcount == 0:
            logger.warning(f"Attempted to save audio for non-existent story ID: {story_id}")
            raise HTTPException(status_code=404, detail=f"Story with ID {story_id} not found to save audio for.")
        logger.info(f"Audio data saved for story ID: {story_id}")
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving audio data to DB for story ID {story_id}: {str(e)}")
        raise

def get_story_history_db(db: Session, limit: int = 10, offset: int = 0, include_images: bool = True) -> List[dict]:
    try:
        columns_to_select = [
            stories_table.c.id, stories_table.c.keywords,
            stories_table.c.created_at, stories_table.c.language,
            stories_table.c.student_name, stories_table.c.school_name, stories_table.c.class_name
        ]
        if include_images:
            columns_to_select.insert(2, stories_table.c.image_data) # Insert image_data at the correct position

        query = select(*columns_to_select).order_by(desc(stories_table.c.created_at)).limit(limit).offset(offset)
        result_proxy = db.execute(query)
        
        story_list = []
        # Using .mappings().all() can be cleaner for converting rows to dicts
        for row_mapping in result_proxy.mappings().all():
            row_dict = dict(row_mapping)
            # Ensure created_at is ISO format string if it exists
            if row_dict.get("created_at") and isinstance(row_dict["created_at"], datetime):
                 row_dict["created_at"] = row_dict["created_at"].isoformat()
            row_dict.setdefault("language", "Arabic") # Ensure language has a default
            story_list.append(row_dict)
        return story_list
    except Exception as e:
        logger.error(f"Error retrieving story history from DB: {str(e)}")
        # It's better to raise the error to the endpoint to return a 500, rather than an empty list on error.
        raise

def get_story_by_id_db(db: Session, story_id: str, include_audio: bool = False) -> Optional[dict]:
    try:
        columns_to_select = [
            stories_table.c.id, stories_table.c.image_data, stories_table.c.keywords,
            stories_table.c.story, stories_table.c.created_at, stories_table.c.language,
            stories_table.c.student_name, stories_table.c.school_name, stories_table.c.class_name
        ]
        if include_audio:
            columns_to_select.append(stories_table.c.audio_data)

        query = select(*columns_to_select).where(stories_table.c.id == story_id)
        result_proxy = db.execute(query)
        row_mapping = result_proxy.mappings().first() # .first() returns one or None

        if not row_mapping:
            return None
        
        row_dict = dict(row_mapping)
        if row_dict.get("created_at") and isinstance(row_dict["created_at"], datetime):
            row_dict["created_at"] = row_dict["created_at"].isoformat()
        return row_dict
    except Exception as e:
        logger.error(f"Error retrieving story by ID ({story_id}) from DB: {str(e)}")
        raise

def delete_story_by_id_db(db: Session, story_id: str) -> bool:
    try:
        dele = stories_table.delete().where(stories_table.c.id == story_id)
        result = db.execute(dele)
        db.commit()
        return result.rowcount > 0
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting story ID ({story_id}) from DB: {str(e)}")
        raise

# --- API Endpoints (Refactored) ---

# Endpoint to serve app.html from the root path
@app.get("/", response_class=FileResponse)
async def read_index():
    # Assumes 'html' folder is at the project root, and uvicorn runs from project root.
    return "html/app.html"

@app.get("/story-history")
def story_history_endpoint(limit: int = Query(10, ge=1, le=100), offset: int = Query(0, ge=0),
                           include_images: bool = Query(True), db: Session = Depends(get_db)):
    # Input validation for limit/offset is good, already present in original.
    # The original code had some logic to default limit/offset if out of bounds, which is fine.
    # if limit < 1: limit = 10 # This logic is not standard, Query(ge=1) handles it.
    # elif limit > 100: limit = 100
    # if offset < 0: offset = 0
    try:
        stories_list = get_story_history_db(db, limit=limit, offset=offset, include_images=include_images)
        return stories_list
    except Exception as e:
        logger.error(f"Error in story-history endpoint: {str(e)}")
        # Check for specific DB errors if needed, e.g., disk space, but generic 500 is okay.
        # The original code had specific handling for "No space left on device", which is good practice
        # but requires parsing the error string. For now, a generic 500.
        if "No space left on device" in str(e):
             logger.critical(f"DATABASE ERROR: No space left on device. {e}")
             raise HTTPException(status_code=503, detail="Database storage is full. Please contact the administrator.")
        raise HTTPException(status_code=500, detail="Internal server error retrieving story history.")

@app.get("/story/{story_id}")
def get_story_endpoint(story_id: str, include_audio: bool = Query(False), db: Session = Depends(get_db)):
    try:
        story_data = get_story_by_id_db(db, story_id, include_audio)
        if not story_data:
            raise HTTPException(status_code=404, detail=f"Story with ID {story_id} not found")
        return story_data
    except HTTPException: # Re-raise HTTPExceptions (like 404)
        raise
    except Exception as e:
        logger.error(f"Error in get_story endpoint for ID {story_id}: {str(e)}")
        if "No space left on device" in str(e):
             logger.critical(f"DATABASE ERROR: No space left on device. {e}")
             raise HTTPException(status_code=503, detail="Database storage is full. Please contact the administrator.")
        raise HTTPException(status_code=500, detail=f"Internal server error retrieving story ID {story_id}.")

@app.post("/analyze-image")
async def analyze_image_endpoint(
    image: UploadFile = File(...),
    keywords: str = Form(...),
    language: str = Form("Arabic"),
    student_name: Optional[str] = Form(None), # Use Optional and default to None
    school_name: Optional[str] = Form(None),
    class_name: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    try:
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image type")

        image_data = await image.read()
        if not image_data:
            raise HTTPException(status_code=400, detail="Empty image file uploaded")

        base64_image = encode_image(image_data)
        logger.info(f"Analyzing image. Language: {language}, Keywords: '{keywords}', Student: '{student_name}'")

        formatted_system_message = system_message.format(language=language)
        
        completion = client.chat.completions.create(
            model="gpt-4o", # Using model from user's provided file
            messages=[
                {"role": "system", "content": formatted_system_message},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"The keywords are: {keywords}\n\n and here's the drawing:" if keywords else "Here's the drawing:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image.content_type};base64,{base64_image}",
                                "detail": "high", # Consider 'low' if high detail is not needed and to save costs/time
                            },
                        },
                    ],
                }
            ],
        )
        story_content = completion.choices[0].message.content
        if not story_content:
            logger.warning("OpenAI returned an empty story.")
            # Decide how to handle empty story - raise error or return specific message
            raise HTTPException(status_code=503, detail="AI service generated an empty story.")
            
        story_id = str(uuid.uuid4())

        save_story_db(db, story_id, base64_image, keywords, story_content, language, student_name, school_name, class_name)
        
        return {"story": story_content, "id": story_id}

    except OpenAI_APIError as e: # Specific OpenAI error
        logger.error(f"OpenAI API error during image analysis: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Error communicating with AI service: {str(e)}")
    except HTTPException: # Re-raise HTTPExceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in analyze_image endpoint: {str(e)}")
        if "No space left on device" in str(e):
             logger.critical(f"DATABASE ERROR: No space left on device. {e}")
             raise HTTPException(status_code=503, detail="Database storage is full. Please contact the administrator.")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during image analysis: {str(e)}")

@app.delete("/story/{story_id}")
def delete_story_endpoint(story_id: str, db: Session = Depends(get_db)):
    try:
        logger.info(f"Attempting to delete story with ID: {story_id}")
        success = delete_story_by_id_db(db, story_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Story with ID {story_id} not found or could not be deleted")
        return {"message": f"Story with ID {story_id} deleted successfully", "id": story_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete_story endpoint for ID {story_id}: {str(e)}")
        if "No space left on device" in str(e):
             logger.critical(f"DATABASE ERROR: No space left on device. {e}")
             raise HTTPException(status_code=503, detail="Database storage is full. Please contact the administrator.")
        raise HTTPException(status_code=500, detail=f"Internal server error deleting story ID {story_id}.")

# The following is for local development convenience if you run `python app/main.py`
# Render will use the Procfile: `web: uvicorn app.main:app --host 0.0.0.0 --port $PORT`
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for local development on http://0.0.0.0:8000")
    # Ensure uvicorn runs from the project root if this file is in app/main.py
    # For `uvicorn app.main:app`, the current directory when running this script doesn't matter as much
    # as the directory from which the uvicorn command itself is launched.
    # If running `python app/main.py`, then HTML path for FileResponse might need adjustment if not run from project root.
    # However, for Render, the Procfile command `uvicorn app.main:app` is run from the project root.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, app_dir="app") # Assuming main.py is in app/

