import os
import base64
import uuid
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, Table, Column, String, MetaData, select, desc, DateTime, func
from sqlalchemy.orm import sessionmaker, Session
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Database Setup ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/db_name")
if DATABASE_URL == "postgresql://user:password@localhost/db_name":
    logger.warning("DATABASE_URL not set, using default local placeholder. Ensure it's set in your environment for Render.")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
metadata = MetaData()

# Define database table schema
stories_table = Table(
    "stories",
    metadata,
    Column("id", String, primary_key=True),
    Column("image_data", String),
    Column("keywords", String),
    Column("story", String),
    Column("created_at", DateTime, default=func.now(), server_default=func.now()),
    Column("language", String),
    Column("audio_data", String),
    Column("student_name", String),
    Column("school_name", String),
    Column("class_name", String)
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
    yield
    logger.info("Application shutdown.")

app = FastAPI(lifespan=lifespan)

# Mount static files directory
app.mount("/static", StaticFiles(directory="html"), name="static")

# Initialize OpenAI client
client = OpenAI()

# --- System Prompt for GPT ---
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

# --- DB Dependency ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Helpers ---
def encode_image(image_data):
    return base64.b64encode(image_data).decode("utf-8")

# --- DB Functions ---
def save_story_db(db: Session, story_id: str, image_data: str, keywords: str, story_content: str, language: str,
                  student_name: str = "", school_name: str = "", class_name: str = "") -> None:
    try:
        ins = stories_table.insert().values(
            id=story_id,
            image_data=image_data,
            keywords=keywords,
            story=story_content,
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
        logger.error(f"Error saving story to DB: {str(e)}")
        raise

def save_audio_data_db(db: Session, story_id: str, audio_data: bytes) -> None:
    try:
        audio_data_b64 = base64.b64encode(audio_data).decode("utf-8")
        upd = stories_table.update().where(stories_table.c.id == story_id).values(audio_data=audio_data_b64)
        result = db.execute(upd)
        db.commit()
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Story not found to save audio")
    except Exception as e:
        db.rollback()
        raise

def get_story_history_db(db: Session, limit: int = 10, offset: int = 0, include_images: bool = True) -> List[dict]:
    try:
        columns_to_select = [
            stories_table.c.id, stories_table.c.keywords,
            stories_table.c.created_at, stories_table.c.language,
            stories_table.c.student_name, stories_table.c.school_name, stories_table.c.class_name
        ]
        if include_images:
            columns_to_select.insert(2, stories_table.c.image_data)

        query = select(*columns_to_select).order_by(desc(stories_table.c.created_at)).limit(limit).offset(offset)
        result_proxy = db.execute(query)

        story_list = []
        for row in result_proxy.fetchall():
            story_dict = {
                "id": row.id,
                "keywords": row.keywords,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "language": row.language or "Arabic",
                "student_name": row.student_name or "",
                "school_name": row.school_name or "",
                "class_name": row.class_name or ""
            }
            if include_images:
                story_dict["image_data"] = row.image_data
            story_list.append(story_dict)
        return story_list
    except Exception as e:
        logger.error(f"Error retrieving story history from DB: {str(e)}")
        return []

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
        row = result_proxy.first()

        if not row:
            return None

        story_dict = {
            "id": row.id,
            "image_data": row.image_data,
            "keywords": row.keywords,
            "story": row.story,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "language": row.language,
            "student_name": row.student_name or "",
            "school_name": row.school_name or "",
            "class_name": row.class_name or ""
        }
        if include_audio:
            story_dict["audio_data"] = row.audio_data
        return story_dict
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
        raise

# --- API Endpoints ---
@app.get("/")
def root():
    return {"message": "Welcome to the Hekayah API. Visit /story-history to view stories."}

@app.get("/story-history")
def story_history_endpoint(limit: int = Query(10, ge=1, le=100), offset: int = Query(0, ge=0),
                           include_images: bool = Query(True), db: Session = Depends(get_db)):
    try:
        return get_story_history_db(db, limit, offset, include_images)
    except Exception as e:
        logger.error(f"Error in story-history endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving story history")

@app.get("/story/{story_id}")
def get_story_endpoint(story_id: str, include_audio: bool = Query(False), db: Session = Depends(get_db)):
    story_data = get_story_by_id_db(db, story_id, include_audio)
    if not story_data:
        raise HTTPException(status_code=404, detail="Story not found")
    return story_data

@app.post("/analyze-image")
async def analyze_image_endpoint(
    image: UploadFile = File(...),
    keywords: str = Form(...),
    language: str = Form("Arabic"),
    student_name: str = Form(""),
    school_name: str = Form(""),
    class_name: str = Form(""),
    db: Session = Depends(get_db)
):
    try:
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Uploaded file is not an image")

        image_data = await image.read()
        if not image_data:
            raise HTTPException(status_code=400, detail="Empty image file")

        base64_image = encode_image(image_data)
        formatted_system_message = system_message.format(language=language)

        completion = client.chat.completions.create(
            model="gpt-4o",
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
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
        )

        story_content = completion.choices[0].message.content
        story_id = str(uuid.uuid4())
        save_story_db(db, story_id, base64_image, keywords, story_content, language, student_name, school_name, class_name)

        return {"story": story_content, "id": story_id}
    except Exception as e:
        logger.error(f"Error in analyze_image: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing image and generating story.")

@app.delete("/story/{story_id}")
def delete_story_endpoint(story_id: str, db: Session = Depends(get_db)):
    success = delete_story_by_id_db(db, story_id)
    if not success:
        raise HTTPException(status_code=404, detail="Story not found or could not be deleted")
    return {"message": "Story deleted successfully", "id": story_id}

# --- Run Locally ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server at http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
