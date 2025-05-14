import os
import base64
import uuid
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, Table, Column, String, MetaData, select, desc, DateTime, func
from sqlalchemy.orm import sessionmaker, Session
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Database Setup ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/db_name")
if DATABASE_URL == "postgresql://user:password@localhost/db_name":
    logger.warning("DATABASE_URL not set. Using default placeholder.")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
metadata = MetaData()

stories_table = Table(
    "stories", metadata,
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

# --- App Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up and creating DB tables...")
    try:
        metadata.create_all(bind=engine)
    except Exception as e:
        logger.error(f"Error setting up DB: {e}")
    yield
    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="html"), name="static")

client = OpenAI()

system_message = """You are a helpful assistant tasked with analyzing traditional Emirati children's drawings.
These drawings capture elements of the rich cultural heritage, local traditions, and daily life of the UAE as interpreted by children.
Your role is to create a cohesive, imaginative story inspired by the visuals...
The story should be in {language}.
"""

# --- Dependencies ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def encode_image(image_data):
    return base64.b64encode(image_data).decode("utf-8")

# --- Routes ---

@app.api_route("/", methods=["GET", "HEAD"])
async def serve_homepage():
    return FileResponse("html/app.html")

@app.get("/history", response_class=FileResponse)
async def serve_history_page():
    return FileResponse("html/history.html")

@app.get("/story-history")
def story_history(limit: int = Query(10, ge=1, le=100), offset: int = Query(0, ge=0),
                  include_images: bool = Query(True), db: Session = Depends(get_db)):
    try:
        columns = [
            stories_table.c.id, stories_table.c.keywords,
            stories_table.c.created_at, stories_table.c.language,
            stories_table.c.student_name, stories_table.c.school_name, stories_table.c.class_name
        ]
        if include_images:
            columns.insert(2, stories_table.c.image_data)

        query = select(*columns).order_by(desc(stories_table.c.created_at)).limit(limit).offset(offset)
        result = db.execute(query)

        return [
            {
                "id": row.id,
                "keywords": row.keywords,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "language": row.language,
                "student_name": row.student_name or "",
                "school_name": row.school_name or "",
                "class_name": row.class_name or "",
                **({"image_data": row.image_data} if include_images else {})
            }
            for row in result.fetchall()
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/story/{story_id}")
def get_story(story_id: str, include_audio: bool = Query(False), db: Session = Depends(get_db)):
    try:
        columns = [
            stories_table.c.id, stories_table.c.image_data, stories_table.c.keywords,
            stories_table.c.story, stories_table.c.created_at, stories_table.c.language,
            stories_table.c.student_name, stories_table.c.school_name, stories_table.c.class_name
        ]
        if include_audio:
            columns.append(stories_table.c.audio_data)

        query = select(*columns).where(stories_table.c.id == story_id)
        row = db.execute(query).first()

        if not row:
            raise HTTPException(status_code=404, detail="Story not found")

        story = {
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
            story["audio_data"] = row.audio_data
        return story
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-image")
async def analyze_image(
    image: UploadFile = File(...),
    keywords: str = Form(...),
    language: str = Form("Arabic"),
    student_name: str = Form(""),
    school_name: str = Form(""),
    class_name: str = Form(""),
    db: Session = Depends(get_db)
):
    try:
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file is not an image")

        image_data = await image.read()
        if not image_data:
            raise HTTPException(status_code=400, detail="Empty image file")

        base64_img = encode_image(image_data)
        prompt = system_message.format(language=language)

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"The keywords are: {keywords}\n\n and here's the drawing:" if keywords else "Here's the drawing:"},
                        {"type": "image_url", "image_url": {"url": f"data:{image.content_type};base64,{base64_img}", "detail": "high"}}
                    ]
                }
            ],
        )

        story_content = completion.choices[0].message.content
        story_id = str(uuid.uuid4())

        db.execute(stories_table.insert().values(
            id=story_id,
            image_data=base64_img,
            keywords=keywords,
            story=story_content,
            language=language,
            student_name=student_name,
            school_name=school_name,
            class_name=class_name
        ))
        db.commit()

        return {"story": story_content, "id": story_id}

    except Exception as e:
        logger.error(f"Unexpected error in analyze_image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during image analysis: {str(e)}")

@app.delete("/story/{story_id}")
def delete_story(story_id: str, db: Session = Depends(get_db)):
    try:
        result = db.execute(stories_table.delete().where(stories_table.c.id == story_id))
        db.commit()
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Story not found")
        return {"message": "Deleted successfully", "id": story_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Dev Server Run ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
