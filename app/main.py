import os
import base64
import uuid
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, Table, Column, String, MetaData, select, desc, DateTime, func
from sqlalchemy.orm import sessionmaker, Session
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

load_dotenv()

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Database Setup ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/db_name")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
metadata = MetaData()

stories_table = Table(
    "stories", metadata,
    Column("id", String, primary_key=True),
    Column("image_data", String),
    Column("keywords", String),
    Column("story", String),
    Column("created_at", DateTime, default=func.now()),
    Column("language", String),
    Column("audio_data", String),
    Column("student_name", String),
    Column("school_name", String),
    Column("class_name", String)
)

# --- FastAPI App ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        metadata.create_all(bind=engine)
        logger.info("Database initialized.")
    except Exception as e:
        logger.error(f"DB init error: {e}")
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="html"), name="static")

client = OpenAI()

# --- Prompts ---
system_message = """
You are a helpful assistant tasked with analyzing traditional Emirati children's drawings...
The story should be in {language}.
"""

# --- Helpers ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def encode_image(img_bytes):
    return base64.b64encode(img_bytes).decode("utf-8")

# --- Routes ---
@app.api_route("/", methods=["GET", "HEAD"])
async def home():
    return FileResponse("html/app.html")

@app.get("/history", response_class=FileResponse)
async def history_page():
    return FileResponse("html/history.html")

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

        logger.info("Sending image to OpenAI...")
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"The keywords are: {keywords}"},
                        {"type": "image_url", "image_url": {"url": f"data:{image.content_type};base64,{base64_img}", "detail": "high"}}
                    ]
                }
            ],
        )

        story = completion.choices[0].message.content
        story_id = str(uuid.uuid4())

        db.execute(stories_table.insert().values(
            id=story_id,
            image_data=base64_img,
            keywords=keywords,
            story=story,
            language=language,
            student_name=student_name,
            school_name=school_name,
            class_name=class_name
        ))
        db.commit()

        return {"story": story, "id": story_id}

    except OpenAIError as oe:
        logger.error(f"OpenAI Error: {oe}")
        raise HTTPException(status_code=503, detail=f"OpenAI error: {str(oe)}")
    except Exception as e:
        logger.error(f"Internal error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# --- Story history route (optional) ---
@app.get("/story-history")
def story_history(limit: int = 10, db: Session = Depends(get_db)):
    query = select(stories_table).order_by(desc(stories_table.c.created_at)).limit(limit)
    return [dict(row._mapping) for row in db.execute(query)]

# --- Dev only ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
