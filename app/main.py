import os
import base64
import uuid
from typing import List
from sqlalchemy import Table, Column, String, MetaData, select, text, desc

from dbos import DBOS
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
DBOS(fastapi=app)
app.mount("/static", StaticFiles(directory="html"), name="static")

client = OpenAI()
metadata = MetaData()

# === Database Table ===
stories = Table(
    "stories", metadata,
    Column("id", String, primary_key=True),
    Column("image_data", String),
    Column("keywords", String),
    Column("story", String),
    Column("created_at", String),
    Column("language", String),
    Column("audio_data", String),
    Column("student_name", String),
    Column("school_name", String),
    Column("class_name", String)
)

# === System Prompt ===
system_message = """
You are a helpful assistant tasked with analyzing traditional Emirati children's drawings.
These drawings capture elements of the rich cultural heritage, local traditions, and daily life of the UAE as interpreted by children.
Your role is to create a cohesive, imaginative story inspired by the visuals.
Ensure the story is in {language} and enjoyable by teenagers.
Return ONLY the story text. No extra text.
"""

# === Helpers ===
def encode_image(image_data):
    return base64.b64encode(image_data).decode("utf-8")

# === Save Story with FIFO Deletion ===
@DBOS.transaction()
def save_story_with_limit(story_id, image_data, keywords, story, language, student_name, school_name, class_name):
    DBOS.sql_session.execute(
        stories.insert().values(
            id=story_id,
            image_data=image_data,
            keywords=keywords,
            story=story,
            created_at=text("NOW()"),
            language=language,
            student_name=student_name,
            school_name=school_name,
            class_name=class_name
        )
    )

    count = DBOS.sql_session.execute(select([text("COUNT(*)")]).select_from(stories)).scalar()
    if count > 20:
        old_ids = DBOS.sql_session.execute(
            select([stories.c.id]).order_by(stories.c.created_at).limit(count - 20)
        ).fetchall()
        for row in old_ids:
            DBOS.logger.info(f"Deleting old story ID {row.id}")
            DBOS.sql_session.execute(stories.delete().where(stories.c.id == row.id))

# === Routes ===

@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return FileResponse("html/app.html")

@app.get("/history", response_class=FileResponse)
def history_page():
    return FileResponse("html/history.html")

@app.get("/story-history")
def get_story_history(limit: int = Query(10, ge=1, le=100)):
    rows = DBOS.sql_session.execute(
        select([
            stories.c.id,
            stories.c.keywords,
            stories.c.language,
            stories.c.created_at,
            stories.c.student_name,
            stories.c.school_name,
            stories.c.class_name,
            stories.c.image_data
        ]).order_by(desc(stories.c.created_at)).limit(limit)
    ).fetchall()

    return [dict(row._mapping) for row in rows]

@app.get("/story/{story_id}")
def get_story_by_id(story_id: str):
    row = DBOS.sql_session.execute(
        select([stories]).where(stories.c.id == story_id)
    ).first()

    if not row:
        raise HTTPException(status_code=404, detail="Story not found")

    return dict(row._mapping)

@app.post("/analyze-image")
async def analyze_image(
    image: UploadFile = File(...),
    keywords: str = Form(...),
    language: str = Form("Arabic"),
    student_name: str = Form(""),
    school_name: str = Form(""),
    class_name: str = Form("")
):
    try:
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file is not an image")

        image_data = await image.read()
        base64_image = encode_image(image_data)
        prompt = system_message.format(language=language)

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",  # safer for all keys
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"The keywords are: {keywords}"},
                        {"type": "image_url", "image_url": {
                            "url": f"data:{image.content_type};base64,{base64_image}",
                            "detail": "high"
                        }}
                    ]
                }
            ]
        )

        story_text = completion.choices[0].message.content
        story_id = str(uuid.uuid4())

        save_story_with_limit(
            story_id, base64_image, keywords, story_text, language,
            student_name, school_name, class_name
        )

        return {"id": story_id, "story": story_text}

    except OpenAIError as e:
        DBOS.logger.error(f"OpenAI error: {e}")
        raise HTTPException(status_code=503, detail=f"OpenAI error: {str(e)}")
    except Exception as e:
        DBOS.logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.delete("/story/{story_id}")
def delete_story(story_id: str):
    result = DBOS.sql_session.execute(stories.delete().where(stories.c.id == story_id))
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Story not found")
    return {"message": "Deleted", "id": story_id}
