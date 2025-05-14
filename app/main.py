import os
import base64
import uuid
from typing import List, Optional
from sqlalchemy import Table, Column, String, MetaData, select, text, desc

from dbos import DBOS
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Depends
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from dotenv import load_dotenv
from pyt2s.services import stream_elements

load_dotenv()

# Initialize Maydan Al Hekayah application with DBOS and FastAPI
app = FastAPI()
DBOS(fastapi=app)

# Mount static files directory to serve images
app.mount("/static", StaticFiles(directory="html"), name="static")

# Initialize OpenAI client
client = OpenAI()

# Define database table schema
stories = Table(
    "stories", 
    MetaData(), 
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

# Define the system message for OpenAI
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

# Helper function to encode image to base64
def encode_image(image_data):
    return base64.b64encode(image_data).decode("utf-8")

# Database transaction to save a story
@DBOS.transaction()
def save_story(story_id: str, image_data: str, keywords: str, story: str, language: str, 
               student_name: str = "", school_name: str = "", class_name: str = "") -> None:
    # Get current timestamp in ISO format
    timestamp = text("NOW()")
    
    # Insert the new story
    DBOS.sql_session.execute(
        stories.insert().values(
            id=story_id,
            image_data=image_data,
            keywords=keywords,
            story=story,
            created_at=timestamp,
            language=language,
            student_name=student_name,
            school_name=school_name,
            class_name=class_name
        )
    )
    
    # Enforce 20-story limit with FIFO (delete the oldest stories if exceeding the limit)
    enforce_story_limit()

# Database transaction to enforce a 20-story limit
@DBOS.transaction()
def enforce_story_limit(max_stories: int = 20) -> None:
    try:
        # Count the total number of stories
        count_result = DBOS.sql_session.execute(
            select([text("COUNT(*)")]).select_from(stories)
        ).scalar()
        
        # If we have more than the maximum allowed stories
        if count_result > max_stories:
            # Calculate how many stories need to be deleted
            stories_to_delete = count_result - max_stories
            
            # Get the IDs of the oldest stories that need to be deleted
            oldest_stories = DBOS.sql_session.execute(
                select([stories.c.id])
                .order_by(stories.c.created_at)
                .limit(stories_to_delete)
            ).fetchall()
            
            # Delete each of the oldest stories
            for story_row in oldest_stories:
                DBOS.logger.info(f"FIFO limit: Deleting old story with ID: {story_row[0]}")
                DBOS.sql_session.execute(
                    stories.delete().where(stories.c.id == story_row[0])
                )
            
            DBOS.logger.info(f"FIFO limit: Deleted {stories_to_delete} oldest stories to maintain 20-story limit")
    except OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise HTTPException(status_code=503, detail=f"⚠️ OpenAI error → {str(e)}")

    except Exception as e:
        DBOS.logger.error(f"Error enforcing story limit: {str(e)}")
        # Don't raise the exception to avoid disrupting the main story saving process
        pass

# Database transaction to save audio data for a story
@DBOS.transaction()
def save_audio_data(story_id: str, audio_data: bytes) -> None:
    # Encode audio data as base64
    audio_data_b64 = base64.b64encode(audio_data).decode("utf-8")
    
    DBOS.sql_session.execute(
        stories.update()
        .where(stories.c.id == story_id)
        .values(audio_data=audio_data_b64)
    )

# Database transaction to get story history with pagination support
@DBOS.transaction()
def get_story_history(limit: int = 10, offset: int = 0, include_images: bool = True) -> List[dict]:
    try:
        # Query with pagination and optional image data
        if include_images:
            result = DBOS.sql_session.execute(
                select(stories.c.id, stories.c.keywords, stories.c.image_data, stories.c.created_at, stories.c.language, 
                       stories.c.student_name, stories.c.school_name, stories.c.class_name)
                .order_by(desc(stories.c.created_at))
                .limit(limit).offset(offset)
            ).fetchall()
            
            # Return data including image_data
            return [{
                "id": row[0], 
                "keywords": row[1], 
                "image_data": row[2], 
                "created_at": row[3], 
                "language": row[4] or "Arabic",
                "student_name": row[5] or "",
                "school_name": row[6] or "",
                "class_name": row[7] or ""
            } for row in result]
        else:
            # Query without image data to reduce payload size
            result = DBOS.sql_session.execute(
                select(stories.c.id, stories.c.keywords, stories.c.created_at, stories.c.language,
                       stories.c.student_name, stories.c.school_name, stories.c.class_name)
                .order_by(desc(stories.c.created_at))
                .limit(limit).offset(offset)
            ).fetchall()
            
            # Return minimal data without images
            return [{
                "id": row[0], 
                "keywords": row[1], 
                "created_at": row[2], 
                "language": row[3] or "Arabic",
                "student_name": row[4] or "",
                "school_name": row[5] or "",
                "class_name": row[6] or ""
            } for row in result]
    except Exception as e:
        DBOS.logger.error(f"Error retrieving story history: {str(e)}")
        # Return empty list on error instead of failing
        return []

# Endpoint to get story history
@app.get("/story-history")
def get_story_history(limit: int = Query(10, ge=1, le=100), offset: int = Query(0, ge=0), include_images: bool = Query(False), db: Session = Depends(get_db)):
    try:
        columns = [
            stories_table.c.id,
            stories_table.c.keywords,
            stories_table.c.created_at,
            stories_table.c.language,
            stories_table.c.student_name,
            stories_table.c.school_name,
            stories_table.c.class_name
        ]
        if include_images:
            columns.insert(2, stories_table.c.image_data)
        query = select(*columns).order_by(desc(stories_table.c.created_at)).limit(limit).offset(offset)
        results = db.execute(query).fetchall()
        response = []
        for row in results:
            item = dict(row._mapping)
            response.append(item)
        return response
    except Exception as e:
        logger.error(f"Error in story-history: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

        # Ensure offset is non-negative
        if offset < 0:
            offset = 0
            
        stories = get_story_history(limit=limit, offset=offset, include_images=include_images)
        
        # Return empty list if no stories found
        if stories is None:
            return []
            
        return stories
    except Exception as e:
        # Log the error with details
        error_message = str(e)
        DBOS.logger.error(f"Error in story-history endpoint: {error_message}")
        
        # Check if it's a disk space error
        if "No space left on device" in error_message:
            DBOS.logger.critical("DATABASE ERROR: No space left on device. Please free up disk space.")
            # Return a more specific error for disk space issues
            return {"error": "Database storage is full. Please contact the administrator."}
        
        # For other errors, return an empty list instead of error
        return []

# Database transaction to get a specific story by ID
@DBOS.transaction()
def get_story_by_id(story_id: str, include_audio: bool = False) -> Optional[dict]:
    if include_audio:
        row = DBOS.sql_session.execute(
            select(stories).where(stories.c.id == story_id)
        ).first()
    else:
        # Exclude audio_data to reduce payload size
        row = DBOS.sql_session.execute(
            select(stories.c.id, stories.c.image_data, stories.c.keywords, 
                   stories.c.story, stories.c.created_at, stories.c.language,
                   stories.c.student_name, stories.c.school_name, stories.c.class_name)
            .where(stories.c.id == story_id)
        ).first()
    
    if not row:
        return None
    
    if include_audio:
        return {
            "id": row[0],
            "image_data": row[1],
            "keywords": row[2],
            "story": row[3],
            "created_at": row[4],
            "language": row[5],
            "audio_data": row[6],
            "student_name": row[7] or "",
            "school_name": row[8] or "",
            "class_name": row[9] or ""
        }
    else:
        return {
            "id": row[0],
            "image_data": row[1],
            "keywords": row[2],
            "story": row[3],
            "created_at": row[4],
            "language": row[5],
            "student_name": row[6] or "",
            "school_name": row[7] or "",
            "class_name": row[8] or ""
        }

# Endpoint to get a specific story by ID
@app.get("/story/{story_id}")
def get_story(story_id: str, include_audio: bool = Query(False)):
    try:
        story = get_story_by_id(story_id, include_audio)
        if not story:
            raise HTTPException(status_code=404, detail="Story not found")
        return story
    except HTTPException:
        # Re-raise HTTP exceptions like 404
        raise
    except Exception as e:
        # Log the error
        error_message = str(e)
        DBOS.logger.error(f"Error in get_story endpoint: {error_message}")
        
        # Check if it's a disk space error
        if "No space left on device" in error_message:
            DBOS.logger.critical("DATABASE ERROR: No space left on device. Please free up disk space.")
            return JSONResponse(
                status_code=500,
                content={"error": "Database storage is full. Please contact the administrator."}
            )
            
        # For other errors, return a generic error
        raise HTTPException(status_code=500, detail="Internal server error")

# Database transaction to delete a story by ID
@DBOS.transaction()
def delete_story_by_id(story_id: str) -> bool:
    result = DBOS.sql_session.execute(
        stories.delete().where(stories.c.id == story_id)
    )
    return result.rowcount > 0

# Endpoint to analyze the image and generate a story
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
        # Check if the image file is valid
        if not image.content_type.startswith('image/'):
            return JSONResponse(
                status_code=400,
                content={"error": "Uploaded file is not an image"}
            )
            
        # Read the uploaded image
        try:
            image_data = await image.read()
            if not image_data:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Empty image file"}
                )
        except Exception as e:
            DBOS.logger.error(f"Error reading image: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={"error": "Could not read image file"}
            )
        
        # Encode the image to base64
        try:
            base64_image = encode_image(image_data)
        except Exception as e:
            DBOS.logger.error(f"Error encoding image: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": "Could not process image"}
            )
        
        # Log the request
        DBOS.logger.info(f"Analyzing image for keywords: {keywords}, language: {language}")
        
        # Format system message with the selected language
        formatted_system_message = system_message.format(language=language)
        
        # Create the OpenAI API request
        try:
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
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
            )
            
            story = completion.choices[0].message.content
        except Exception as e:
            DBOS.logger.error(f"OpenAI API error: {str(e)}")
            return JSONResponse(
                status_code=503,
                content={"error": "Error communicating with AI service"}
            )
        
        # Generate a unique ID for the story
        story_id = str(uuid.uuid4())
        
        # Save the story to the database
        try:
            direct_save_story(story_id, base64_image, keywords, story, language, student_name, school_name, class_name)
        except Exception as e:
            error_message = str(e)
            DBOS.logger.error(f"Error saving story: {error_message}")
            
            # Check if it's a disk space error
            if "No space left on device" in error_message:
                DBOS.logger.critical("DATABASE ERROR: No space left on device. Please free up disk space.")
                return JSONResponse(
                    status_code=500,
                    content={"error": "Database storage is full. Please contact the administrator."}
                )
                
            return JSONResponse(
                status_code=500,
                content={"error": "Could not save story to database"}
            )
        
        # Return the generated story and its ID
        return {"story": story, "id": story_id}
    except Exception as e:
        # Catch all other exceptions
        DBOS.logger.error(f"Unexpected error in analyze_image: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "An unexpected error occurred"}
        )

# Endpoint to delete a story by ID
@app.delete("/story/{story_id}")
def delete_story(story_id: str):
    try:
        DBOS.logger.info(f"Deleting story with ID: {story_id}")
        success = delete_story_by_id(story_id)
        if not success:
            return {"success": False, "error": "Story not found"}
        return {"success": True}
    except Exception as e:
        # Log the error
        error_message = str(e)
        DBOS.logger.error(f"Error in delete_story endpoint: {error_message}")
        
        # Check if it's a disk space error
        if "No space left on device" in error_message:
            DBOS.logger.critical("DATABASE ERROR: No space left on device. Please free up disk space.")
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "Database storage is full. Please contact the administrator."}
            )
            
        # For other database errors
        return {"success": False, "error": "Database error occurred"}

# Endpoint to convert text to speech
@app.get("/tts/{story_id}")
def text_to_speech(story_id: str):
    try:
        # Get the story with audio data
        story = get_story_by_id(story_id, include_audio=True)
        if not story:
            raise HTTPException(status_code=404, detail="Story not found")
        
        # Check if we already have audio data saved for this story
        if story.get("audio_data"):
            try:
                # Decode the stored audio data
                audio_bytes = base64.b64decode(story["audio_data"])
                return Response(
                    content=audio_bytes,
                    media_type="audio/mpeg"
                )
            except Exception as e:
                DBOS.logger.error(f"Error decoding stored audio: {str(e)}")
                # Fall through to regenerate audio if there's an error
        
        # If no audio data is saved or there was an error, generate it
        # Get the text and language
        text = story["story"]
        language = story.get("language", "Arabic")
        
        try:
            # Select voice based on language
            if language == "Arabic":
                # Use an Arabic TTS voice
                audio_data = stream_elements.requestTTS(text, stream_elements.Voice.Hoda.value)
            else:
                # Use an English TTS voice
                audio_data = stream_elements.requestTTS(text, stream_elements.Voice.Amy.value)
            
            # Try to save the audio data to the database, but don't fail if it can't be saved
            try:
                save_audio_data(story_id, audio_data)
            except Exception as save_error:
                # Log the error but continue without saving
                DBOS.logger.error(f"Failed to save audio data: {str(save_error)}")
                # Check if it's a disk space error and log a more specific message
                if "No space left on device" in str(save_error):
                    DBOS.logger.critical("DATABASE ERROR: No space left on device. Please free up disk space.")
            
            # Return the audio file even if saving failed
            return Response(
                content=audio_data,
                media_type="audio/mpeg"
            )
        except Exception as e:
            DBOS.logger.error(f"TTS error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"TTS service error: {str(e)}")
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the error
        error_message = str(e)
        DBOS.logger.error(f"Error in text_to_speech endpoint: {error_message}")
        
        # Check if it's a disk space error
        if "No space left on device" in error_message:
            DBOS.logger.critical("DATABASE ERROR: No space left on device. Please free up disk space.")
            return JSONResponse(
                status_code=500,
                content={"error": "Database storage is full. Please contact the administrator."}
            )
            
        # For other errors, return a generic error
        raise HTTPException(status_code=500, detail="Internal server error")

# Serve the HTML frontend
@app.api_route("/", methods=["GET", "HEAD"])
def home():
    with open("html/app.html", "r", encoding="utf-8") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.get("/history")
def history_page():
    with open("html/history.html", "r", encoding="utf-8") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

# Database transaction to delete all stories
@DBOS.transaction()
def delete_all_stories() -> int:
    try:
        result = DBOS.sql_session.execute(
            stories.delete()
        )
        return result.rowcount
    except Exception as e:
        DBOS.logger.error(f"Error deleting all stories: {str(e)}")
        raise

# Admin endpoint to clear the database
@app.get("/admin/clear-database")
def clear_database():
    try:
        DBOS.logger.info("Applying 20-story FIFO limit to database")
        # Instead of clearing all stories, enforce the 20-story limit
        # This will delete only the oldest stories if exceeding 20
        count_result = DBOS.sql_session.execute(
            select([text("COUNT(*)")]).select_from(stories)
        ).scalar()
        
        # Enforce the story limit
        enforce_story_limit(max_stories=20)
        
        DBOS.logger.info(f"Successfully enforced 20-story limit, current count: {count_result}")
        return {"success": True, "message": f"Enforced 20-story limit. Database now contains the most recent stories."}
    except Exception as e:
        error_message = str(e)
        DBOS.logger.error(f"Error enforcing story limit: {error_message}")
        
        # Check if it's a disk space error
        if "No space left on device" in error_message:
            DBOS.logger.critical("DATABASE ERROR: No space left on device. Please free up disk space.")
            return JSONResponse(
                status_code=500, 
                content={"success": False, "error": "Database storage is full. Please contact the administrator."}
            )
        
        # For other errors
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Database error: {error_message}"}
        )

# Admin endpoint to vacuum the database
@app.get("/admin/vacuum-database")
def vacuum_database():
    try:
        DBOS.logger.info("Running VACUUM FULL on database")
        
        # Create database connection parameters from dbos-config.yaml
        db_host = "userdb-8ca4bad3-091a-4435-bd71-5ec99b2d6cfb.cvc4gmaa6qm9.us-east-1.rds.amazonaws.com"
        db_port = 5432
        db_user = "dbos_user"
        db_pass = "heritage2025"
        db_name = "postgres"
        
        # Use psycopg directly (version 3) instead of SQLAlchemy and psycopg2
        import psycopg
        
        DBOS.logger.info("Connecting to PostgreSQL database")
        # Connect with autocommit=True because VACUUM can't run inside a transaction
        conn = psycopg.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_pass,
            dbname=db_name,
            autocommit=True
        )
        
        try:
            with conn.cursor() as cur:
                # Execute vacuum command on the stories table only
                DBOS.logger.info("Vacuuming stories table")
                cur.execute("VACUUM FULL stories")
                
                # Also run ANALYZE to update statistics
                DBOS.logger.info("Analyzing stories table")
                cur.execute("ANALYZE stories")
                
                DBOS.logger.info("VACUUM FULL completed successfully")
                return {"success": True, "message": "Database maintenance completed successfully"}
        finally:
            conn.close()
            
    except Exception as e:
        error_message = str(e)
        DBOS.logger.error(f"Error vacuuming database: {error_message}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Database maintenance error: {error_message}"}
        )

# Direct database function to save a story (no DBOS transaction)
def direct_save_story(story_id: str, image_data: str, keywords: str, story: str, language: str, 
               student_name: str = "", school_name: str = "", class_name: str = "") -> None:
    
    # Create database connection parameters from dbos-config.yaml
    db_host = "userdb-8ca4bad3-091a-4435-bd71-5ec99b2d6cfb.cvc4gmaa6qm9.us-east-1.rds.amazonaws.com"
    db_port = 5432
    db_user = "dbos_user"
    db_pass = "heritage2025"
    db_name = "postgres"
    
    # Use psycopg directly
    import psycopg
    
    try:
        # Connect to database
        conn = psycopg.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_pass,
            dbname=db_name
        )
        
        # Create a cursor and begin transaction
        with conn:  # This automatically manages transactions
            with conn.cursor() as cur:
                # Insert the new story
                cur.execute(
                    """
                    INSERT INTO stories 
                    (id, image_data, keywords, story, created_at, language, student_name, school_name, class_name)
                    VALUES (%s, %s, %s, %s, NOW(), %s, %s, %s, %s)
                    """,
                    (story_id, image_data, keywords, story, language, student_name, school_name, class_name)
                )
                
                # Count total stories
                cur.execute("SELECT COUNT(*) FROM stories")
                count_result = cur.fetchone()[0]
                
                # If we have more than 20 stories, delete the oldest ones
                if count_result > 20:
                    # Calculate how many stories need to be deleted
                    stories_to_delete = count_result - 20
                    
                    # Find the IDs of the oldest stories
                    cur.execute(
                        """
                        SELECT id FROM stories
                        ORDER BY created_at
                        LIMIT %s
                        """,
                        (stories_to_delete,)
                    )
                    
                    oldest_stories = cur.fetchall()
                    
                    # Delete each of the oldest stories
                    for story_row in oldest_stories:
                        DBOS.logger.info(f"FIFO limit: Deleting old story with ID: {story_row[0]}")
                        cur.execute("DELETE FROM stories WHERE id = %s", (story_row[0],))
                    
                    DBOS.logger.info(f"FIFO limit: Deleted {stories_to_delete} oldest stories to maintain 20-story limit")
        
    except Exception as e:
        DBOS.logger.error(f"Error saving story: {str(e)}")
        raise
