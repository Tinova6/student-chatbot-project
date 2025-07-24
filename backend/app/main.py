from fastapi import FastAPI, WebSocket, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uuid
from typing import List
import datetime
import re

from .llm_service import get_gemini_response
from .db import create_db_tables, get_db, User, Conversation, TestSchedule, Score
from .scheduler import scheduler, add_scheduled_job, schedule_test_notification
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:3000",  # React app default port
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request/Response Validation ---
class Message(BaseModel):
    user_id: str
    message: str

class TestScheduleRequest(BaseModel):
    user_id: str
    test_subject: str
    schedule_time: datetime.datetime

class ScoreCreate(BaseModel):
    user_id: str
    subject: str
    score_value: float

class ScoreResponse(ScoreCreate):
    id: int
    date_recorded: datetime.datetime

    class Config:
        from_attributes = True # updated from orm_mode = True

class ConversationResponse(BaseModel):
    message: str
    is_user: int
    timestamp: datetime.datetime

    class Config:
        from_attributes = True # updated from orm_mode = True

# --- FastAPI Event Handlers ---
@app.on_event("startup")
async def startup_event():
    print("Creating database tables...")
    create_db_tables()
    if not scheduler.running:
        scheduler.start()
    print("Scheduler started.")

@app.on_event("shutdown")
async def shutdown_event():
    if scheduler.running:
        scheduler.shutdown()
    print("Scheduler shutdown.")

# --- API Endpoints ---

@app.post("/user/{user_id}")
async def create_user(user_id: str, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="User already exists")
    new_user = User(id=user_id)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully", "user_id": new_user.id}

@app.get("/user/{user_id}/history", response_model=List[ConversationResponse])
async def get_conversation_history(user_id: str, db: Session = Depends(get_db)):
    history = db.query(Conversation).filter(Conversation.user_id == user_id).order_by(Conversation.timestamp).all()
    return history

@app.get("/user/{user_id}/scores", response_model=List[ScoreResponse])
async def get_user_scores(user_id: str, db: Session = Depends(get_db)):
    scores = db.query(Score).filter(Score.user_id == user_id).order_by(Score.date_recorded.desc()).all()
    return scores

@app.post("/user/{user_id}/score", response_model=ScoreResponse)
async def add_user_score(user_id: str, score: ScoreCreate, db: Session = Depends(get_db)):
    if user_id != score.user_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User ID in path does not match user ID in request body")
    db_score = Score(**score.dict())
    db.add(db_score)
    db.commit()
    db.refresh(db_score)
    return db_score

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, db: Session = Depends(get_db)):
    await websocket.accept()

    # Create user if not exists
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        new_user = User(id=user_id)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        print(f"Created new user: {user_id}")

    print(f"WebSocket connected for user: {user_id}")

    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received from {user_id}: {data}")

            # Save user message to DB
            user_message = Conversation(user_id=user_id, message=data, is_user=1)
            db.add(user_message)
            db.commit()
            db.refresh(user_message)

            # Check for scheduling command
            test_schedule_match = re.search(
                r"schedule test for (.+?) in (\d+)\s*(minute|hour|day)s?",
                data,
                re.IGNORECASE
            )
            if test_schedule_match:
                test_subject = test_schedule_match.group(1).strip()
                amount = int(test_schedule_match.group(2))
                unit = test_schedule_match.group(3).lower()

                schedule_time = datetime.datetime.now()
                if unit == "minute":
                    schedule_time += datetime.timedelta(minutes=amount)
                elif unit == "hour":
                    schedule_time += datetime.timedelta(hours=amount)
                elif unit == "day":
                    schedule_time += datetime.timedelta(days=amount)

                # Save schedule to DB
                new_schedule = TestSchedule(user_id=user_id, test_subject=test_subject, schedule_time=schedule_time)
                db.add(new_schedule)
                db.commit()
                db.refresh(new_schedule)

                add_scheduled_job(
                    schedule_test_notification,
                    schedule_time,
                    websocket, # Pass websocket to the scheduled job
                    test_subject,
                    user_id
                )
                await websocket.send_text(f"Okay, I've scheduled a {test_subject} test for you on {schedule_time.strftime('%Y-%m-%d %H:%M')}.")
                continue # Skip Gemini response for scheduling commands

            # Get Gemini response
            bot_response = await get_gemini_response(data)
            print(f"Sending to {user_id}: {bot_response}")

            # Save bot message to DB
            bot_message = Conversation(user_id=user_id, message=bot_response, is_user=0)
            db.add(bot_message)
            db.commit()
            db.refresh(bot_message)

            await websocket.send_text(bot_response)

    except Exception as e:
        print(f"WebSocket error for {user_id}: {e}")
    finally:
        await websocket.close()
        print(f"WebSocket disconnected for user: {user_id}")