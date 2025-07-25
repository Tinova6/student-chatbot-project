from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .config import DATABASE_URL
import datetime

# Create a SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Declare a base for declarative models
Base = declarative_base()

# Define the User model
class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, index=True) # UUID as string
    created_at = Column(DateTime, default=datetime.datetime.now)

# Define the Conversation model
class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    message = Column(Text)
    is_user = Column(Integer) # 1 for user, 0 for bot
    timestamp = Column(DateTime, default=datetime.datetime.now)

# Define the TestSchedule model
class TestSchedule(Base):
    __tablename__ = "test_schedules"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    test_subject = Column(String)
    schedule_time = Column(DateTime)
    is_completed = Column(Integer, default=0) # 0 for not completed, 1 for completed
    created_at = Column(DateTime, default=datetime.datetime.now)

# Define the Score model
class Score(Base):
    __tablename__ = "scores"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    subject = Column(String)
    score_value = Column(Float)
    date_recorded = Column(DateTime, default=datetime.datetime.now)


# Function to create tables
def create_db_tables():
    Base.metadata.create_all(bind=engine)

# Function to get a database session
def get_db():
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()