import os

from dotenv import load_dotenv

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

load_dotenv()


POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_SERVER = os.getenv("POSTGRES_SERVER", 'localhost')
POSTGRES_DB = os.getenv("POSTGRES_DB")

RENDER_POSTGRES_USER: str= os.getenv("RENDER_POSTGRES_USER")
RENDER_POSTGRES_PASSWORD: str= os.getenv("RENDER_PASSWORD")
RENDER_POSTGRES_SERVER: str= os.getenv("RENDER_SERVER")
RENDER_DB: str=os.getenv("RENDER_DB")




#DATABASE_URL = os.getenv('DATABASE_URL_LOCAL')
#DATABASE_URL = os.getenv('DATABASE_URL_RENDER')
#DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_SERVER}/{POSTGRES_DB}"
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_SERVER}:5432/{POSTGRES_DB}"
#DATABASE_URL= f"postgresql://{RENDER_POSTGRES_USER}:{RENDER_POSTGRES_PASSWORD}@{RENDER_POSTGRES_SERVER}/{RENDER_DB}"
engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit = False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        

        
        
