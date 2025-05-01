import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import App.database as database
from Auth.controllers.matcher import Matcher
from App import routers

load_dotenv()

database.Base.metadata.create_all(bind=database.engine)

app = FastAPI()

# Allow requests from 

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv('FRONTEND_URL'), os.getenv('LOCALHOST'), '*'],
    allow_credentials=True,
    allow_methods=["POST", "PATCH", "DELETE", "GET", "PUT", "OPTIONS"],
    allow_headers=["*"], # This allows all headers
    )


Matcher()
print(".............................................................")

app.include_router(routers.router)


@app.get('/')
async def hello():
    return {"greeting": "Welcome to City Watch"}
