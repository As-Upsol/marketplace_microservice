from fastapi import APIRouter
from .match import match_router




authentication_router = APIRouter()


authentication_router.include_router(match_router)