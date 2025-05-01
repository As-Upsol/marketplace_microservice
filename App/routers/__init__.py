from fastapi import APIRouter

from Auth.routers import authentication_router



router = APIRouter(prefix="/api")

router.include_router(router=authentication_router, tags=["Users Authentication"])
