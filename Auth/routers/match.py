# app/routes/users.py
from fastapi import APIRouter,Request, HTTPException, status
from sqlalchemy.orm import Session
from App.database import get_db
from Auth.schemas.match import SellerData
from typing import List
import logging
from Auth.controllers.matcher import Matcher


match_router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)

# Register a new user
@match_router.post("/matches")
async def match_selelr_to_buyers(seller_data: SellerData, request: Request):
    try:
        seller_dict = seller_data.dict()
        result = {}
        
        #Handle field name variables
        seller_dict['Number of employees'] = seller_dict.pop('total_employees')
        seller_dict['Headquarters - Country/Region'] = seller_dict.pop('Headquarters')
        
        logger.info(f"Processing match request for: {seller_dict['company_name']}")
        print(seller_dict)
        
        # Access the matcher from app state
        matcher = request.app.state.matcher
        # Run the matcher
        results = matcher.run(seller_dict)
        print(results)
        result["matches"] = results
        # Process dictionary directly
        rationale = matcher.explain_best_match(seller_dict["company_name"])
        result["rationale"] = rationale
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing match request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))