# app/routes/users.py
from fastapi import APIRouter,Request, HTTPException, status
from sqlalchemy.orm import Session
from App.database import get_db
from Auth.schemas.match import SellerData, MatchResult
from typing import List
import logging
from Auth.controllers.matcher import Matcher


match_router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)

# Register a new user
@match_router.post("/matches", response_model= List[MatchResult])
async def match_selelr_to_buyers(seller_data: SellerData, request: Request):
    try:
        seller_dict = seller_data.dict()
        
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
        # Process dictionary directly
        formatted_results = []
        for result in results:
            formatted_results.append(MatchResult(
                Buyer_Name=result['Buyer Name'],
                Overall_Match_Score=result['Overall Match Score'],
                Sector_Match=result.get('Sector Match', 0),
                Offering_Match=result.get('Offering Match', 0),
                Customer_Match=result.get('Customer Match', 0),
                Geography_Match=result.get('Geography Match', 0),
                Size_Match=result.get('Size Match', 0),
                Market_Position_Match=result.get('Market Position Match', 0)
            ))
        
        return formatted_results
    
    except Exception as e:
        logger.error(f"Error processing match request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))