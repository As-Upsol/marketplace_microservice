from pydantic import BaseModel
from typing import Optional

"""class SellerData(BaseModel):
    company_name: str
    company_website: Optional[str] = None
    company_description: Optional[str] = None
    products_and_services: Optional[str] = None
    revenue_model: Optional[str] = None
    unique_selling_points: Optional[str] = None
    industry_keywords: Optional[str] = None
    value_chain: Optional[str] = None
    target_customers: Optional[str] = None
    customer_industries: Optional[str] = None
    main_competitors: Optional[str] = None
    growth_plan: Optional[str] = None
    revenue_by_geography: Optional[str] = None
    revenue_by_product_type: Optional[str] = None
    Number_of_employees: Optional[str] = None
    Revenue: Optional[str] = None
    EBITDA: Optional[str] = None"""
    
class SellerData(BaseModel):
    # Required fields
    company_name: Optional[str] = ""
    company_description: Optional[str] = ""
    products_and_services: Optional[str] = ""
    customer_industries: Optional[str] = ""
    Headquarters: Optional[str] = ""
    Revenue: Optional[str] = ""
    total_employees: Optional[str] = ""
    
    # Optional fields
    industry_keywords: Optional[str] = ''
    regulatory_bodies: Optional[str] = ''
    accreditations: Optional[str] = ''
    industry_associations: Optional[str] = ''
    recent_awards: Optional[str] = ''
    main_competitors: Optional[str] = ''
    growth_plan: Optional[str] = ''
    value_chain: Optional[str] = ''
    unique_selling_points: Optional[str] = ''
    revenue_model: Optional[str] = ''
    business_model_type: Optional[str] = ''
    revenue_by_product_type: Optional[str] = ''
    target_customers: Optional[str] = ''
    revenue_by_customer_type: Optional[str] = ''
    revenue_by_geography: Optional[str] = ''
    
    
class MatchResult(BaseModel):
    Buyer_Name: str
    Overall_Match_Score: float
    Sector_Match: Optional[float] = None
    Offering_Match: Optional[float] = None
    Customer_Match: Optional[float] = None
    Geography_Match: Optional[float] = None
    Size_Match: Optional[float] = None
    Market_Position_Match: Optional[float] = None