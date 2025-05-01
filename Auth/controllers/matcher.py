import pandas as pd
import numpy as np
import spacy
import yaml
import re
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class Matcher:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.nlp = self._load_spacy_model()
        self.seller_df = None
        self.buyer_df = None
        self.matches = []
        self.decimal_places = 1
        self.base_boost = 1.5 
        self.match_boost = 1.2
        self.load_data()
        self.preprocess_data()

        # Load SentenceTransformer model
        self.st_model = SentenceTransformer("thenlper/gte-base")

  
        self.max_raw_score = 0.0
        for model_config in self.config['models'].values():
            subcat_weight_sum = sum(subcat['weight'] for subcat in model_config['categories'])
            self.max_raw_score += subcat_weight_sum * model_config['weight']

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_spacy_model(self):
        nlp = spacy.load(self.config['preprocessing']['spacy']['model'])
        for component in self.config['preprocessing']['spacy']['disabled_components']:
            nlp.disable_pipe(component)
        return nlp

    def _preprocess_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    def _combine_fields(self, row, fields):
        return ' '.join(str(row[field]) for field in fields if field in row and pd.notna(row[field]))

    def _boosted_similarity(self, text1, text2):
        """Use SentenceTransformer embeddings + cosine similarity + optional fuzzy boost"""
        if not text1 or not text2:
            return 0.0

        # Preprocess texts
        text1 = self._preprocess_text(text1)
        text2 = self._preprocess_text(text2)

        # Compute embeddings
        embeddings = self.st_model.encode([text1, text2])
        sim_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]  # in [0,1]

        # Apply base boost
        boosted_score = sim_score * self.base_boost

        # Optional: fuzzy matching boost (can be removed if you want purely transformer-based)
        fuzzy_score = (fuzz.token_set_ratio(text1, text2) / 100.0) * self.base_boost

        if boosted_score > 0.6:
            boosted_score *= self.match_boost
        if fuzzy_score > 0.6:
            fuzzy_score *= self.match_boost

        return max(boosted_score, fuzzy_score)

    def _industry_similarity(self, value1, value2):
        """Industry similarity using fuzzy matching with boosting (can be enhanced similarly)"""
        if pd.isna(value1) or pd.isna(value2):
            return 0.0
        base_score = fuzz.token_set_ratio(str(value1).lower(), str(value2).lower()) / 100.0
        boosted_score = base_score * self.base_boost * (self.match_boost if base_score > 0.6 else 1.0)
        return boosted_score

    def _range_comparison(self, value1, value2, range_threshold=0.4):
        """Range comparison remains the same"""
        try:
            val1 = float(value1) if not pd.isna(value1) else 0.0
            val2 = float(value2) if not pd.isna(value2) else 0.0

            if val1 == 0.0 or val2 == 0.0:
                return range_threshold

            ratio = min(val1, val2) / max(val1, val2)
            boosted_ratio = ratio * self.base_boost * (self.match_boost if ratio > 0.6 else 1.0)
            return max(boosted_ratio, range_threshold)
        except:
            return range_threshold
        
    def load_data(self):
        csv_options = self.config['io']['input']['csv_options']
        """
        self.seller_df = pd.read_csv(
            self.config['io']['input']['seller_file'],
            encoding=self.config['io']['input']['encoding'],
            **csv_options
        )
        """
        self.buyer_df = pd.read_csv(
            self.config['io']['input']['buyer_file'],
            encoding=self.config['io']['input']['encoding'],
            **csv_options
        )

    def preprocess_data(self, df=None):
        if df is not None:
            self.seller_df = df
            for df in [self.seller_df]:
                for column in df.columns:
                    if df[column].dtype == 'object':
                        df[column] = df[column].apply(self._preprocess_text)
                        
        else:
            for df in [self.buyer_df]:
                for column in df.columns:
                    if df[column].dtype == 'object':
                        df[column] = df[column].apply(self._preprocess_text)
            

    def _calculate_category_score(self, seller_row, buyer_row, category_config, match_result, category_prefix):
        total_score = 0.0
        sub_scores = {}

        for subcategory in category_config['categories']:
            strategy = subcategory['strategy']
            seller_fields = subcategory['fields']['seller']
            buyer_fields = subcategory['fields']['buyer']

            seller_text = self._combine_fields(seller_row, seller_fields)
            buyer_text = self._combine_fields(buyer_row, buyer_fields)

            if strategy == 'keyword_match':
                score = self._boosted_similarity(seller_text, buyer_text)

            elif strategy == 'industry_similarity':
                score = self._industry_similarity(seller_text, buyer_text)

            elif strategy == 'revenue_comparison':
                score = self._range_comparison(
                    seller_row.get('Revenue'),
                    buyer_row.get('Revenue')
                )

            elif strategy == 'employee_comparison':
                score = self._range_comparison(
                    seller_row.get('Number of employees'),
                    buyer_row.get('Employees')
                )

            else:
                continue

            # Do NOT cap here; allow >1.0 for normalization later
            sub_score_name = f"{category_prefix}_{subcategory['name']}_score"
            sub_scores[sub_score_name] = score

            total_score += score * subcategory['weight']

        match_result.update(sub_scores)
        return total_score

    def calculate_matches(self):
        valid_sellers = self.seller_df[self.seller_df['company_name'].notna() &
                       (self.seller_df['company_name'] != '')]

        for _, seller in valid_sellers.iterrows():
            seller_name = seller.get('company_name', '')
            if not seller_name:
                continue

            for _, buyer in self.buyer_df.iterrows():
                buyer_name = buyer.get('Company Name', '')
                if not buyer_name:
                    continue

                match_result = {
                    'Seller Name': seller_name,
                    'Buyer Name': buyer_name
                }
                raw_total_score = 0.0
                raw_sub_scores = {}

                for model_name, model_config in self.config['models'].items():
                    raw_score = self._calculate_category_score(
                        seller,
                        buyer,
                        model_config,
                        match_result,
                        model_name
                    )
                    weighted_raw_score = raw_score * model_config['weight']
                    raw_sub_scores[model_name] = weighted_raw_score
                    raw_total_score += weighted_raw_score

                # Normalize scores to percentage relative to theoretical max raw score
                overall_percentage = (raw_total_score / self.max_raw_score) * 100
                overall_percentage = round(min(overall_percentage, 100), self.decimal_places)
                match_result['Overall Match Score'] = overall_percentage

                # Normalize and store subcategory scores as percentages
                for model_name, weighted_raw_score in raw_sub_scores.items():
                    normalized_sub_score = (weighted_raw_score / self.max_raw_score) * 100
                    normalized_sub_score = round(min(normalized_sub_score, 100), self.decimal_places)
                    display_name = self.config['models'][model_name]['display_name']
                    match_result[display_name] = normalized_sub_score

                self.matches.append(match_result)


    def save_results(self):
        result_df = pd.DataFrame(self.matches)
        
        
        result_df = result_df[result_df['Seller Name'].notna() & 
                            (result_df['Seller Name'] != '')]
        
        sub_score_cols = [col for col in result_df.columns if '_score' in col]
        column_order = self.config['io']['output']['column_order'] + sub_score_cols
        result_df = result_df[column_order]
        
        for col in result_df.columns:
            if result_df[col].dtype == 'float64':
                result_df[col] = result_df[col].round(self.decimal_places)
        
        return result_df

    def run(self, standalone_seller: dict= None):
        if standalone_seller:
            seller_df = pd.DataFrame([standalone_seller])
        self.preprocess_data(df=seller_df)
        self.calculate_matches()
        results = self.save_results()
        return results.to_dict(orient='records')




"""
        
matcher = Matcher('config.yaml')
print("Running Match.....................................................................................................")
standalone_seller = {
    # Basic Info
    "company_name": "Atomus Limited",
    "company_website": "https://atomus.com/",
    "Headquarters - Country/Region": "United Kingdom",
    
    # Business Information
    "company_description": "Atomus are software developers and creators of a market-leading Sales Coaching & Development platform, aCoach. Working within the Global Life Science sector for over 15 years, they have 5 of the big 10 global Pharmaceutical companies as clients.",
    "products_and_services": "aCoach delivers Accelerated Skill Development through enhanced coaching. Organizations with large sales forces invest significant sums in their training programs often without a framework to ensure that the lessons are being taught and learnt in the field.",
    "revenue_model": "B2B SaaS",
    "unique_selling_points": "learning management platform for life sciences",
    "industry_keywords": "learning management system; life sciences technology; sales coaching platform; sales training platform",
    "value_chain": "Software",
    "target_customers": "Enterprise Organizations within the life sciences sector",
    "customer_industries": "Life sciences",
    "main_competitors": "https://www.quantified.ai/; https://www.allego.com/",
    "growth_plan": "Atomus aims to expand its reach within the life sciences sector by continually enhancing its coaching platform",
    
    # Metrics
    "revenue_by_geography": "USA 70%; UK 30%",
    "revenue_by_product_type": "Recurring Revenue from SaaS offering 77%; Non Recurring Revenue from onboarding 23%",
    "Number of employees": "20",
    "Revenue": "3000000",  # $3M
    "EBITDA": "1000000",   # $1M
    
    # Optional fields (included to prevent missing field errors)
    "share_sale_type": "",
    "transition_period": "",
    "reason_for_selling": "",
    "accreditations": "",
    "recent_awards": "",
    "outstanding_litigation": "No",
    "negative_media_coverage": "No"
}
matcher.run(standalone_seller)

"""
