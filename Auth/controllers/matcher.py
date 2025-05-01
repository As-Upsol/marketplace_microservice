import pandas as pd
import numpy as np
import spacy
import yaml
import re
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()  
CHATGROQ_API_KEY = os.getenv("GROQ_API_KEY")  

class Matcher:
    def __init__(self, config_path):
        self.groq_client = None
        if CHATGROQ_API_KEY:
            self.groq_client = ChatGroq(
                api_key=CHATGROQ_API_KEY,
                model_name="llama3-8b-8192"  
            )
        self.config = self._load_config(config_path)
        self.nlp = self._load_spacy_model()
        self.seller_df = None
        self.buyer_df = None
        self.matches = []
        self.decimal_places = self.config['io']['output'].get('decimal_places', 1)
        self.load_data()
        self.preprocess_data()
        
        
        # Load SentenceTransformer model for semantic similarity
        self.st_model = SentenceTransformer("thenlper/gte-base")
        
        # Initialize TF-IDF vectorizer for keyword matching
        self.tfidf_vectorizer = TfidfVectorizer()

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
        text = str(text)
        if self.config['preprocessing']['text_cleaning']['lowercase']:
            text = text.lower()
        if self.config['preprocessing']['text_cleaning']['remove_punctuation']:
            text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    def _combine_fields(self, row, fields):
        combined = []
        for field in fields:
            if field in row and pd.notna(row[field]):
                combined.append(str(row[field]))
        return ' '.join(combined)

    def _split_string_to_list(self, text, delimiter=','):
        if pd.isna(text) or not text:
            return []
        return [item.strip() for item in str(text).split(delimiter) if item.strip()]

    def keyword_similarity(self, text1, text2):
        if not text1 or not text2:
            return 0.0
        try:
            tfidf = self.tfidf_vectorizer.fit_transform([text1, text2])
            return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        except Exception:
            return 0.0

    def semantic_similarity(self, text1, text2):
        if not text1 or not text2:
            return 0.0
        embeddings = self.st_model.encode([text1, text2], convert_to_tensor=True)
        return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

    def categorical_similarity(self, text1, text2):
        items1 = self._split_string_to_list(text1)
        items2 = self._split_string_to_list(text2)
        if not items1 or not items2:
            return 0.0
        set1, set2 = set(items1), set(items2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0.0

    def hierarchical_similarity(self, text1, text2):
        if not text1 or not text2:
            return 0.0
        levels1 = [level.strip() for level in str(text1).split('>')]
        levels2 = [level.strip() for level in str(text2).split('>')]
        common = 0
        for l1, l2 in zip(levels1, levels2):
            if l1 == l2:
                common += 1
            else:
                break
        return common / max(len(levels1), len(levels2)) if max(len(levels1), len(levels2)) > 0 else 0.0

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

    def _calculate_category_score(self, seller_row, buyer_row, category_config, match_result, model_name):
        category_score = 0.0
        category_display_name = category_config['display_name']

        for subcategory in category_config['categories']:
            strategy = subcategory['strategy']
            seller_fields = subcategory['fields']['seller']
            buyer_fields = subcategory['fields']['buyer']

            seller_text = self._combine_fields(seller_row, seller_fields)
            buyer_text = self._combine_fields(buyer_row, buyer_fields)
            subcategory_name = subcategory['name']

            if strategy == 'keyword_match':
                score = self.keyword_similarity(seller_text, buyer_text)
            elif strategy == 'spacy_similarity':
                score = self.semantic_similarity(seller_text, buyer_text)
            elif strategy in ['exact_match', 'categorical']:
                if seller_text and buyer_text:
                    score = fuzz.token_set_ratio(seller_text, buyer_text) / 100.0
                else:
                    score = 0.0
            elif strategy in ['industry_similarity', 'hierarchical']:
                score = self.hierarchical_similarity(seller_text, buyer_text)
            else:
                score = 0.0

            subcategory_key = f"{category_display_name}: {subcategory_name}"
            match_result[subcategory_key] = round(score * 100, self.decimal_places)
            category_score += score * subcategory['weight']

        match_result[category_display_name] = round(category_score * 100, self.decimal_places)
        return category_score * category_config['weight']

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
                total_score = 0.0

                for model_name, model_config in self.config['models'].items():
                    weighted_score = self._calculate_category_score(
                        seller,
                        buyer,
                        model_config,
                        match_result,
                        model_name
                    )
                    total_score += weighted_score

                # Calculate overall match score 
                overall_score = (total_score / sum(m['weight'] for m in self.config['models'].values())) * 100
                match_result['Overall Match Score'] = round(overall_score, self.decimal_places)
                
                self.matches.append(match_result)

    def save_results(self):
        if not self.matches:
            print("No matches found to save.")
            return
            
        result_df = pd.DataFrame(self.matches)
        
        # Filter out empty seller names
        result_df = result_df[result_df['Seller Name'].notna() & 
                            (result_df['Seller Name'] != '')]
        
        # Reorder columns as specified in config
        column_order = self.config['io']['output']['column_order']
        available_columns = [col for col in column_order if col in result_df.columns]
        result_df = result_df[available_columns]
        
        for col in result_df.columns:
            if result_df[col].dtype == 'float64':
                result_df[col] = result_df[col].round(self.decimal_places)
        
        if 'Overall Match Score' in result_df.columns:
            result_df = result_df.sort_values(
                by='Overall Match Score',
                ascending=False,
                inplace=False
            )
        else:
            print("Warning: 'Overall Match Score' column not found - skipping sorting")
        return result_df

    def generate_category_rationale(self, entity_row, other_row, model_key, display_name, score):
        """Wrapper method that calls either LLM or simple rationale generator"""
        return self._generate_llm_rationale(entity_row, other_row, display_name, score)

    def _generate_llm_rationale(self, seller_row, buyer_row, model_key, score):
        """Generate rationale for a match score using LLM or fallback to simple method."""
        # First find the model config that matches this display name
        model_config = None
        for m_name, m_config in self.config['models'].items():
            if m_config['display_name'] == model_key:
                model_config = m_config
                break
        
        if not model_config:
            return self._generate_rationale_for_category_simple(model_key, score)
        
        # Get all relevant fields for this model
        seller_fields = set()
        buyer_fields = set()
        for subcat in model_config['categories']:
            seller_fields.update(subcat['fields']['seller'])
            buyer_fields.update(subcat['fields']['buyer'])
        
        # Prepare the data for the prompt
        seller_data = {}
        for field in seller_fields:
            if field in seller_row and pd.notna(seller_row[field]) and seller_row[field] != "":
                seller_data[field] = seller_row[field]
        
        buyer_data = {}
        for field in buyer_fields:
            if field in buyer_row and pd.notna(buyer_row[field]) and buyer_row[field] != "":
                buyer_data[field] = buyer_row[field]
        
        seller_name = seller_row.get('company_name', 'N/A')
        buyer_name = buyer_row.get('Company Name', 'N/A')
        
        prompt = f"""As an M&A analyst, explain why these companies have a {model_key} compatibility score of {score:.1f}%.

**Seller Company**: {seller_name}
**Relevant Attributes**: {seller_data}

**Buyer Company**: {buyer_name}  
**Relevant Attributes**: {buyer_data}

Provide a concise 1-2 sentence business rationale focusing on the most significant matching attributes. If no meaningful match exists, simply state that."""
        
        if hasattr(self, 'groq_client') and self.groq_client:
            try:
                response = self.groq_client.invoke(prompt)
                return response.content.strip()
            except Exception as e:
                print(f"Error generating LLM rationale: {e}")
                return self._generate_rationale_for_category_simple(model_key, score)
        
        # Fallback to simple rationale
        return self._generate_rationale_for_category_simple(model_key, score)

    def _generate_rationale_for_category_simple(self, category_name, score):
        """Fallback when LLM isn't available"""
        if score > 80:
            return f"Exceptional alignment in {category_name.lower()} based on key business attributes."
        elif score > 60:
            return f"Strong compatibility in {category_name.lower()} with multiple matching characteristics."
        elif score > 40:
            return f"Moderate alignment in {category_name.lower()} with some shared attributes."
        elif score > 20:
            return f"Limited compatibility in {category_name.lower()} with few matching aspects."
        else:
            return f"Minimal alignment in {category_name.lower()} with no significant matches."

    def explain_best_match(self, company_name):
        """
        Finds the best match for a given company name (seller or buyer) and generates category rationales.
        Returns a formatted string.
        """
        name = company_name.strip().lower()
        output_lines = []

        # Try to find as a seller
        seller_matches = [m for m in self.matches if m['Seller Name'].strip().lower() == name]
        if seller_matches:
            matches = seller_matches
            entity_df = self.seller_df
            entity_col = 'company_name'
            other_df = self.buyer_df
            other_col = 'Company Name'
            entity_col_match = 'Seller Name'
            other_col_match = 'Buyer Name'
        else:
            #find as a buyer
            buyer_matches = [m for m in self.matches if m['Buyer Name'].strip().lower() == name]
            if buyer_matches:
                matches = buyer_matches
                entity_df = self.buyer_df
                entity_col = 'Company Name'
                other_df = self.seller_df
                other_col = 'company_name'
                entity_col_match = 'Buyer Name'
                other_col_match = 'Seller Name'
            else:
                return "No matches found for this company name."

        best = max(matches, key=lambda x: x['Overall Match Score'])

        entity_row = entity_df[entity_df[entity_col].str.strip().str.lower() == best[entity_col_match].strip().lower()]
        other_row = other_df[other_df[other_col].str.strip().str.lower() == best[other_col_match].strip().lower()]
        if entity_row.empty or other_row.empty:
            return "The buyer or seller provided is not in the database."
        entity_row = entity_row.iloc[0]
        other_row = other_row.iloc[0]

        output_lines.append([f"Best Match for {company_name}: {best[other_col_match]}",
                       f"Overall Match Score: {int(round(best['Overall Match Score']))}%"])

        # Generate rationales for each category
        for model_key, model_config in self.config['models'].items():
            display_name = model_config['display_name']
            score = best.get(display_name, None)
            if score is not None:
                rationale = self.generate_category_rationale(
                    entity_row,
                    other_row,
                    display_name,
                    display_name,
                    score
                )
                output_lines.append([f"{display_name} ({int(round(score))}%):\n{rationale}"])

        return output_lines

    def run(self, standalone_seller: dict= None):
        print("Starting matching process...")
        if standalone_seller:
            seller_df = pd.DataFrame([standalone_seller])
        
        self.preprocess_data(df=seller_df)
        self.calculate_matches()
        results = self.save_results()
        return results.to_dict(orient='records')
        
"""
# Example usage
if __name__ == "__main__":
    matcher = Matcher('config.yaml')
    standalone_seller = {"company_name": 'Buleknight', 
'industry_keywords': '', 'company_description': 'Artificial Intelligence Company', 'regulatory_bodies': 'NEPA, PHCN', 'accreditations': '', 'industry_associations': '', 'recent_awards': '', 'main_competitors': 'Polaris;Google', 'growth_plan': '', 'value_chain': '', 'products_and_services': 'Geospatial Intelligence, Artificial Intelligence, Machine learning', 'unique_selling_points': '', 'revenue_model': '', 'business_model_type': '', 'revenue_by_product_type': 'Drone 50, Analytics 50', 'customer_industries': 'commercial-products;commercial-services;apparel-accessories;consumer-durables;consumer-non-durables;energy-equipment;exploration-production-refining;energy-services;healthcare-devices-supplies;healthcare-services;construction-non-wood;chemicals-gases', 'target_customers': 'Geospatial Developers and planners', 'revenue_by_customer_type': 'Blacks 50, Whites 50', 'Headquarters': 'United Kingdom', 'revenue_by_geography': 'Nigeria 50, United Kingdom 50', 'Revenue': '23455', 'total_employees': '15'}
    matcher.run(standalone_seller)
    
    print("\nRationale:")
    print(matcher.explain_best_match(standalone_seller['company_name']))
    
    
    """