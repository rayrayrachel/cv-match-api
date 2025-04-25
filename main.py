import spacy
from fastapi import FastAPI
from pydantic import BaseModel

# Load Spacy model
nlp = spacy.load("en_core_web_sm")

# FastAPI API connection
app = FastAPI()
class MatchRequest(BaseModel):
    cv_text: str
    job_description: str

# Keyword extraction function
def extract_keywords(text: str) -> (set, set):
    doc = nlp(text)
    
    important_keywords = set()
    less_important_keywords = set()

    for ent in doc.ents:
        important_keywords.add(ent.text.lower()) 

    for chunk in doc.noun_chunks:
        if not chunk.root.is_stop:  
            less_important_keywords.add(chunk.text.lower())  

    for token in doc:
        if token.pos_ in {"PROPN, NOUN"} and not token.is_stop and token.is_alpha:
            important_keywords.add(token.text.lower())  

        less_important_keywords -= important_keywords
    return important_keywords, less_important_keywords

# Return CV match score
def calculate_keyword_match(cv_text: str, job_description: str) -> dict:
    cv_important, cv_less_important = extract_keywords(cv_text)
    jd_important, jd_less_important = extract_keywords(job_description)
    
    matched_important = cv_important.intersection(jd_important)
    
    # Only calculate score based on matching important keywords
    total_important_keywords = len(jd_important)
    total_matched = len(matched_important)

    if total_important_keywords > 0:
        match_score = (total_matched / total_important_keywords) * 100
    else:
        match_score = 0
    
    return {
        "score": round(match_score, 2),
        "matched_important_keywords": list(matched_important),
        "matched_less_important_keywords": list(cv_less_important.intersection(jd_less_important)),
        "missed_important_keywords": list(jd_important - cv_important),
        "missed_less_important_keywords": list(jd_less_important - cv_less_important),
    }

@app.post("/cv-match")
def match_cv(req: MatchRequest):
    return calculate_keyword_match(req.cv_text, req.job_description)
