from app.models.model import StringResponse 
import spacy
import numpy as np
import pytextrank
import nltk
import yake
from fastapi import HTTPException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer,util
from textblob import TextBlob
from rake_nltk import Rake
from rake_nltk import Rake
from keybert import KeyBERT
from transformers import pipeline

model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model='all-MiniLM-L6-v2') 
nlp = spacy.load("en_core_web_sm")
nlp_lg = spacy.load("en_core_web_lg")
nlp_lg.add_pipe("textrank")


class NlpService:

    @staticmethod
    async def get_even_numbers(start,end):
        if start > end:
            raise HTTPException(status_code=400, detail="Start must be less than or equal to end")  
        evens = [i for i in range(start, end + 1) if i % 2 == 0]
        return {"even_numbers": evens}
    
    @staticmethod
    async def auto_response(query):
        responses = [
            "How can I help you today?",
            "What services are you looking for?",
            "We offer web development, AI solutions, and more.",
            "Please provide more details about your issue.",
            "Our working hours are 9 AM to 6 PM, Monday to Friday.",
            "Thank you for reaching out! We'll get back to you soon.",
            "hi welcome to chat bot ",
            " iam good what about you !",
            " all good ",
            " iam exhausted ",
            "This issue can be solved by restarting the system.",
            "You can contact our support team via email.",
            "We also specialize in mobile app development.",
            "Your request has been received and is being processed.",
            "For billing issues, please visit our support portal.",
            "We appreciate your patience.",
            "Please restart the application and try again.",
            "Do you want to schedule a meeting with our team?",
            "java Developer",
            "java Full stack",
            "is java fast growing language"

        ]
        response_embeddings = model.encode(responses)
        query_embedding = model.encode([query])
        similarities = cosine_similarity(query_embedding, response_embeddings)[0]
        top_indices = np.argsort(similarities)[-3:][::-1]
        top_responses = [responses[i] for i in top_indices]
        return {"words":top_responses}
    
    @staticmethod
    async def similarity_score(text1:str,text2:str):
        if (not text1.strip() or not text2.strip()):
            raise HTTPException(status_code=400, detail="request should not be empty") 
        embeddings = model.encode([text1, text2], convert_to_tensor=True)
        sim_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        return {"similarity_score": round(sim_score,2)}
    
    # @staticmethod
    # async def behaviour_of_sentence(sentence:str):
    #     if (not sentence):
    #         raise HTTPException(status_code=400,detail="request should not be empty")
    #     bolb = TextBlob(sentence)
    #     polarity = bolb.sentiment.polarity
    #     if polarity > 0:
    #         return {"string":"positive"}
    #     elif polarity < 0 :
    #         return {"string" : "negative"}
    #     else:
    #         return {"string":"netural"}

    @staticmethod
    async def behaviour_of_sentence(sentence:str):
        if (not sentence):
            raise HTTPException(status_code=400,detail="request should not be empty")
        classifier = pipeline("sentiment-analysis")
        result = classifier(sentence)
        behaviour = result[0]["label"]
        return{"string":behaviour}
        
    @staticmethod
    async def extract_entities(sentence:str):
        if (not sentence):
            raise HTTPException(status_code=400,detail="request should not be empty")
        doc = nlp(sentence)
        entities = {ent.text: ent.label_ for ent in doc.ents}
        return {"entities": entities}
        
    @staticmethod
    async def text_summarize(sentence:str):
        if (not sentence):
            raise HTTPException(status_code=400,detail="request should not be empty")
        doc = nlp_lg(sentence)
        for summary in doc._.textrank.summary(limit_phrases=2, limit_sentences=2):
            return StringResponse(string=str(summary))
        
        
    # @staticmethod
    # async def extract_keywords(sentence:str):
    #     if (not sentence):
    #         raise HTTPException(status_code=400,detail="request should not be empty")
    #     r = yake.KeywordExtractor(lan="en", n=3,top=5)
    #     key_words = r.extract_keywords(sentence)
    #     words = [kw for kw, score in key_words
    #     return {"words":words}
    
    # @staticmethod
    # async def extract_keywords(sentence:str):
    #     if (not sentence):
    #         raise HTTPException(status_code=400,detail="request should not be empty")
    #     r = Rake()
    #     r.extract_keywords_from_text(sentence)
    #     keywords = r.get_ranked_phrases()
    #     return {"words":keywords}
    
    @staticmethod
    async def extract_keywords(sentence:str):
        if (not sentence):
            raise HTTPException(status_code=400,detail="request should not be empty")
        words = kw_model.extract_keywords(sentence,keyphrase_ngram_range=(2,3),stop_words='english',top_n=5)
        keywords = [kw for kw,score in words]
        return {"words":keywords}

