import os
import re
import json
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from dotenv import load_dotenv
from pathlib import Path
import requests
from typing import List
import faiss
import evadb

warnings.filterwarnings("ignore")

# Set up API key for Google Gemini
import google.generativeai as genai

API_KEY = "Your api key"
genai.configure(api_key=API_KEY)


def free_llm_call(model, user_prompt, system_prompt=None):
    """Calls Google's Gemini API to generate text."""
    full_prompt = f"{system_prompt}\n{user_prompt}" if system_prompt else user_prompt

    try:
        gemini_model = genai.GenerativeModel(model)
        response = gemini_model.generate_content(full_prompt)
        return response.text

    except Exception:
        return "Error: Could not generate response."


def generate_vector_stores(cursor, docs):
    """Creates vector stores for documents in EvaDB."""
    for doc in docs:
        try:
            cursor.query(f"DROP TABLE IF EXISTS {doc};").df()
            cursor.query(f"LOAD DOCUMENT 'data/{doc}.txt' INTO {doc};").df()
            evadb_path = os.path.dirname(evadb.__file__)

            cursor.query(
                f"""CREATE FUNCTION IF NOT EXISTS SentenceFeatureExtractor
                IMPL  '{evadb_path}/functions/sentence_feature_extractor.py';
                """
            ).df()

            cursor.query(
                f"""CREATE TABLE IF NOT EXISTS {doc}_features AS
                SELECT SentenceFeatureExtractor(data), data FROM {doc};"""
            ).df()

            cursor.query(
                f"CREATE INDEX IF NOT EXISTS {doc}_index ON {doc}_features (features) USING FAISS;"
            ).df()

        except Exception:
            pass


def vector_retrieval(cursor, llm_model, question, doc_name):
    """Returns the answer to a factoid question using vector retrieval. Falls back to Gemini if needed."""
    try:
        res_batch = cursor.query(
            f"""SELECT data FROM {doc_name}_features
            ORDER BY Similarity(SentenceFeatureExtractor('{question}'),features)
            LIMIT 3;"""
        ).df()

        if res_batch.empty:
            return None  # No relevant results, fall back to Gemini

        context_list = [res_batch["data"][i] for i in range(len(res_batch))]
        context = "\n".join(context_list)

        user_prompt = f"""You are an assistant for question-answering tasks.
                    Use the following pieces of retrieved context to answer the question.
                    If you don't know the answer, just say that you don't know.
                    Use three sentences maximum and keep the answer concise.
                    Question: {question}
                    Context: {context}
                    Answer:"""

        return free_llm_call(model=llm_model, user_prompt=user_prompt)

    except Exception:
        return None  # Ensure safe fallback to Gemini



def summary_retrieval(llm_model, question, doc):
    """Returns the answer to a summarization question over the document using summary retrieval."""
    try:
        user_prompt = f"""Here is some context: {doc}
        Use only the provided context to answer the question.
        Here is the question: {question}"""

        return free_llm_call(model=llm_model, user_prompt=user_prompt)

    except Exception:
        return "Error: Could not generate summary."


def response_aggregator(llm_model, question, responses):
    """Aggregates the responses from the subquestions to generate the final response."""
    try:
        system_prompt = """You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise."""

        context = "\n".join(responses)
        user_prompt = f"Question: {question}\nContext: {context}\nAnswer:"
        return free_llm_call(model=llm_model, system_prompt=system_prompt, user_prompt=user_prompt)

    except Exception:
        return "Error: Could not aggregate responses."


def load_wiki_pages(page_titles=["Toronto", "Chicago", "Houston", "Boston", "Atlanta"]):
    """Loads Wikipedia pages and saves them as text files."""
    for title in page_titles:
        try:
            response = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "format": "json",
                    "titles": title,
                    "prop": "extracts",
                    "explaintext": True,
                },
            ).json()

            page = next(iter(response["query"]["pages"].values()))
            wiki_text = page["extract"]
            data_path = Path("data")

            if not data_path.exists():
                Path.mkdir(data_path)

            with open(data_path / f"{title}.txt", "w", encoding="utf-8") as fp:
                fp.write(wiki_text)

        except Exception:
            pass

    city_docs = {}
    for wiki_title in page_titles:
        try:
            input_text = open(f"data/{wiki_title}.txt", "r", encoding="utf-8").read()
            city_docs[wiki_title] = input_text[:10000]
        except Exception:
            pass

    return city_docs
