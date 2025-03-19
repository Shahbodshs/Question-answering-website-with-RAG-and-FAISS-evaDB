import os
import re
import json
from enum import Enum
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from typing import List
from pydantic import Field, create_model
import google.generativeai as genai
from dotenv import load_dotenv
from Functions import free_llm_call


warnings.filterwarnings("ignore")

# DEFAULT_SUBQUESTION_GENERATOR_PROMPT = """
#                  You are an AI agent that takes a complex user question and returns a list of simple subquestions to answer the user's question.
#                  You are provided a set of functions and data sources that you can use to answer each subquestion.
#                  If the user question is simple, just return the user question, the function, and the data source to use.
#                  You can only use the provided functions and data sources.
#                  The subquestions should be complete questions that can be answered by a single function and a single data source.
#                  """

# DEFAULT_SUBQUESTION_GENERATOR_PROMPT = """
#     You are an AI assistant that specializes in breaking down complex questions into simpler, manageable sub-questions.
#     When presented with a complex user question, your role is to generate a list of sub-questions that, when answered, will comprehensively address the original query.
#     You have at your disposal a pre-defined set of functions and data sources to utilize in answering each sub-question.
#     If a user question is straightforward, your task is to return the original question, identifying the appropriate function and data source to use for its solution.
#     Please remember that you are limited to the provided functions and data sources, and that each sub-question should be a full question that can be answered using a single function and a single data source.
# """

DEFAULT_SUBQUESTION_GENERATOR_PROMPT = """
    You are an AI assistant that specializes in breaking down complex questions into simpler, manageable sub-questions.
    You have at your disposal a pre-defined set of functions and files to utilize in answering each sub-question.
    Please remember that your output should only contain the provided function names and file names, and that each sub-question should be a full question that can be answered using a single function and a single file.
    Return only a pure JSON object without any additional formatting, markdown, or code block markers. Do not include triple backticks, language hints, or any extra text.
"""

DEFAULT_USER_TASK = ""


class FunctionEnum(str, Enum):
    """The function to use to answer the questions."""
    VECTOR_RETRIEVAL = "vector_retrieval"
    LLM_RETRIEVAL = "llm_retrieval"

def generate_subquestions(
    question,
    file_names: List[str] = None,
    system_prompt=DEFAULT_SUBQUESTION_GENERATOR_PROMPT,
    user_task=DEFAULT_USER_TASK,
    llm_model="gemini-1.5-flash",
):
    """
    Generates a list of subquestions from a user question along with the file name and the function
    to use to answer the question using Google's Gemini API via free_llm_call.
    """
    try:
        FilenameEnum = Enum("FilenameEnum", {x.upper(): x for x in file_names})

        QuestionBundle = create_model(
            "QuestionBundle",
            question=(str, Field(None, description="The subquestion extracted from the user's question")),
            function=(FunctionEnum, Field(None)),
            file_name=(FilenameEnum, Field(None)),
        )

        SubQuestionBundleList = create_model(
            "SubQuestionBundleList",
            subquestion_bundle_list=(
                List[QuestionBundle],
                Field(None, description="A list of subquestions - each item in the list contains a question, a function, and a file name")
            )
        )

        user_prompt = f"{user_task}\nHere is the user question: {question}\n"

        few_shot_examples = [
            {
                "role": "user",
                "content": "Compare the population of Atlanta and Toronto?",
            },
            {
                "role": "function",
                "name": "SubQuestionBundleList",
                "content": """
                {
                    "subquestion_bundle_list": [
                        {
                            "question": "What is the population of Atlanta?",
                            "function": "vector_retrieval",
                            "file_name": "Atlanta"
                        },
                        {
                            "question": "What is the population of Toronto?",
                            "function": "vector_retrieval",
                            "file_name": "Toronto"
                        }
                    ]
                }"""
            },
            {
                "role": "user",
                "content": "Summarize the history of Chicago and Houston.",
            },
            {
                "role": "function",
                "name": "SubQuestionBundleList",
                "content": """
                {
                    "subquestion_bundle_list": [
                        {
                            "question": "What is the history of Chicago?",
                            "function": "llm_retrieval",
                            "file_name": "Chicago"
                        },
                        {
                            "question": "What is the history of Houston?",
                            "function": "llm_retrieval",
                            "file_name": "Houston"
                        }
                    ]
                }"""
            }
        ]
        user_prompt += "\nFew-shot examples:\n" + json.dumps(few_shot_examples, indent=2)
        user_prompt += "\nPlease output your answer as valid JSON. Ensure that the 'file_name' field is one of the following exactly: 'Toronto', 'Chicago', 'Houston', 'Boston', 'Atlanta'."

        response = free_llm_call(
            model=llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        try:
            subquestions_dict = json.loads(response)
        except json.JSONDecodeError:
            return []

        subquestions_list = subquestions_dict["subquestion_bundle_list"]
        subquestions_pydantic_obj = SubQuestionBundleList(subquestion_bundle_list=subquestions_list)

        return subquestions_pydantic_obj.subquestion_bundle_list

    except Exception:
        return []