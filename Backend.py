from flask import Flask, request, jsonify
from flask_cors import CORS
import json 
import evadb
from Functions import vector_retrieval, summary_retrieval, response_aggregator , free_llm_call ,load_wiki_pages, generate_vector_stores
from Subquestion import generate_subquestions
import os


app = Flask(__name__)
CORS(app)

cursor = evadb.connect().cursor()


# Loading the wiki docs: 
doc_names = ["Toronto", "Chicago", "Houston", "Boston", "Atlanta"]
wiki_docs = load_wiki_pages(page_titles=doc_names)
generate_vector_stores(cursor, doc_names)


@app.route("/ask", methods=["POST"])
def ask_question(): 
    try: 
        data = request.json
        question = data.get("question" , '')

        if not question : 
            return jsonify({"error": "No question provided"})
        subquestions_bundle_list = generate_subquestions(
            question=question,
            file_names=doc_names,
            user_task="We have a database of Wikipedia articles about cities.",
            llm_model="gemini-1.5-flash",
        )

        responses = []
        for item in subquestions_bundle_list:
            subquestion = item.question
            selected_func = item.function.value
            selected_doc = item.file_name.value

            if selected_func == "vector_retrieval":
                response = vector_retrieval(cursor, "gemini-1.5-flash", subquestion, selected_doc)
            elif selected_func == "llm_retrieval":
                response = summary_retrieval("gemini-1.5-flash", subquestion, wiki_docs[selected_doc])
            else:
                response = "I don't know."

            responses.append(response)

        final_response = response_aggregator("gemini-1.5-flash", question, responses)

        return jsonify({"answer": final_response})

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False)