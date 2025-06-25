from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import traceback

from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage
from langchain.memory import ConversationBufferWindowMemory

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# LangChain setup
system_msg = SystemMessage(content=(
        "You are a friendly and professional assistant for Virtual IP, a company that provides AI, web, and cybersecurity solutions."
        " You are embedded on the landing page of https://www.virtualipus.com to assist visitors."
        " Help users with questions about services, pricing, business solutions, and how to contact the team through the contact form."
        " Keep your answers clear, concise, relevant and professional."
        " If you don’t know something or if a visitor needs more help or want to schedule a meeting, kindly suggest they reach out via the contact form."
        " Do not provide personal opinions or engage in casual conversation other than professional assistance."
))
prompt = ChatPromptTemplate.from_messages([
    system_msg,
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
memory = ConversationBufferWindowMemory(k=3, return_messages=True)
llm = ChatOpenAI(model="gpt-4o", temperature=0.5, api_key=openai_api_key)
chain = prompt | llm | StrOutputParser()

# Flask server
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "Missing 'message' in request body"}), 400

        user_input = data["message"]

        memory.chat_memory.add_user_message(user_input)
        response = chain.invoke({
            "input": user_input,
            "history": memory.load_memory_variables({})["history"]
        })
        memory.chat_memory.add_ai_message(response)
        return jsonify({"response": response}), 200

    except Exception as e:
        print("Error:", traceback.format_exc())  # Log detailed trace on server
        return jsonify({"error": "Something went wrong", "details": str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
