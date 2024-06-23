from flask import Flask, request, jsonify
import asyncio
from langchain_community.vectorstores import ScaNN
from langchain.chains import RetrievalQA
from langchain_community.embeddings import GPT4AllEmbeddings
from llamaapi import LlamaAPI
from langchain_experimental.llms import ChatLlamaAPI

app = Flask(__name__)

# Model initialization
model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
gpt4all_kwargs = {'allow_download': 'True'}
embeddings = GPT4AllEmbeddings(
    model_name=model_name,
    gpt4all_kwargs=gpt4all_kwargs
)
llama = LlamaAPI("LL-K20AV0bKjaEuK2Y4B6JDCuPG83hbYRkUOxe0wCPrKomiXlRQZYJgpV98bYLpA9yH")
llm = ChatLlamaAPI(client=llama)
db = ScaNN.load_local("RMKScann", embeddings, allow_dangerous_deserialization=True)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(k=5))

@app.route('/invoke', methods=['POST'])
async def invoke():
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Run the model asynchronously
    result = await asyncio.to_thread(qa.invoke, query)
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
