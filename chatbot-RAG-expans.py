import os
import uuid

import openai
import requests
import chromadb
import csv
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI


from chromadb.utils import embedding_functions

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# OpenAI client used for chat completion
openai_client = OpenAI(api_key=openai.api_key)

# Chroma client used for vector retrieval
chroma_client = chromadb.PersistentClient(path="./chroma_db")

#2. create embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai.api_key,
    model_name="text-embedding-3-small"
)

#3. check if collection exists
existing = [c.name for c in chroma_client.list_collections()]
if "csv_documents" in existing:
    collection = chroma_client.get_collection("csv_documents", embedding_function=openai_ef)
else:
    collection = chroma_client.get_or_create_collection(
        name="csv_documents",
        embedding_function=openai_ef
    )
    #4. read the csv file and add to collection
    with open('data/rag_sample_qas_from_kis.csv', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        ids = []
        documents = []
        metadatas = []

        for row in reader:
            text_parts = []
            for key, value in row.items():
                if value is not None and str(value).strip() != "":
                    text_parts.append(f"{key}: {value}")

            text = " | ".join(text_parts)

            ids.append(str(uuid.uuid4()))
            documents.append(text)
            metadatas.append(row)

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

#5. define a function to augment the query with retrieved documents.
def augment_query_generated(query):

    prompt = """You are a helpful expert assistant. Provide an example answer to the given question, that might be found generally.If you don't know the answer, say "I don't knowdo not make up an answer."""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Question: {query}"}
    ]

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0
    )

    return response.choices[0].message.content


def get_recent_history(history, limit=5):
    return history[-limit:]


def chat_with_rag(user_query, history=None):
    if history is None:
        history = []

    # Step 1: call the augmentation function
    augmented_answer = augment_query_generated(user_query)

    # Step 2: combine current query + augmentation output
    combined_input = (
        f"User question: {user_query}\n\n"
        f"Augmented answer: {augmented_answer}"
    )

    # Step 3: query Chroma for relevant documents
    retrieval_results = collection.query(
        query_texts=[combined_input],
        n_results=5
    )
    relevant_documents = retrieval_results["documents"][0]
    context_documents = "\n\n".join(relevant_documents)

    # Step 4: maintain recent chat history
    recent_history = get_recent_history(history, limit=5)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the recent conversation and the retrieved documents to answer the user. If the context is not enough, say you do not know and do not guess."
        }
    ]

    for item in recent_history:
        messages.append({"role": item["role"], "content": item["content"]})

    messages.append({
        "role": "user",
        "content": (
            f"Context Documents:\n{context_documents}\n\n"
            f"Augmented answer:\n{augmented_answer}\n\n"
            f"User question:\n{user_query}"
        )
    })

    # Step 5: final answer from OpenAI
    final_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0
    )

    return final_response.choices[0].message.content


def chat_bot():
    history = []
    print("Chatbot ready. Type 'exit' to quit.")

    while True:
        user_query = input("You: ").strip()
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = chat_with_rag(user_query, history)
        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": response})
        print(f"Bot: {response}")


def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
    st.title("RAG Chatbot")

    if "history" not in st.session_state:
        st.session_state.history = []

    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question...")

    if prompt:
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Searching the knowledge base and generating a response..."):
            response = chat_with_rag(prompt, st.session_state.history)

        st.session_state.history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


if __name__ == "__main__":
    main()

