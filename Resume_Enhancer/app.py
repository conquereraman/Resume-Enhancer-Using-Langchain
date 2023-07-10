import streamlit as st
import pickle
import os
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
# from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate, OpenAI, LLMChain
with st.sidebar:
    st.title("Chat App")
    add_vertical_space(5)
    st.write("Made with langchain")


def predict_profession(query):
    jobs = [
        "software engineer",
        "data analyst",
        "data scientist",
        "product manager",
        "research scientist",
        "IT technician"
    ]

    # Calculate similarity scores for each word in the array
    scores = process.extract(query, jobs, scorer=fuzz.token_sort_ratio)

    # Sort the scores in descending order
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0]


def convert_to_embeddings(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    # st.write(chunks)

    store_name = pdf.name[:-4]

    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            VectorStore = pickle.load(f)
        # st.write('Embeddings Loaded')
    else:
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(VectorStore, f)


def main():
    load_dotenv()
    st.write("Hello")
    # uploading pdf
    pdf = st.file_uploader("Upload your pdf", type='pdf')
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        # st.write(chunks)

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded')
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
            # st.write("Embeddings Completed")

        # Accept user question
        # query = st.text_input("Ask Questions Or Query About Your Document")
        profession = st.text_input("What role you want to apply for?")
        if profession:
            predicted_profession = predict_profession(profession)
            print(predicted_profession)
            if os.path.exists(f"{predicted_profession}.pkl"):
                with open(f"{predicted_profession}.pkl", "rb") as f:
                    VectorStore1 = pickle.load(f)
                # st.write('Embeddings Loaded')
            else:
                embeddings = OpenAIEmbeddings()
                VectorStore1 = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{predicted_profession}.pkl", "wb") as f:
                    pickle.dump(VectorStore1, f)
            query1 = "What are the required skills?"
            docs1 = VectorStore1.similarity_search(query=query1, k=3)
            llm = OpenAI(temperature=0)
            chain1 = load_qa_chain(llm=llm, chain_type="stuff")
            jd = chain1.run(input_documents=docs1, question=query1)
            st.write(jd)
            query = "What skills does the candidate have?"
        # st.write(query)
            docs = VectorStore.similarity_search(query=query, k=3)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)

            st.write(response)
            prompt_template = f"""You are given skills of job description {jd} and a skills of candidate {response} compare both and give critical suggestions to the candidate"""
            llm = OpenAI(temperature=0)
            llm_chain = LLMChain(
                llm=llm,
                prompt=PromptTemplate.from_template(prompt_template)
            )
            input_variables = {
                "jd": jd,
                "response": response
            }

# Apply the input dictionary to the LLMChain
            temp = llm_chain(input_variables)
            st.write(temp['text'])


if __name__ == '__main__':
    main()
