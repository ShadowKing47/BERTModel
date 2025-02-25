from langchain_community.llms import Ollama
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Union
import torch
import numpy as np
import streamlit as st


llm = Ollama(model="samantha-mistral")

class RAGNMistral:
    def __init__(self, corpus: List[str]):
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.corpus = corpus
        self.doc_embeddings = self.embed_doc(corpus)
        self.llm = llm 

    def embed_doc(self, documents: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(documents, str):
            documents = [documents]

        embeddings = []
        for doc in documents:
            inputs = self.tokenizer(doc, return_tensors='pt', padding=True, 
                                  max_length=512, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
        return torch.tensor(np.array(embeddings))

    def find_relevant_documents(self, query: str, k: int = 3) -> List[tuple]:
        query_emb = self.embed_doc(query)
        similarity = cosine_similarity(query_emb, self.doc_embeddings)[0]
        top_k = similarity.argsort()[-k:][::-1]
        return [(idx, similarity[idx], self.corpus[idx]) for idx in top_k]

    def generate_with_mistral(self, query: str, context: str) -> str:
        prompt = f"Based on this context: {context}, Answer the given question: {query}"
        res = self.llm.invoke(prompt)  
        return res

    def final(self, query: str, k: int = 3) -> str:
        relevant_docs = self.find_relevant_documents(query, k)
        context = " ".join([doc for _, _, doc in relevant_docs])
        ans = self.generate_with_mistral(query, context)
        return ans

sample_corpus = [
    """Artificial Intelligence (AI) is the simulation of human intelligence by machines. 
    It includes machine learning, natural language processing, and robotics. AI systems 
    can learn from experience, adjust to new inputs, and perform human-like tasks.""",
    
    """Machine Learning is a subset of AI that provides systems the ability to automatically 
    learn and improve from experience without being explicitly programmed. It focuses on 
    the development of computer programs that can access data and use it to learn for themselves.""",
    
    """Natural Language Processing (NLP) is a branch of AI that helps computers understand, 
    interpret, and manipulate human language. NLP combines computational linguistics with 
    statistical, machine learning, and deep learning models.""",
    
    """Deep Learning is part of machine learning based on artificial neural networks. 
    It uses multiple layers to progressively extract higher-level features from raw input. 
    For example, in image processing, lower layers identify edges, while higher layers 
    identify concepts."""
]

# Streamlit App
def main():
    st.title("RAG with Mistral and Streamlit")
    st.write("Ask a question about AI, Machine Learning, NLP, or Deep Learning!")
    rag = RAGNMistral(sample_corpus)
    query = st.text_input("Enter your question:")
    if query:        
        response = rag.final(query)    
        st.write("### Answer:")
        st.write(response)

if __name__ == "__main__":
    main()