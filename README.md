# RAG_Basic

# 

## 🚀 Funcionalidades

- 📚 Aprendizado personalizado
- 🤖 IA adaptativa
- 📊 Análise de desempenho
- 💡 Recomendações inteligentes

## 🛠️ Tecnologias

- Python
- Jupyter Notebook
- RAG (Retrieval-Augmented Generation)
- Processamento de Linguagem Natural

## 📦Como usar?

# Instale as dependências
pip install -r requirements.txt
```

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

# Importando as bibliotecas necessárias
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
import faiss
import torch
import warnings
warnings.filterwarnings("ignore")

# Base de conhecimento
documents = [
    """Danilo é um Especialista em Machine Learning e Gestão de Agentes de IA, com foco em automação de atendimento, 
    análise preditiva e integração de sistemas. Atuo no desenvolvimento de modelos supervisionados e não supervisionados 
    (classificação, churn, recomendação), usando Python, Scikit-Learn, Pandas e TensorFlow. Além de execução e construção 
    projetos com RAG para criar agentes de IA com respostas contextuais, combinando modelos generativos com recuperação 
    de dados em tempo real. Pós-graduado em Inteligência Artificial e Machine Learning (XP Educação), com formação 
    complementar em Gestão de Negócios e Ciências da Saúde. Trabalho na interseção entre tecnologia, dados e estratégia"""
]

# Inicializando os modelos
class RAGSystem:
    def __init__(self):
        # Modelo para embeddings
        self.embed_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.embed_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        # Modelo gerador
        self.gen_tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.gen_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        
        # Criando índice FAISS
        self.document_embeddings = self.embed_documents(documents)
        self.index = self.create_faiss_index(self.document_embeddings)
    
    def embed_documents(self, texts):
        inputs = self.embed_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.embed_model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.numpy()
    
    def create_faiss_index(self, embeddings):
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index
    
    def retrieve_documents(self, question, top_k=1):
        question_embedding = self.embed_documents([question])
        distances, indices = self.index.search(question_embedding, top_k)
        return [documents[idx] for idx in indices[0]]
    
    def generate_answer(self, context, question):
        input_text = f"contexto: {context} pergunta: {question}"
        inputs = self.gen_tokenizer.encode(input_text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.gen_model.generate(inputs, max_length=150, num_beams=4, temperature=0.7)
        return self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def answer_question(self, question):
        retrieved_docs = self.retrieve_documents(question)
        context = " ".join(retrieved_docs)
        return self.generate_answer(context, question)

# Inicializando o sistema
rag_system = RAGSystem()


# Testando o sistema RAG
def test_rag(question):
    print("Pergunta:", question)
    print("\nResposta:", rag_system.answer_question(question))
    print("-" * 50)

# Exemplos de perguntas
perguntas = [
    "Quais são as áreas de especialização de Danilo?",
    "Qual é a formação acadêmica de Danilo?",
    "Quais tecnologias Danilo utiliza no seu trabalho?"
]

for pergunta in perguntas:
    test_rag(pergunta)

