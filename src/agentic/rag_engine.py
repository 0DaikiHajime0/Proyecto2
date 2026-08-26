"""
Motor RAG para analisis de Churn
---------------------------------
Utiliza Retrieval-Augmented Generation para analizar patrones
de churn y generar recomendaciones de retencion.
"""
import pandas as pd
import numpy as np
from pathlib import Path


class ChurnRAGEngine:
    """Motor RAG para analisis de churn con embeddings y retrieval."""

    def __init__(self, data_path: str = None):
        self.data = None
        self.documents = []
        self.vector_store = None
        self.embeddings = None
        if data_path:
            self.load_data(data_path)

    def load_data(self, path: str):
        """Carga datos historicos de churn como documentos."""
        self.data = pd.read_csv(path)
        churn_dtype = str(self.data['Churn'].dtype)
        if churn_dtype in ('object', 'string', 'str'):
            self.data['Churn'] = self.data['Churn'].map({'Yes': 1, 'No': 0}).astype(float)
        self.data['TotalCharges'] = pd.to_numeric(self.data['TotalCharges'], errors='coerce')
        self.data['TotalCharges'] = self.data['TotalCharges'].fillna(self.data['TotalCharges'].median())
        self.data['num_services'] = 0
        self.documents = self._create_documents()
        print(f"Documentos creados: {len(self.documents)}")

    def _create_documents(self) -> list:
        """Convierte registros en documentos de texto descriptivos."""
        if self.data is None:
            return []

        documents = []
        for _, row in self.data.iterrows():
            churn_label = "abandono" if row.get('Churn') == 1 else "permanencia"

            doc = (
                f"Cliente con {row.get('tenure', 0)} meses de antiguedad, "
                f"cargo mensual de ${row.get('MonthlyCharges', 0):.2f}, "
                f"cargo total de ${row.get('TotalCharges', 0):.2f}. "
                f"Contrato: {row.get('Contract', 'N/A')}. "
                f"Servicio de internet: {row.get('InternetService', 'N/A')}. "
                f"Servicios contratados: {row.get('num_services', 0)}. "
                f"Resultado: {churn_label}."
            )
            documents.append(doc)
        return documents

    def build_vector_store(self):
        """Construye el vector store con FAISS."""
        try:
            from langchain.embeddings import OpenAIEmbeddings
            from langchain.vectorstores import FAISS
            from langchain.docstore import InMemoryDocstore
            from langchain.schema import Document

            docs = [
                Document(page_content=doc, metadata={"index": i})
                for i, doc in enumerate(self.documents)
            ]

            self.embeddings = OpenAIEmbeddings()
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            print(f"Vector store construido con {len(docs)} documentos")
        except ImportError:
            print("Instalar langchain y faiss para usar RAG")
        except Exception as e:
            print(f"Error construyendo vector store: {e}")

    def retrieve(self, query: str, k: int = 5) -> list:
        """Recupera documentos relevantes por similitud."""
        if self.vector_store is None:
            return self._simple_retrieve(query, k)
        return self.vector_store.similarity_search(query, k=k)

    def _simple_retrieve(self, query: str, k: int = 5) -> list:
        """Retrieval basico sin embeddings (fallback)."""
        query_lower = query.lower()
        scored = []
        for i, doc in enumerate(self.documents):
            score = sum(1 for word in query_lower.split() if word in doc.lower())
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:k]]

    def get_churn_stats(self) -> dict:
        """Retorna estadisticas de churn para contexto."""
        if self.data is None:
            return {}
        churn_data = self.data[self.data['Churn'] == 1]
        no_churn_data = self.data[self.data['Churn'] == 0]
        return {
            "total_clientes": len(self.data),
            "churn_rate": float(self.data['Churn'].mean()),
            "avg_tenure_churn": float(churn_data['tenure'].mean()),
            "avg_tenure_no_churn": float(no_churn_data['tenure'].mean()),
            "avg_monthly_churn": float(churn_data['MonthlyCharges'].mean()),
            "avg_monthly_no_churn": float(no_churn_data['MonthlyCharges'].mean()),
            "top_contract_churn": str(churn_data['Contract'].mode().iloc[0]) if len(churn_data) > 0 else "N/A",
        }
