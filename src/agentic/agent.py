"""
Agente de IA para analisis de Churn
------------------------------------
Agente que combina predicciones del modelo con RAG para
generar recomendaciones personalizadas de retencion.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from .rag_engine import ChurnRAGEngine


class ChurnAgent:
    """Agente de IA para analisis de churn y recomendaciones de retencion."""

    def __init__(self, model=None, rag_engine: Optional[ChurnRAGEngine] = None):
        self.model = model
        self.rag_engine = rag_engine

    def explain_prediction(self, customer_data: dict, prediction: int,
                           probability: float) -> str:
        """Explica la prediccion para un cliente especifico."""
        risk_level = "ALTO" if probability > 0.7 else "MEDIO" if probability > 0.4 else "BAJO"

        explanation = (
            f"**Analisis del Cliente**\n\n"
            f"- Riesgo de churn: {risk_level} ({probability:.1%})\n"
            f"- Antiguedad: {customer_data.get('tenure', 'N/A')} meses\n"
            f"- Cargo mensual: ${customer_data.get('MonthlyCharges', 0):.2f}\n"
            f"- Tipo de contrato: {customer_data.get('Contract', 'N/A')}\n"
            f"- Servicio internet: {customer_data.get('InternetService', 'N/A')}\n\n"
        )

        if prediction == 1:
            explanation += self._generate_risk_factors(customer_data)
        else:
            explanation += "**Cliente estable.** No se requiere accion inmediata."

        return explanation

    def _generate_risk_factors(self, customer_data: dict) -> str:
        """Identifica factores de riesgo del cliente."""
        factors = []

        if customer_data.get('Contract') == 'Month-to-month':
            factors.append("- Contrato mes a mes (mayor riesgo de abandono)")
        if customer_data.get('tenure', 0) < 12:
            factors.append("- Cliente nuevo (< 12 meses)")
        if customer_data.get('InternetService') == 'Fiber optic':
            factors.append("- Servicio de fibra optica (mayor churn)")
        if customer_data.get('OnlineSecurity') == 'No':
            factors.append("- Sin seguridad online")
        if customer_data.get('TechSupport') == 'No':
            factors.append("- Sin soporte tecnico")
        if customer_data.get('PaymentMethod') == 'Electronic check':
            factors.append("- Pago con cheque electronico")

        if factors:
            return "**Factores de Riesgo:**\n" + "\n".join(factors)
        return "No se identificaron factores de riesgo significativos."

    def generate_retention_recommendation(self, customer_data: dict,
                                          prediction: int,
                                          probability: float) -> str:
        """Genera recomendacion de retencion basada en prediccion."""
        if prediction == 0:
            return (
                "El cliente tiene baja probabilidad de abandono.\n"
                "Recomendacion: Mantener nivel de servicio actual."
            )

        recommendations = []

        if customer_data.get('Contract') == 'Month-to-month':
            recommendations.append(
                "1. Ofrecer descuento por cambio a contrato anual"
            )
        if customer_data.get('OnlineSecurity') == 'No':
            recommendations.append(
                "2. Ofrecer seguridad online gratis por 3 meses"
            )
        if customer_data.get('TechSupport') == 'No':
            recommendations.append(
                "3. Incluir soporte tecnico premium como beneficio"
            )
        if customer_data.get('InternetService') == 'Fiber optic':
            recommendations.append(
                "4. Revisar precio de fibra optica - considerar ajuste"
            )
        if customer_data.get('tenure', 0) < 12:
            recommendations.append(
                "5. Programa de fidelizacion para nuevos clientes"
            )

        if not recommendations:
            recommendations.append(
                "1. Contactar al cliente para evaluacion personalizada"
            )

        header = (
            f"**Plan de Retencion** (Riesgo: {probability:.1%})\n\n"
            "Acciones recomendadas:\n"
        )
        return header + "\n".join(recommendations)

    def analyze_cohort(self, customers_df: pd.DataFrame) -> dict:
        """Analiza un grupo de clientes y retorna insights."""
        if self.model is not None and 'Churn' not in customers_df.columns:
            X = customers_df.copy()
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(X)[:, 1]
                predictions = self.model.predict(X)
            else:
                probs = np.zeros(len(X))
                predictions = self.model.predict(X)

            high_risk = int((probs > 0.7).sum())
            medium_risk = int(((probs > 0.4) & (probs <= 0.7)).sum())
            low_risk = int((probs <= 0.4).sum())
        else:
            high_risk = int((customers_df.get('Churn', 0) == 1).sum()) if 'Churn' in customers_df.columns else 0
            medium_risk = 0
            low_risk = len(customers_df) - high_risk

        return {
            'total_customers': len(customers_df),
            'high_risk': high_risk,
            'medium_risk': medium_risk,
            'low_risk': low_risk,
            'risk_percentage': high_risk / len(customers_df) * 100 if len(customers_df) > 0 else 0,
        }

    def query(self, question: str) -> str:
        """Responde preguntas sobre los datos de churn usando RAG."""
        if self.rag_engine is None:
            return "Motor RAG no inicializado. Cargue datos primero."

        relevant_docs = self.rag_engine.retrieve(question, k=3)
        stats = self.rag_engine.get_churn_stats()

        context = "\n".join(relevant_docs[:3])
        response = (
            f"**Respuesta basada en datos historicos:**\n\n"
            f"Estadisticas generales:\n"
            f"- Total clientes: {stats.get('total_clientes', 0)}\n"
            f"- Churn rate: {stats.get('churn_rate', 0):.1%}\n"
            f"- Antiguedad promedio (churn): {stats.get('avg_tenure_churn', 0):.1f} meses\n"
            f"- Antiguedad promedio (sin churn): {stats.get('avg_tenure_no_churn', 0):.1f} meses\n"
            f"- Cargo mensual promedio (churn): ${stats.get('avg_monthly_churn', 0):.2f}\n"
            f"- Cargo mensual promedio (sin churn): ${stats.get('avg_monthly_no_churn', 0):.2f}\n\n"
            f"Documentos relevantes encontrados: {len(relevant_docs)}"
        )
        return response
