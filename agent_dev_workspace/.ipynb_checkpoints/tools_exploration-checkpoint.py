"""Tools for data exploration and similarity analysis."""

from typing import Any, Dict
from langchain_core.tools import tool
from sklearn.metrics.pairwise import cosine_similarity


def make_tools(session: Any) -> list:
    """Factory function creating tools bound to the provided session state."""

    @tool
    def cosine_similarity_tool(text1: str, text2: str) -> Dict[str, Any]:
        """Compute cosine similarity between two text documents.

        Requires that build_dtm_tool has already been run to fit and store
        the count vectorizer in session.artifacts['count_vectorizer'].

        Args:
            text1: First raw text string to compare.
            text2: Second raw text string to compare.

        Returns:
            Dict containing status, result_id, and numerical similarity_score.
        """
        if "count_vectorizer" not in session.artifacts:
            return {
                "status": "error",
                "message": "Missing 'count_vectorizer' in session.artifacts. Please run build_dtm_tool first.",
            }

        vectorizer = session.artifacts["count_vectorizer"]
        vecs = vectorizer.transform([text1, text2])

        similarity_matrix = cosine_similarity(vecs[0], vecs[1])
        score = float(similarity_matrix[0][0])
        score_rounded = round(score, 4)

        result_id = session.next_result_id("cosine_sim")

        args_summary = {
            "text1": text1[:30] + ("..." if len(text1) > 30 else ""),
            "text2": text2[:30] + ("..." if len(text2) > 30 else ""),
        }

        summary = {
            "result_id": result_id,
            "similarity_score": score_rounded,
        }

        session.store_result("cosine_similarity_tool", args_summary, summary)

        return summary

    return [cosine_similarity_tool]
