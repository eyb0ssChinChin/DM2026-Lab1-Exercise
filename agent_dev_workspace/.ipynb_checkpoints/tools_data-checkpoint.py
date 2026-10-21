import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from langchain_core.tools import tool

# Ensure NLTK punkt tokenizer data is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def make_tools(session):
    @tool
    def tokenize_tool(text_column: str = "text") -> dict:
        """Tokenize document text into unigram tokens using NLTK word_tokenize.

        Reads session.dataframe, lowercases the specified text column, and splits
        each document into a list of word tokens. Stores the resulting token lists
        in session.dataframe['unigrams'] and session.artifacts['unigrams'].

        Args:
            text_column: Name of the column in session.dataframe containing document text. Default is 'text'.

        Returns:
            A summary dictionary containing result_id, document count, and token statistics.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        df = session.dataframe
        if text_column not in df.columns:
            return {"error": f"Column '{text_column}' not found in dataframe."}

        # Tokenize each text entry (lowercased) into word unigrams
        def _tokenize(val):
            if pd.isna(val) or not str(val).strip():
                return []
            return word_tokenize(str(val).lower())

        tokens_series = df[text_column].apply(_tokenize)
        df["unigrams"] = tokens_series
        session.artifacts["unigrams"] = tokens_series.tolist()

        result_id = session.next_result_id("tokenize")
        total_tokens = sum(len(toks) for toks in tokens_series)

        summary = {
            "result_id": result_id,
            "n_documents": len(df),
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": float(total_tokens / len(df)) if len(df) > 0 else 0.0,
        }

        session.store_result("tokenize_tool", {"text_column": text_column}, summary)
        return summary

    return [tokenize_tool]
