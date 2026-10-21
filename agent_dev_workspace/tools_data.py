"""tools_data.py - Data tools for loading, inspecting, tokenizing, and preprocessing data."""

from typing import Dict, Any, Optional
import os
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
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
    """Factory function creating data tools attached to the shared session."""

    @tool
    def tokenize_tool(text_column: str = "text") -> Dict[str, Any]:
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

    @tool
    def list_files_tool(subdirectory: str = "") -> Dict[str, Any]:
        """List files and subdirectories in a workspace directory.

        Args:
            subdirectory: Path relative to the workspace root to list files from. Defaults to root ("").

        Returns:
            Dict containing lists of directories and files found, and the result_id.
        """
        base_dir = os.getcwd()
        target_dir = os.path.normpath(os.path.join(base_dir, subdirectory)) if subdirectory else base_dir

        if not target_dir.startswith(base_dir):
            return {"error": "Access outside workspace directory is restricted."}

        if not os.path.exists(target_dir) or not os.path.isdir(target_dir):
            return {"error": f"Directory '{subdirectory}' does not exist."}

        entries = os.listdir(target_dir)
        dirs = []
        files = []

        for entry in sorted(entries):
            if entry.startswith("."):
                continue
            full_path = os.path.join(target_dir, entry)
            if os.path.isdir(full_path):
                dirs.append(entry)
            else:
                files.append(entry)

        summary = {
            "result_id": session.next_result_id("list_files"),
            "subdirectory": subdirectory,
            "directories": dirs,
            "files": files,
        }
        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def inspect_data_tool(n: int = 5) -> Dict[str, Any]:
        """Inspect the top rows and structure of the loaded working DataFrame.

        Args:
            n: Number of rows to peek at (default is 5).

        Returns:
            Dict containing dataframe dimensions, column information, and the first n rows.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Call load_dataset_tool first."}

        df = session.dataframe
        head_rows = df.head(n).to_dict(orient="records")

        summary = {
            "result_id": session.next_result_id("inspect_data"),
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "sample_rows": head_rows,
        }
        session.store_result("inspect_data_tool", {"n": n}, summary)
        return summary

    @tool
    def check_missing_tool(column: str = "text") -> Dict[str, Any]:
        """Check for missing or empty text values in the loaded dataset.

        Args:
            column: Name of the column to check for missing/empty values. Defaults to 'text'.

        Returns:
            Dict containing result_id, total_rows, missing_count, and missing_indices.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        df = session.dataframe
        if column not in df.columns:
            return {"error": f"Column '{column}' not found in dataframe."}

        # Identify rows that are null, NaN, or whitespace-only strings
        series = df[column]
        missing_mask = series.isna() | series.astype(str).str.strip().eq("")
        missing_indices = df.index[missing_mask].tolist()

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "total_rows": len(df),
            "missing_count": len(missing_indices),
            "missing_indices": missing_indices,
        }

        session.store_result("check_missing_tool", {"column": column}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> Dict[str, Any]:
        """Check for and optionally drop duplicate documents based on the 'text' column.

        Args:
            drop: If True, drops all copies of duplicated rows using keep=False. Defaults to False.

        Returns:
            Dict containing result_id, initial_rows, duplicates_found, and final_rows.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "Column 'text' not found in dataframe."}

        initial_rows = len(df)
        duplicates_mask = df.duplicated(subset=["text"], keep=False)
        duplicates_found = int(duplicates_mask.sum())

        if drop:
            df.drop_duplicates(subset=["text"], keep=False, inplace=True)

        final_rows = len(session.dataframe)
        result_id = session.next_result_id("check_duplicates")

        summary = {
            "result_id": result_id,
            "initial_rows": initial_rows,
            "duplicates_found": duplicates_found,
            "final_rows": final_rows,
        }

        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(n: int, random_state: Optional[int] = None) -> Dict[str, Any]:
        """Subsample the working DataFrame using pandas sample.

        Args:
            n: Number of samples to draw.
            random_state: Seed for random number generator.

        Returns:
            Dict containing result_id and total_rows after sampling.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        sampled_df = session.dataframe.sample(n=n, random_state=random_state)
        session.dataframe = sampled_df

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "n_sampled": len(session.dataframe),
            "total_rows": len(session.dataframe),
        }

        session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Compute statistical descriptions of text character lengths and generate box plot.

        Returns:
            Dict containing overall stats and per-category stats for text_length.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "Column 'text' not found in dataframe."}

        text_length = df["text"].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
        df_temp = df.copy()
        df_temp["text_length"] = text_length

        overall_stats = df_temp["text_length"].describe().to_dict()

        category_stats = {}
        if "category_name" in df_temp.columns:
            grouped = df_temp.groupby("category_name")["text_length"]
            for cat, group in grouped:
                category_stats[str(cat)] = group.describe().to_dict()

        # Build box plot using seaborn
        fig, ax = plt.subplots(figsize=(8, 6))
        if "category_name" in df_temp.columns:
            sns.boxplot(data=df_temp, x="category_name", y="text_length", ax=ax)
            ax.set_title("Text Length by Category")
            ax.set_xlabel("Category")
        else:
            sns.boxplot(data=df_temp, y="text_length", ax=ax)
            ax.set_title("Text Length Distribution")

        ax.set_ylabel("Text Length (characters)")
        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall": overall_stats,
            "by_category": category_stats,
        }

        session.store_result("describe_data_tool", {}, summary)
        return summary

    return [
        tokenize_tool,
        list_files_tool,
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool,
    ]
