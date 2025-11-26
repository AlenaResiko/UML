import pandas as pd


def parquet_to_docs(path: str, text_col: str = "text", doc_col: str = "doc_id"):
    """
    Convert parquet with columns [doc_col, text_col] into docs: List[List[str]].
    Each doc is the list of its sentences in order.
    """
    df = pd.read_parquet(path)
    # Make sure we’re sorted by (doc_id, some position) if you have a pos column.
    if "sent_idx" in df.columns:
        df = df.sort_values([doc_col, "sent_idx"])
    else:
        df = df.sort_values(doc_col)

    docs: list[list[str]] = []
    for _, g in df.groupby(doc_col):
        docs.append(g[text_col].astype(str).tolist())
    return docs
