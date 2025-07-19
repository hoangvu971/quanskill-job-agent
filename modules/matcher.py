import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class JobMatcher:
    """
    Matches resume embedding with job description embeddings.
    """

    @staticmethod
    def rank_jobs(resume_emb: np.ndarray, job_embs: np.ndarray, df_jobs: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
        """
        Compute cosine similarity between resume and each job posting.
        Return top_k ranked jobs.
        """
        similarities = cosine_similarity([resume_emb], job_embs)[0]
        df_jobs = df_jobs.copy()
        df_jobs["similarity"] = similarities
        return df_jobs.sort_values(by="similarity", ascending=False).head(top_k).reset_index(drop=True)
