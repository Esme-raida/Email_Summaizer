from rouge_score import rouge_scorer
import pandas as pd

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def compute_rouge_scores(reference, summary):
    if not isinstance(reference, str) or not isinstance(summary, str):
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scores = scorer.score(reference, summary)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }

def compute_rouge_for_dataframe(df, ref_col, hyp_col, prefix):
    all_scores = []
    for _, row in df.iterrows():
        scores = compute_rouge_scores(row[ref_col], row[hyp_col])
        all_scores.append(scores)
    scores_df = pd.DataFrame(all_scores).add_prefix(f"{prefix}_")
    return pd.concat([df.reset_index(drop=True), scores_df], axis=1)

