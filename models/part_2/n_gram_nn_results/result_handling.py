import pandas as pd

def normalize_context_windows(df):
    normalized = df.copy()
    
    normalized['context_window_norm'] = None
    
    languages = normalized['language'].unique()

    for lang in languages:
        lang_mask = normalized['language'] == lang
        windows = normalized.loc[lang_mask, 'context_window']
        
        max_window = windows.max()
        # exclude 1 when looking for min
        min_window = windows[windows > 1].min()
        
        normalized.loc[lang_mask, 'context_window_norm'] = windows.apply(
            lambda x: 'max' if x == max_window else ('min' if x == min_window else None)
        )

    normalized = normalized.drop(columns=['context_window'])
    normalized = normalized.dropna(subset=['context_window_norm']).reset_index(drop=True)
    
    return normalized


def best_configuration(result_files, result_folder):
    dfs = []
    for lang, file in result_files.items():
        df = pd.read_csv(f"{result_folder}/{file}")
        df["language"] = lang
        df = normalize_context_windows(df)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    param_cols = ["replace_type", "top_fraction", "embedding_dim", "hidden_dim", "context_window_norm", "replace_frac"]

    # Pivot to have each language's loss as a separate column
    pivot_df = combined.pivot_table(
        index=param_cols,
        columns="language",
        values="loss"
    ).reset_index()

    # Compute mean and std across the three languages (row-wise)
    pivot_df["mean_loss"] = pivot_df[["ko", "ar", "te"]].mean(axis=1)
    pivot_df["std_loss"] = pivot_df[["ko", "ar", "te"]].std(axis=1)

    # Keep only configurations that have loss for all languages
    pivot_df = pivot_df.dropna(subset=["ko", "ar", "te"])
    return pivot_df, dfs