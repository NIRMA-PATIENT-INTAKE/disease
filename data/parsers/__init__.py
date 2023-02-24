def surrogate_remover(df):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: np.nan
                if x == np.nan
                else str(x).encode("utf-8", "replace").decode("utf-8")
            )

    return df
