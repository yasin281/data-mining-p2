import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

def full_preprocessing(df: pd.DataFrame,
                       z_thresh: float = 3.0) -> pd.DataFrame:
    """
    Pipeline completo de preprocessing:
      1. Detección de missing values.
      2. Eliminación de outliers por Z‑score (con reporte de cuántos se eliminan).
      3. Log-transform de BMI.
      4. Escalado estándar de variables numericas.
    """
    df = df.copy()
    
    # 1. Detección de missings
    missings = df.isnull().sum().sum()
    print(f"Total de valores faltantes en el dataset: {missings}")
    
    # 2. (No hay imputación porque no hay missings)
    
    # 3. Detección y eliminación de outliers via Z-score
    num_cols = ['PhysHlth', 'MentHlth', 'Age', 'Education', 'Income', 'BMI']
    initial_count = len(df)
    z_scores = np.abs(stats.zscore(df[num_cols]))
    mask = (z_scores < z_thresh).all(axis=1)
    df = df.loc[mask].reset_index(drop=True)
    removed = initial_count - len(df)
    if removed > 0:
        print(f"Outliers eliminados: {removed} filas (de {initial_count} a {len(df)})")
    else:
        print("No se eliminaron outliers (ninguna fila superó el umbral).")
    
    # 4. Log-transform de BMI 
    df['BMI_log'] = np.log1p(df['BMI'])

    # 5. Escalado estándar
    cols_to_scale = ['PhysHlth', 'MentHlth', 'Age', 'Education', 'Income', 'BMI_log']
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    return df

# Ejemplo de uso
if __name__ == "__main__":
    df = pd.read_csv("Data_sample_10000.csv")
    df_pre = full_preprocessing(df)
    df_pre.to_csv("Data_preprocessed_full.csv", index=False)

