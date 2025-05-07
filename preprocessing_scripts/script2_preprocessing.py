import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

def full_preprocessing(df: pd.DataFrame, z_thresh: float = 3) -> pd.DataFrame:
    """
    Pipeline completo de preprocessing:
      1. Detección de missing values.
      2. Log-transform de BMI.
      3. Deteccion de outliers 
      4. Escalado estándar de variables numéricas.
    """
    df = df.copy()
    
    # 1. Detección de missings
    missings = df.isnull().sum().sum()
    print(f"Total de valores faltantes en el dataset: {missings}") 
    #(Como veremos que no hay missings, no hara falta imputacion ni nigun otro metodo)
    
    # 2. Log-transform de BMI 
    df['BMI'] = np.log1p(df['BMI'])

    # 3. Deteccion de outliers
    # Las variables (GenHlth, PhysHlth, MentHlth, Age, Education,
    # Income, BMI) tienen rangos naturales que justifican sus valores extremos, por lo que no habran outliers
    # Bmi_log en cambio, se puede ver que aun tiene una ligera cola alargada indicando outliers
    z_bmi = np.abs(zscore(df['BMI']))
    mask = z_bmi < z_thresh
    removed = len(df) - mask.sum()
    print(f"Outliers en BMI eliminados: {removed} filas")
    df = df.loc[mask].reset_index(drop=True)
      
    # 4. Escalado estándar
    scaler = StandardScaler()
    df[['BMI']] = scaler.fit_transform(df[['BMI']])
    
    return df

# Ejemplo de uso
if __name__ == "__main__":
    df = pd.read_csv("Data_before_10000.csv")
    df_pre = full_preprocessing(df)
    df_pre.to_csv("Data_preprocessed_10000.csv", index=False)

