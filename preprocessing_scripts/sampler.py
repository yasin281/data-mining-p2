import pandas as pd

def random_sample_csv(input_path: str, output_path: str, sample_size: int = 10000, random_state: int = 42):
    """
    Carga un CSV, toma una muestra aleatoria de filas y guarda el resultado.
    
    Parámetros:
    - input_path: ruta al archivo CSV original.
    - output_path: ruta al CSV donde se guardará la muestra.
    - sample_size: número de filas a muestrear.
    - random_state: semilla para reproducibilidad.
    """
    df = pd.read_csv(input_path)
    df_sample = df.sample(n=sample_size, random_state=random_state, replace=False)
    df_sample.to_csv(output_path, index=False)
    print(f"Muestra de {sample_size} filas creada en '{output_path}'")

if __name__ == "__main__":
    random_sample_csv("Data.csv", "Data_sample_10000.csv")
