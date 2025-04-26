import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_barplots_all(input_csv: str, output_dir: str = "graficas"):
    """
    Genera barplots de frecuencias para todas las variables del CSV
    y los guarda en la carpeta 'graficas'.
    """
    # Crear carpeta si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Leer datos
    df = pd.read_csv(input_csv)
    
    # Generar y guardar cada barplot
    for col in df.columns:
        counts = df[col].value_counts().sort_index()
        fig, ax = plt.subplots()
        counts.plot(kind='bar', ax=ax)
        ax.set_title(f"Barplot de '{col}'")
        ax.set_xlabel(col)
        ax.set_ylabel("Frecuencia")
        plt.tight_layout()
        
        # Guardar figura
        file_path = os.path.join(output_dir, f"{col}.png")
        fig.savefig(file_path)
        plt.close(fig)
    
    print(f"Todas las gráficas se han guardado en la carpeta '{output_dir}'")

if __name__ == "__main__":
    plot_barplots_all("Data_preprocessed_full.csv")
