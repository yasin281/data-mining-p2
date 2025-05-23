import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar tu muestra de 10 000 filas
df = pd.read_csv('Data_preprocessed_10000.csv')

# Asegúrate de que todo es numérico (label-encoded ya viene de Kaggle)
corr = df.corr(method='pearson')

plt.figure(figsize=(12,10))
sns.heatmap(corr,
            annot=False,          # True si quieres los números
            vmin=-1, vmax=1,
            cmap='coolwarm',
            linewidths=.5)
plt.title('Matriz de correlación (coeficiente de Pearson)')
plt.tight_layout()
plt.show()
