import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    'IA1': ['Random', 'Random', 'MiniMax', 'MiniMaxalphabeta2'],
    'IA2': ['MiniMax', 'MiniMaxalphabeta', 'MiniMaxalphabeta','MiniMaxalphabeta'],
    'Taille_Plateau': [3, 3, 3, 3],
    'Winrate': [0.3, 0.4, 0.5, 0.6]
}

# Créer un DataFrame
df = pd.DataFrame(data)

# Exporter le DataFrame vers un fichier CSV
df.to_csv('donnees.csv', index=False)
# Lire les données à partir du fichier CSV
df = pd.read_csv('donnees.csv')

# Créer une table pivot pour la heatmap
heatmap_data = df.pivot_table(values='Winrate', index=['IA1', 'Taille_Plateau'], columns='IA2')

# Créer la heatmap avec seaborn
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt=".2f", linewidths=.5, cbar_kws={'label': 'Winrate'})
plt.title('Comparaison des Stratégies par Taille de Plateau')
plt.show()