import matplotlib.pyplot as plt
import numpy as np

# Dati di esempio
variables = ['Var1', 'Var2', 'Var3', 'Var4', 'Var5']
critical_values = [20, 35, 30, 35, 27]
non_critical_values = [25, 32, 34, 20, 25]

# Indici delle variabili
x = np.arange(len(variables))

# Larghezza delle barre
width = 0.35

fig, ax = plt.subplots()

# Creazione delle barre per i valori critici
rects1 = ax.bar(x - width/2, critical_values, width, label='Critical')
# Creazione delle barre per i valori non critici
rects2 = ax.bar(x + width/2, non_critical_values, width, label='Non-Critical')

# Aggiunta delle etichette, titolo e legenda
ax.set_ylabel('Values')
ax.set_title('Critical Value Diagram')
ax.set_xticks(x)
ax.set_xticklabels(variables)
ax.legend()

# Aggiunta delle etichette alle barre
def autolabel(rects):
    """Aggiunge etichette di valore sopra le barre"""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 punti di spostamento verso l'alto
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()
