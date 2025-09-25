import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Imprimir directorio de trabajo actual y del script para depurar
print("Directorio de trabajo actual:", os.getcwd())
print("Directorio del script:", os.path.dirname(os.path.abspath(__file__)))
print("¿Existe el archivo CSV?:", os.path.exists('emails_dataset.csv'))

# Fase de preparación de datos
try:
    # Ruta relativa: archivo en el directorio de trabajo actual
    df = pd.read_csv('emails_dataset.csv')
except FileNotFoundError:
    print("Error: 'emails_dataset.csv' no encontrado en el directorio de trabajo actual.")
    print("Ruta esperada: C:/Users/IK/Documents/Proyectos U/MachineLearning/emails_dataset.csv")
    print("Alternativa: Usa la ruta absoluta: df = pd.read_csv('C:/Users/IK/Documents/Proyectos U/MachineLearning/emails_dataset.csv')")
    raise

# Combinar subject y body en una columna de texto
df['text'] = df['subject'] + ' ' + df['body']

# Keywords para etiquetar SPAM
spam_keywords = ['premio', 'bloqueada', 'crédito', 'descuento', 'inversión', 'ganado', 'gratis', 'oportunidad', 'seleccionada', 'oferta', 'promoción', 'medicamentos', 'viaje', 'dinero', 'rápido', 'tarjeta', 'cuenta', 'exclusiva', 'última', 'aprovecha', 'has ganado', 'viaja', 'gana']

# Etiquetar: 'SPAM' si contiene keywords, sino 'HAM'
df['label'] = df['text'].apply(lambda x: 'SPAM' if any(word.lower() in x.lower() for word in spam_keywords) else 'HAM')

# Vectorización con TF-IDF
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(df['text']).toarray()
y = df['label'].map({'SPAM': 1, 'HAM': 0})

# Listas para métricas
num_runs = 50
accuracies = []
f1_scores = []
precision_scores = []

# Modelo para el árbol final
last_model = None

for i in range(num_runs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    model = DecisionTreeClassifier(criterion='gini', max_depth=5)  # Limitar profundidad para gráfica
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    accuracies.append(acc)
    f1_scores.append(f1)
    precision_scores.append(prec)
    if i == num_runs - 1:  # Guardar el último modelo para la gráfica
        last_model = model

# Directorio del script para guardar la gráfica
script_dir = os.path.dirname(os.path.abspath(__file__))

# Gráfica: Visualizar árbol de decisiones
plt.figure(figsize=(25, 15))
plot_tree(
    last_model,
    feature_names=vectorizer.get_feature_names_out().tolist(),
    class_names=['HAM', 'SPAM'],
    filled=True,
    rounded=True,
    fontsize=12,
    proportion=True,
    precision=2,
    impurity=True
)
plt.title('Gráfica del Árbol de Decisiones', fontsize=16)
plt.savefig(os.path.join(script_dir, 'decision_tree.png'), dpi=300, bbox_inches='tight')
plt.show()

# Estadísticas resumidas 
mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)
mean_prec = np.mean(precision_scores)
std_prec = np.std(precision_scores)

print(f"Media Accuracy: {mean_acc:.4f} (Std: {std_acc:.4f})")
print(f"Media F1 Score: {mean_f1:.4f} (Std: {std_f1:.4f})")
print(f"Media Precision: {mean_prec:.4f} (Std: {std_prec:.4f})")

# Z-scores para Accuracy (y ejemplo de variaciones)
z_scores_acc = (np.array(accuracies) - mean_acc) / std_acc if std_acc > 0 else np.zeros(num_runs)
print("Z-scores para Accuracy (primeras 5 ejecuciones):", z_scores_acc[:5])

# Resumen adicional: Número de SPAM/HAM en el dataset
spam_count = np.sum(y == 1)
ham_count = np.sum(y == 0)
print(f"Distribución de clases: SPAM = {spam_count} ({spam_count/len(y)*100:.1f}%), HAM = {ham_count} ({ham_count/len(y)*100:.1f}%)")