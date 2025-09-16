import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo CSV
df = pd.read_csv(r'C:\Users\IK\Documents\Proyectos U\MachineLearning\Actividad 2\IRIS.csv')  

# Mapear especies a números: 0 = Iris-setosa, 1 = Iris-versicolor, 2 = Iris-virginica
le = LabelEncoder()
df['target'] = le.fit_transform(df['species'])

# Características (features) y etiquetas (targets)
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = df['target'].values

# Dividir en train/test (80/20) para evaluación óptima
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluar el modelo
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Para clasificación: redondear las predicciones a la clase más cercana
y_train_class = np.round(y_train_pred).astype(int)
y_test_class = np.round(y_test_pred).astype(int)

# Métricas
train_accuracy = accuracy_score(y_train, y_train_class) * 100
test_accuracy = accuracy_score(y_test, y_test_class) * 100
mse_test = mean_squared_error(y_test, y_test_pred)

print(f"Precisión en train: {train_accuracy:.2f}%")
print(f"Precisión en test: {test_accuracy:.2f}%")
print(f"Error cuadrático medio (MSE) en test: {mse_test:.4f}")


# Preparar DataFrame para visualización 
df_test = pd.DataFrame(X_test, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df_test['true_species'] = le.inverse_transform(y_test)
df_test['pred_species'] = le.inverse_transform(y_test_class)

# Colores para cada especie
colors = {'Iris-setosa': 'blue', 'Iris-versicolor': 'green', 'Iris-virginica': 'red'}

# 1. Diagrama de Dispersión: Longitud-Petalo vs. Ancho-Petalo
plt.figure(figsize=(10, 6))
for species in df_test['true_species'].unique():
    subset = df_test[df_test['true_species'] == species]
    plt.scatter(subset['petal_length'], subset['petal_width'], 
                c=colors[species], label=f'Real {species}', alpha=0.6, s=100)
for species in df_test['pred_species'].unique():
    subset = df_test[df_test['pred_species'] == species]
    plt.scatter(subset['petal_length'], subset['petal_width'], 
                c=colors[species], marker='x', label=f'Pred {species}', s=100)
plt.xlabel('Longitud Petalo (cm)', fontsize=12)
plt.ylabel('Ancho Petalo (cm)', fontsize=12)
plt.title('Longitud Petalo vs. Ancho Petalo ', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('petal_scatter_iris.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Diagrama de Dispersión: Longitud-Sepalo vs. Ancho-Sepalo
plt.figure(figsize=(10, 6))
for species in df_test['true_species'].unique():
    subset = df_test[df_test['true_species'] == species]
    plt.scatter(subset['sepal_length'], subset['sepal_width'], 
                c=colors[species], label=f'Real {species}', alpha=0.6, s=100)
for species in df_test['pred_species'].unique():
    subset = df_test[df_test['pred_species'] == species]
    plt.scatter(subset['sepal_length'], subset['sepal_width'], 
                c=colors[species], marker='x', label=f'Pred {species}', s=100)
plt.xlabel('Longitud Sepalo (cm)', fontsize=12)
plt.ylabel('Ancho Sepalo (cm)', fontsize=12)
plt.title('Longitud Sepalo vs. Ancho Sepalo', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('sepal_scatter_iris.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Matriz de confusión
cm = confusion_matrix(y_test, y_test_class)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusión', fontsize=14)
plt.savefig('confusion_matrix_iris.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Diagrama de Dispersión: Predicciones vs. Valores Reales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.7, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Valores Reales', fontsize=12)
plt.ylabel('Predicciones', fontsize=12)
plt.title('Predicciones vs. Valores Reales', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('predictions_vs_actual_iris.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Histograma
residuals = y_test - y_test_pred
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=15, edgecolor='black', alpha=0.7)
plt.xlabel('(Real - Predicho)', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.title('Histograma de Residuos', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('residuals_histogram_iris.png', dpi=300, bbox_inches='tight')
plt.show()

