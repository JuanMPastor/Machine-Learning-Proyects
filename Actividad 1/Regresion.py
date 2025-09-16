import pandas as pd  
import re  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc  
from sklearn.preprocessing import StandardScaler  
import matplotlib.pyplot as plt  
import seaborn as sns  
import numpy as np  

# Carga del dataset
df = pd.read_csv('emails_dataset.csv')  

# Generación de pseudo-etiquetas
spam_keywords = ['oferta', 'gratis', 'premio', 'ganado', 'bloqueada', 'crÃ©dito', 'descuento', 'inversiÃ³n', 'promociÃ³n', 'viaje', 'dinero', 'trabajando', 'oportunidad', 'exclusiva', 'seleccionada', 'medicamentos', 'cancÃºn']
def label_email(row):
    text = (row['subject'] + ' ' + row['body']).lower() 
    if any(keyword in text for keyword in spam_keywords): 
        return 1  
    return 0  
df['label'] = df.apply(label_email, axis=1) 

# Extracción de features
def extract_features(row):
    text = row['body'] + ' ' + row['subject']  
    num_links = len(re.findall(r'http[s]?://', text))
    mentions_offer_free = 1 if re.search(r'\boferta\b|\bgratis\b|\bpromociÃ³n\b', text.lower()) else 0
    num_exclamations = text.count('!')
    sender_known = 1 if any(domain in row['email'] for domain in ['gmail.com', 'outlook.com', 'hotmail.com']) else 0
    num_words = len(text.split())
    num_images = 0
    has_attachment = 1 if 'adjunto' in text.lower() or 'adjuntamos' in text.lower() else 0
    all_caps_words = len([word for word in text.split() if word.isupper()])
    has_money_symbols = 1 if '$' in text or '%' in text else 0
    num_emoticons = len(re.findall(r'[:;][\)\(DPO]', text))
    return pd.Series([num_links, mentions_offer_free, num_exclamations, sender_known, num_words, num_images, has_attachment, all_caps_words, has_money_symbols, num_emoticons])

features = ['num_links', 'mentions_offer_free', 'num_exclamations', 'sender_known', 'num_words', 'num_images', 'has_attachment', 'all_caps_words', 'has_money_symbols', 'num_emoticons']
df[features] = df.apply(extract_features, axis=1)

# Mapeo de nombres técnicos a descripciones legibles
feature_names = {
    'num_links': 'Número de enlaces',
    'mentions_offer_free': 'Mención de oferta o gratis',
    'num_exclamations': 'Cantidad de signos de exclamación',
    'sender_known': 'Remitente conocido',
    'num_words': 'Cantidad de palabras',
    'num_images': 'Cantidad de imágenes',
    'has_attachment': 'Archivos adjuntos',
    'all_caps_words': 'Palabras en mayúsculas',
    'has_money_symbols': 'Signos de $ o %',
    'num_emoticons': 'Uso de emoticones'
}

# Seleccionar un subconjunto de features para el heatmap
selected_features = ['num_links', 'mentions_offer_free', 'num_exclamations', 'all_caps_words', 'has_money_symbols']
df_correlation = df[selected_features].rename(columns={f: feature_names[f] for f in selected_features})

# Preparación de datos
X = df[features]  # Matriz de features
y = df['label']  # Etiquetas 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Divide 80% train, 20% test
scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenamiento del modelo
model = LogisticRegression(max_iter=1000)  # Instancia el modelo
model.fit(X_train_scaled, y_train)  # Entrena con datos escalados

# Evaluación del modelo
y_pred = model.predict(X_test_scaled)  # Predice en test
accuracy = accuracy_score(y_test, y_pred)  # Calcula precisión
report = classification_report(y_test, y_pred)  # Reporte detallado

# Predicción en el dataset completo
X_full_scaled = scaler.transform(X)  # Escala todo el dataset
df['predicted_label'] = model.predict(X_full_scaled)  # Predicciones
df['classification'] = df['predicted_label'].apply(lambda x: 'Spam' if x == 1 else 'Ham')  # Mapea a strings
    
# Generación de gráficas
# 1. Distribución de etiquetas (ham/spam)
plt.figure(figsize=(6, 4))
sns.countplot(x='classification', data=df)
plt.title('Distribución de Ham vs Spam')
plt.xlabel('Clasificación')
plt.ylabel('Cantidad')
plt.tight_layout()
plt.show()

# 2. Curva ROC
y_prob = model.predict_proba(X_test_scaled)[:, 1] 
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 3. Importancia de las features (gráfico de puntos)
feature_importance = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_[0]})
feature_importance['Feature'] = feature_importance['Feature'].map(feature_names)  # Renombrar features
feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)
plt.figure(figsize=(8, 6))
plt.scatter(feature_importance['Coefficient'], feature_importance['Feature'], s=100, c=feature_importance['Coefficient'], cmap='coolwarm')
plt.colorbar(label='Coeficiente')
plt.title('Importancia de las Features (Coeficientes del Modelo)')
plt.xlabel('Coeficiente')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# Impresión de resultados en terminal
print("Precisión en el conjunto de test:", accuracy)
print("Reporte de Clasificación:\n", report)
print("\nClasificaciones del Dataset Completo (primeros 20 para brevedad):\n")
print(df[['id', 'subject', 'classification']].head(20).to_string(index=False))
print("\nConteo Total de Ham y Spam (balance aproximado 50/50):")
print(df['classification'].value_counts().to_string())