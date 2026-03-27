import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

#Leemos el dataset WeatherAUS

df = pd.read_csv('weatherAUS.csv')
cols = ['Cloud3pm', 'Humidity3pm', 'Pressure3pm', 'RainToday', 'RainTomorrow']

print("\n### FASE 1: Teorema de Bayes ###\n")

# 1. Seleccione una estación meteorológica especifica del dataset( ej. Sydney o Perth). 
# Se debe hacer para tres estaciones distintas.

estaciones = ['Albury', 'Richmond', 'Sydney']

for ciudad in estaciones:
    # Filtrar por ciudad y quitar nulos en las columnas de interés
    data_city = df[df['Location'] == ciudad].dropna(subset=cols)

    # 2. Calcule manualmente, utilizando el conteo de registros, las siguientes probabilidades.
    # a. Prior P(LluviaTomorrow): Probabilidad de que llueva mañana sin conocer ninguna variable.
    total = len(data_city)
    lluvia_si = len(data_city[data_city['RainTomorrow'] == 'Yes'])

    prior = lluvia_si / total
    
    # b. Likelihood P(Nubes|LluviaTomorrow): 
    # Probabilidad de que hoy a las 3:00 PM esté nublado (Cloud3pm>5), dado que mañana efectivamente llovió.
    nubes_dado_lluvia = len(data_city[(data_city['RainTomorrow'] == 'Yes') & (data_city['Cloud3pm'] > 5)])
    likelihood = nubes_dado_lluvia / lluvia_si

    # c. Evidencia P(Nubes) Probabilidad total de que hoy a las 3:00PM este nublado.
    nubes_si = len(data_city[data_city['Cloud3pm'] > 5])
    evidencia = nubes_si / total

    # 3. Aplique el teorema de bayes para encontrar la Probabilidad Posterior:
    #   Teorema de Bayes: P(Lluvia | Nubes)
    posterior = (likelihood * prior) / evidencia

    # Resultados

    print(f"========== {ciudad} ==========")

    print(f"----------  Conteos ----------")
    print(f"Total registros:               {total}")
    print(f"Dias que llovio mañana:        {lluvia_si}")
    print(f"Dias nublados (Cloud3pm > 5):  {nubes_si}")
    print(f"Dias nublados Y llovio mañana: {nubes_dado_lluvia}")

    print(f"--------- Resultados ---------")
    print(f"Prior P(Lluvia):            {prior:.4f}")
    print(f"Likelihood P(Nubes|Lluvia): {likelihood:.4f}")
    print(f"Evidencia P(Nubes):         {evidencia:.4f}")
    print(f"POSTERIOR P(Lluvia|Nubes):  {posterior:.4f}")

    # 4. Pregunta a responder: 
    # ¿Cómo cambia la probabilidad de lluvia mañana si sabemos que hoy esta nublado respecto a la probabilidad base (Prior)?
    print(f"Cambio respecto al Prior:   {((posterior - prior) / prior) * 100:.2f}%")
    print("\n")

    
print("\n### FASE 2: Clasificación con Naive Bayes ###\n")

data = df[cols].copy()

# Manejo de valores nulos (Imputación)
# Para numéricas usamos la mediana; para categóricas la moda.
data['Cloud3pm'] = data['Cloud3pm'].fillna(data['Cloud3pm'].median())
data['Humidity3pm'] = data['Humidity3pm'].fillna(data['Humidity3pm'].median())
data['Pressure3pm'] = data['Pressure3pm'].fillna(data['Pressure3pm'].median())
data['RainToday'] = data['RainToday'].fillna(data['RainToday'].mode()[0])
data['RainTomorrow'] = data['RainTomorrow'].fillna(data['RainTomorrow'].mode()[0])

# variables categóricas
le = LabelEncoder()
data['RainToday'] = le.fit_transform(data['RainToday'])
data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])

# Separación de características (X) y objetivo (y)
X = data.drop('RainTomorrow', axis=1)
y = data['RainTomorrow']

# División en entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#ENTRENAMIENTO

model = GaussianNB()
model.fit(X_train, y_train)

#EVALUACIÓN Y EXTRACCIÓN DE MÉTRICAS 
y_pred = model.predict(X_test)

#Extraer valores de la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()

#Calcular métricas individuales
fase2_acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

#IMPRESIÓN DE RESULTADOS
print("\n" + "="*40)
print("       MÉTRICAS DE LA FASE 2")
print("="*40)
print(f"Verdaderos Negativos (TN): {tn}")
print(f"Falsos Positivos (FP):     {fp}")
print(f"Falsos Negativos (FN):     {fn}")
print(f"Verdaderos Positivos (TP): {tp}")
print("-" * 40)
print(f"Exactitud (Accuracy):      {fase2_acc:.4f}")
print(f"Precisión (Precision):     {prec:.4f}")         #Probabilidad de que sea cierta la afitmación de "llovera"
print(f"Sensibilidad (Recall):     {rec:.4f}")          #Porcentaje de lluvias reales que capturo el modelo
print(f"F1-Score:                  {f1:.4f}")           #porcentaje de que lo que dice sea cierto
print("="*40)

# Mostrar el gráfico
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Lluvia', 'Lluvia'], 
            yticklabels=['No Lluvia', 'Lluvia'])
plt.title('Matriz de Confusión Final')
plt.xlabel('Predicción del Modelo')
plt.ylabel('Valor Real (Dataset)')
plt.savefig('fase2_matrizConfusión.png')
plt.show()

print("\n### FASE 3: REDES BAYESIANAS ###\n")

data_fase3 = data.copy()

# Normalizacion de las variables
data_fase3['Cloud3pm'] = np.where(data_fase3['Cloud3pm'] <= 0, '0', 'Mayor a 0')
data_fase3['Pressure3pm'] = pd.qcut(data_fase3['Pressure3pm'].rank(method='first'), q=3, labels=['Baja', 'Media', 'Alta'])
data_fase3['Humidity3pm'] = pd.qcut(data_fase3['Humidity3pm'].rank(method='first'), q=3, labels=['Baja', 'Media', 'Alta'])
data_fase3['RainTomorrow'] = data_fase3['RainTomorrow'].map({1: 'Yes', 0: 'No'})

fase3_cols = ['Pressure3pm', 'Cloud3pm', 'Humidity3pm', 'RainTomorrow']

data_fase3_model = data_fase3[fase3_cols].copy()
for col in fase3_cols:
    data_fase3_model[col] = data_fase3_model[col].astype('category')

modelo_bayesiano = DiscreteBayesianNetwork([
    ('Pressure3pm', 'Cloud3pm'),     #Presión influye en las nubes
    ('Cloud3pm', 'Humidity3pm'),     #nubes influye en la humedad
    ('Humidity3pm', 'RainTomorrow'), #humedad influye en si llueve
    ('Pressure3pm', 'RainTomorrow')  #Presión influye en si lluebve
])


# Generacion del grafo de la red bayesiana
plt.figure(figsize=(8, 5))
grafo_fase3 = nx.DiGraph()
grafo_fase3.add_edges_from(modelo_bayesiano.edges())
pos = nx.spring_layout(grafo_fase3, seed=42)
nx.draw(
    grafo_fase3,
    pos,
    with_labels=True,
    node_color='#9ecae1',
    node_size=3000,
    arrows=True,
    arrowsize=18,
    font_size=10,
    edge_color='#4a4a4a'
)
plt.title('Grafo de la Red Bayesiana - Fase 3')
plt.tight_layout()
plt.savefig('fase3_grafo_red_bayesiana.png', dpi=300)
plt.show()
plt.close()
print("Grafo de Fase 3 guardado en: grafo.png")


#Se calculan las tablas de probabilidad
#Mediante un recorrido en el dataset
modelo_bayesiano.fit(data=data_fase3_model, estimator=MaximumLikelihoodEstimator)

#Obtenemos respuestas especificas del modelo previamente entrenado
inferencia = VariableElimination(modelo_bayesiano)

print("\na. Probabilidad de lluvia mañana (dado: Presión baja, sin nubes):")
prob_lluvia = inferencia.query(
    variables=['RainTomorrow'], 
    evidence={'Pressure3pm': 'Baja', 'Cloud3pm': '0'}
)
print(prob_lluvia)

print("\nb. Análisis de causas probables dado que llovio ('Yes'):")
causa_presion = inferencia.query(variables=['Pressure3pm'], evidence={'RainTomorrow': 'Yes'})
causa_humedad = inferencia.query(variables=['Humidity3pm'], evidence={'RainTomorrow': 'Yes'})
causa_nubes = inferencia.query(variables=['Cloud3pm'], evidence={'RainTomorrow': 'Yes'})

print("\nDistribución de Presión:")
print(causa_presion)
print("\nDistribución de Humedad:")
print(causa_humedad)
print("\nDistribución de Nubes:")
print(causa_nubes)

# Comparador mínimo de rendimiento Fase 2 vs Fase 3
test_fase3 = data_fase3_model.loc[X_test.index, fase3_cols].copy()
pred_fase3 = modelo_bayesiano.predict(test_fase3[['Pressure3pm', 'Cloud3pm', 'Humidity3pm']])
fase3_acc = accuracy_score(test_fase3['RainTomorrow'], pred_fase3['RainTomorrow'])

print("\n### COMPARADOR DE RENDIMIENTO (mínimo) ###")
print(f"Fase 2 - GaussianNB (Accuracy): {fase2_acc:.4f}")
print(f"Fase 3 - Red Bayesiana (Accuracy): {fase3_acc:.4f}")
print("\n")
print("\n")
print("\n")