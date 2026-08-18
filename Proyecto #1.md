CC3092 Deep Learning y sistemas inteligentes

## Proyecto 1: Competencia de Modelación

## 1. Descripción general

El proyecto consiste en implementar un Multi-Layer Perceptron (MLP) capaz de realizar predicciones sobre el dataset entregado en el portal del curso. El objetivo central no es únicamente obtener el mejor modelo posible, sino documentar de forma rigurosa el proceso de desarrollo e investigación seguido para llegar a él: qué se probó, qué funcionó, qué no, y por qué.

El día de la presentación se entregará un dataset de prueba (held-out) sobre el cual se medirá el RMSE del modelo final de cada estudiante. La nota de competencia se calculará comparando el desempeño del modelo propio contra el del resto de la clase.

## Información clave

| Modalidad | Individual |
| --- | --- |
| Presentación (presencial) | 17 de agosto de 2026 |
| Entrega del trabajo escrito | 21 de agosto de 2026 |
| Métrica objetivo | RMSE (Root Mean Squared Error) |
| Modelo a implementar | Multi-Layer Perceptron (MLP) |

## Rúbrica

| Componente | Puntos | Total |
| --- | --- | --- |
| Trabajo escrito | 5 pts | 5 |
| Competencia (RMSE en dataset de prueba, comparado con la clase) | 3 pts | 3 |
| Total |   | 8 |


## 2. Estructura del trabajo escrito

El trabajo escrito debe incluir, como mínimo, las siguientes seis secciones.

## 2.1 Análisis exploratorio de datos (EDA)

Presenta un análisis exploratorio completo del dataset entregado antes de entrenar cualquier modelo.

- Dimensiones del dataset, tipos de variables (numéricas, categóricas, ordinales) y variable objetivo.

- Estadísticas descriptivas (media, mediana, desviación estándar, rangos) de las variables relevantes.

- Identificación y tratamiento de valores nulos, atípicos (outliers) y posibles inconsistencias.

- Visualizaciones: distribuciones de variables, correlaciones entre features, relación de cada feature con la variable objetivo.

- Decisiones de preprocesamiento derivadas del EDA (normalización, codificación de categóricas, eliminación o transformación de variables) y su justificación.

## 2.2 Metodología de desarrollo

Describe el proceso seguido para diseñar y entrenar el MLP.

Arquitectura(s) de red consideradas (número de capas, neuronas por capa, funciones de activación).

- Estrategia de división de datos (train/validation/test) y validación cruzada si aplica.

- Función de pérdida, optimizador, tasa de aprendizaje y demás hiperparámetros.

- Técnicas de regularización utilizadas (dropout, weight decay, early stopping, batch normalization, etc.).


## 2.3 Resultados de iteraciones

Documenta el historial de experimentos realizados durante el desarrollo del modelo, no solo el resultado final. Se recomienda una tabla que resuma cada iteración:

- Identificador de la iteración y cambios realizados respecto a la anterior (arquitectura, hiperparámetros, features, preprocesamiento).

- Métrica de RMSE obtenida en entrenamiento y validación para cada iteración.

- Curvas de entrenamiento (loss/RMSE vs. épocas) de las iteraciones más relevantes.

- Evidencia de problemas encontrados (overfitting, underfitting, inestabilidad del entrenamiento) y cómo se abordaron.

## 2.4 Discusión de resultados

Analiza e interpreta los resultados obtenidos, más allá de reportarlos.

- Comparación entre las distintas iteraciones: qué cambios tuvieron mayor impacto en el desempeño y por qué.

- Análisis de errores del modelo final (en qué casos falla más, si hay sesgos o patrones en los residuos).

- Discusión de las limitaciones del enfoque y del dataset.

- Reflexión sobre el trade-off entre complejidad del modelo y capacidad de generalización.

## 2.5 Conclusiones

Resume de forma concisa los hallazgos principales del proyecto:

- Desempeño final del modelo (RMSE) y su interpretación en el contexto del problema.

- Principales aprendizajes técnicos y metodológicos del proceso.

- Posibles mejoras o líneas de trabajo futuro si se contara con más tiempo o recursos.


## 2.6 Enlace al repositorio de GitHub

Incluye el enlace al repositorio público (o con acceso otorgado al curso) que contenga el código completo del proyecto: notebooks o scripts de EDA, entrenamiento, evaluación, y un README que explique cómo reproducir los resultados.

## 3. Competencia

El día de la presentación (17 de agosto) se entregará un dataset de prueba nuevo. Cada estudiante deberá generar predicciones con su modelo final y estas se evaluarán con RMSE.

La nota de competencia se asigna en función del desempeño relativo del modelo frente al resto de la clase: entre menor sea el RMSE en comparación con los demás modelos, mayor será la nota obtenida en este componente.

## Recomendaciones

- 1. Guardar el modelo entrenado (pesos, arquitectura y pipeline de preprocesamiento) de forma que puedas generar predicciones rápidamente el día de la presentación.

- 2. Asegurarse de que el pipeline de preprocesamiento aplicado al dataset de prueba sea idéntico al usado en entrenamiento (mismas transformaciones, mismos parámetros de normalización).

- 3. Tener un script o notebook listo para cargar el dataset de prueba, generar predicciones y calcular el RMSE sin pasos manuales adicionales.

- 4. Documentar desde ya cada iteración: es mucho más fácil escribir la sección de resultados si se lleva un registro continuo en lugar de reconstruirlo al final.
