# Universidad del Valle de Guatemala  
## Departamento de Ciencias de la Computación  
## CC3084 Data Science 

# Laboratorio 5 - Clasificación de tweets sobre desastres

Proyecto del curso CC3084 Data Science de la Universidad del Valle de Guatemala. La primera entrega estudia el dataset **Natural Language Processing with Disaster Tweets** de Kaggle y construye una línea base para clasificar si un tweet describe un desastre real (`target = 1`) o no (`target = 0`).

## Autores

### Cristian Túnchez (231359)  
### Nadissa López (23764)

## Alcance actual

Esta versión contiene la descripción de los datos, el preprocesamiento explicado, unigramas, bigramas, nubes de palabras y un modelo preliminar TF-IDF + regresión logística. Todavía no incluye análisis de sentimientos, variable de negatividad, función final de clasificación ni selección definitiva del mejor modelo; esas actividades pertenecen a la segunda entrega.

## Estructura

```text
CC3084-Lab5/
├── src/                 # funciones reutilizables
├── notebooks/           # exploración y presentación de resultados
├── data/
│   ├── raw/             # archivos originales; no se modifican
│   └── processed/       # archivos generados por código
├── requirements.txt
└── README.md
```

El notebook principal es `notebooks/01_exploracion_clasificacion_tweets.ipynb`. La lógica reutilizable está separada en:

- `src/preprocessing.py`: localización del CSV, limpieza, tokenización y creación del dataset procesado.
- `src/exploration.py`: resumen y gráficos del EDA.
- `src/ngrams.py`: frecuencias, probabilidades y comparación por clase.
- `src/model.py`: partición estratificada, pipeline TF-IDF + regresión logística, métricas e interpretación.

## Datos

Descargue los archivos desde [Natural Language Processing with Disaster Tweets en Kaggle](https://www.kaggle.com/competitions/nlp-getting-started/data). Kaggle puede solicitar iniciar sesión y aceptar las reglas de la competencia.

Coloque `train.csv` sin modificar en `data/raw/`. También se admite la carpeta que produce la descarga oficial:

```text
data/raw/nlp-getting-started/train.csv
```

El código nunca sobrescribe ese archivo. `data/processed/train_processed.csv` se produce al ejecutar el notebook y está ignorado por Git porque puede regenerarse.

## Instalación

En PowerShell, desde la raíz del proyecto:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m nltk.downloader stopwords
```

Solo se descarga el corpus `stopwords`; la tokenización usa expresiones regulares y no necesita `punkt`. No se utiliza spaCy porque la primera línea base no aplica lematización.

## Ejecución

Inicie Jupyter y ejecute todas las celdas en orden:

```powershell
jupyter notebook notebooks/01_exploracion_clasificacion_tweets.ipynb
```

La ejecución completa carga el CSV crudo, crea `data/processed/train_processed.csv`, calcula tablas y gráficos, entrena los modelos preliminares y reporta sus métricas. Las funciones también se pueden importar desde `src/` en otros notebooks de la segunda entrega.

Si solo se desea regenerar el dataset procesado desde el CSV crudo:

```powershell
python -m src.preprocessing
```

`exploration.py`, `ngrams.py` y `model.py` son módulos de funciones; el notebook los coordina para evitar duplicar análisis y figuras.

## Decisiones de preprocesamiento

- Se convierte a minúsculas para unificar variantes tipográficas.
- Las URLs y menciones se reemplazan por `token_url` y `token_user`. Se elimina su valor específico, pero se conserva la información de que existían.
- En hashtags se retira `#` y se conserva la palabra, porque puede resumir el tema del tweet.
- Los emojis se convierten en tokens basados en sus códigos Unicode. Así no se pierde una señal que será útil en la etapa posterior de sentimientos.
- Se eliminan signos de puntuación salvo el apóstrofe dentro de contracciones.
- Los números se conservan: cantidades, fechas y expresiones como `911` pueden aportar información sobre emergencias.
- Se eliminan stopwords inglesas de NLTK, pero se conservan negaciones como `no`, `not`, `never`, `don't` y `can't`.
- No se aplica stemming ni lematización. El stemming reduce interpretabilidad y puede unir formas indebidamente; la lematización agrega costo y dependencias. Para este corpus corto en inglés, TF-IDF con `min_df` controla el vocabulario y deja términos legibles para interpretar el modelo. Esta decisión se podrá reevaluar en la segunda entrega.

## Fuga de información

El texto se limpia con reglas fijas que no aprenden del corpus. Después se realiza una partición estratificada 80/20. El `TfidfVectorizer` y la regresión logística viven en un `Pipeline` que se ajusta únicamente con el conjunto de entrenamiento; el conjunto de prueba solo se transforma y evalúa después. Por lo tanto, el vocabulario y los pesos IDF no incorporan información de prueba.

## Modelo preliminar

Se utiliza regresión logística por ser una línea base sólida e interpretable para matrices dispersas. Se observan dos representaciones exploratorias: TF-IDF con unigramas y TF-IDF con unigramas + bigramas. Esta comparación no constituye todavía la selección definitiva del mejor modelo.

Se reportan `accuracy`, `precision`, `recall`, `F1` y matriz de confusión. F1 es especialmente relevante porque equilibra precision y recall de la clase de desastre real, y evita depender únicamente de accuracy cuando las clases no tienen exactamente el mismo tamaño.

## Referencias

- Addison Howard, devrishi, Phil Culliton y Yufeng Guo. *Natural Language Processing with Disaster Tweets*. Kaggle, 2019.
- Daniel Jurafsky y James H. Martin. *Speech and Language Processing*.
- Feinerer, I., Hornik, K., & Meyer, D. (2008). Text Mining Infrastructure in R. *Journal of Statistical Software, 25*(5).
- Documentación de [scikit-learn](https://scikit-learn.org/), [NLTK](https://www.nltk.org/) y [wordcloud](https://amueller.github.io/word_cloud/).
