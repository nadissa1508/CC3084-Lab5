# Universidad del Valle de Guatemala  
## Departamento de Ciencias de la Computación  
## CC3084 Data Science 

# Laboratorio 5 - Clasificación de tweets sobre desastres

Proyecto del curso CC3084 Data Science de la Universidad del Valle de Guatemala. Se estudia el dataset **Natural Language Processing with Disaster Tweets** de Kaggle y se construye un modelo para clasificar si un tweet describe un desastre real (`target = 1`) o no (`target = 0`).

## Autores

### Cristian Túnchez (231359)  
### Nadissa López (23764)

## Alcance actual

Esta versión contiene el laboratorio completo: descripción de los datos, preprocesamiento explicado, unigramas, bigramas, nubes de palabras, comparación de clasificadores, análisis de sentimientos con VADER, variable de negatividad, evaluación de su aporte, selección final y función para clasificar tweets nuevos.

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
- `src/sentiment.py`: análisis VADER y generación reproducible de variables de sentimiento.
- `src/model.py`: partición estratificada, comparación, selección, evaluación y función final.

## Datos

Descargue los archivos desde [Natural Language Processing with Disaster Tweets en Kaggle](https://www.kaggle.com/competitions/nlp-getting-started/data). 

Coloque `train.csv` sin modificar en `data/raw/`. También se admite la carpeta que produce la descarga oficial:

```text
data/raw/nlp-getting-started/train.csv
```

El archivo. `data/processed/train_processed.csv` se produce al ejecutar el notebooks.

## Instalación

En PowerShell, desde la raíz del proyecto:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m nltk.downloader stopwords vader_lexicon
```

Se descargan `stopwords` y `vader_lexicon`. La tokenización usa expresiones regulares y no necesita `punkt`. No se utiliza spaCy porque el pipeline no aplica lematización.

## Ejecución

Inicie Jupyter y ejecute todas las celdas en orden:

```powershell
jupyter notebook notebooks/01_exploracion_clasificacion_tweets.ipynb
```

La ejecución completa carga el CSV crudo, crea `data/processed/train_processed.csv`, calcula tablas y gráficos, compara modelos mediante validación cruzada, analiza sentimientos y reporta la evaluación final.

Si solo se desea regenerar el dataset procesado desde el CSV crudo:

```powershell
python -m src.preprocessing
```

Para regenerar el dataset procesado incluyendo las variables de sentimiento:

```powershell
python -m src.sentiment
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
- No se aplica stemming ni lematización. El stemming reduce interpretabilidad y puede unir formas indebidamente; la lematización agrega costo y dependencias. Para este corpus corto en inglés, TF-IDF con `min_df` controla el vocabulario y deja términos legibles para interpretar el modelo.

## Fuga de información

El texto se limpia con reglas fijas que no aprenden del corpus. Después se realiza una partición estratificada 80/20. El `TfidfVectorizer` y la regresión logística viven en un `Pipeline` que se ajusta únicamente con el conjunto de entrenamiento; el conjunto de prueba solo se transforma y evalúa después. Por lo tanto, el vocabulario y los pesos IDF no incorporan información de prueba.

## Modelado y selección

Se comparan regresión logística, Naive Bayes multinomial y SVM lineal con TF-IDF de unigramas y de unigramas + bigramas. La selección usa F1 promedio de validación cruzada estratificada sobre entrenamiento. Después se compara la configuración elegida con y sin la variable `negativity`; el conjunto de prueba se reserva para el reporte final.

Se reportan `accuracy`, `precision`, `recall`, `F1` y matriz de confusión. F1 es especialmente relevante porque equilibra precision y recall de la clase de desastre real, y evita depender únicamente de accuracy cuando las clases no tienen exactamente el mismo tamaño.

## Sentimientos

VADER se aplica al texto crudo porque utiliza negaciones, puntuación, mayúsculas, emoticones e intensificadores. Se retira únicamente el símbolo `#` para permitir que el término del hashtag participe en el léxico. Las puntuaciones sirven para describir el corpus y crear `negativity`; no se interpretan como una medida de la gravedad objetiva del evento.

## Referencias

- Addison Howard, devrishi, Phil Culliton y Yufeng Guo. *Natural Language Processing with Disaster Tweets*. Kaggle, 2019.
- Daniel Jurafsky y James H. Martin. *Speech and Language Processing*.
- Feinerer, I., Hornik, K., & Meyer, D. (2008). Text Mining Infrastructure in R. *Journal of Statistical Software, 25*(5).
- Hutto, C. J., & Gilbert, E. (2014). VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text. *Proceedings of ICWSM, 8*(1), 216–225.
- Documentación de [scikit-learn](https://scikit-learn.org/), [NLTK](https://www.nltk.org/) y [wordcloud](https://amueller.github.io/word_cloud/).
