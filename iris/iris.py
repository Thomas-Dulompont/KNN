import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Import dataset
data = pd.read_csv('iris.csv')

y = data["species"] # Liste de toutes les especes
X = data.drop("species", axis=1) # Liste de toutes les infos sur les iris

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

k = 5

model = KNeighborsClassifier(n_neighbors=k) # Création model de sklearn
model.fit(X_train, y_train) # Entrainement
model.score(X_test, y_test) # Evaluation

# Range pour sepal
length_sepal = st.slider("Longueur du Sepal", min_value=4.0, max_value=8.0)
width_sepal = st.slider("Largeur du Sepal", min_value=2.0, max_value=4.5)

# Range pour width
length_petal = st.slider("Longueur d'une Pétale", min_value=0.2, max_value=7.0)
width_petal = st.slider("Largeur d'une Pétale", min_value=0.2, max_value=3.0)

# Dictionnaire pour stocker nos infos
d = {
    'sepal_length': [length_sepal], 
    'sepal_width': [width_sepal], 
    'petal_length': [length_petal], 
    'petal_width': [width_petal], 
    'species': 'inconnue'
} 

# Création d'un dataframe à partir du dictionnaire
d = pd.DataFrame(data=d)

# Fusion des deux dataFrame
graphique = pd.concat([data, d])

# Prediction des valeurs entrées
a_determiner = (length_sepal, width_sepal, length_petal, width_petal)
espece = model.predict([a_determiner])

# Titre
st.title('Votre espece est :  {}'.format(espece[0]))

# Graphiques
st.pyplot(sns.FacetGrid(graphique, hue="species").map(plt.scatter, "sepal_length", "sepal_width").add_legend())
st.pyplot(sns.FacetGrid(graphique, hue="species").map(plt.scatter, "petal_length", "petal_width").add_legend())