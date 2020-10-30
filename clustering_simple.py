import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("A very quick clustering visualisation")

data = pd.read_csv('data.csv')

cols_drop = ['id', 'Unnamed: 32']
data = data.drop(cols_drop, axis=1)

data['diagnosis'] = data['diagnosis'].replace({'M':1,'B':0})


X = data.drop('diagnosis', axis=1).values
X = StandardScaler().fit_transform(X)

km = KMeans(n_clusters=2, init="k-means++", n_init=10)

km_pred = km.fit_predict(X)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.scatter(X[:,0], X[:,1], c=data["diagnosis"], cmap="jet", edgecolor="None", alpha=0.35)
ax1.set_title("Actual clusters of 2 diagnoses")

ax2.scatter(X[:,0], X[:,1], c=km_pred, cmap="jet", edgecolor="None", alpha=0.35)
ax2.set_title(f"KMeans clustering plot of 2 clusters")
st.pyplot()
