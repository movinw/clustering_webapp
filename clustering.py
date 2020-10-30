import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

#Removing a warning that will come due to deprication
st.set_option('deprecation.showPyplotGlobalUse', False)

#Set the title of the app
st.title("A very quick clustering visualisation")

#Read in the data
data = pd.read_csv('data.csv')

#Set some text to show the originald dataframe
st.text("This is the original dataframe:")

#Show the original dataframe
st.dataframe(data)

#Replacing the strings with numerical values
data['diagnosis'] = data['diagnosis'].replace({'M':1,'B':0})

#This is the correlation heatmap
#This correlates the data
corr = data.corr()
#This plots the heatmap
sns.heatmap(corr,cmap='magma', annot=False)
st.pyplot()

#Drop the columns selected from the multiselect
#Select multiple columns
cols_drop = st.multiselect("Select columns to drop", data.columns)
#Drop columns
data = data.drop(cols_drop, axis=1)


#Show the new dataframe
#Text to show it's the new dataframe
st.text("Once features have been engineered, it looks like this:")
#Show the new dataframe
st.dataframe(data)

#Change the number of clusters you want the data to be broken up into
k=st.slider("Select the number of clusters",2,10)

#A button that clusters when you press it
if st.button("Cluster Results"):
	#Dropping the diagnosis column so that the features can be used to predict it
	X = data.drop('diagnosis', axis=1).values
	#Scaling the values of the dataframe
	X = StandardScaler().fit_transform(X)

	#Creating a KMeans clusterer called km
	km = KMeans(n_clusters=k, init="k-means++", n_init=10)

	#Passing our transformed dataframe into our KMeans clusterer
	km_pred = km.fit_predict(X)

	#Plotting the data
	#Making two plots side by side to share the Y axis
	f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

	#Plotting a scatter plot that shows the original data as two different colours
	ax1.scatter(X[:,0], X[:,1], c=data["diagnosis"], cmap="jet", edgecolor="None", alpha=0.35)
	ax1.set_title("Actual clusters of 2 diagnoses")

	#Plotting a scatter plot that shows the predicted data according to how many clusters we chose too
	ax2.scatter(X[:,0], X[:,1], c=km_pred, cmap="jet", edgecolor="None", alpha=0.35)
	ax2.set_title(f"KMeans clustering plot of {k} clusters")
	st.pyplot()


#Go to the command line where this is and type "streamlit run clustering.py" - or whatever your file is called