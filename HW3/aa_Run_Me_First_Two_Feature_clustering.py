"""
Just for easier comparison of both the methods...I'm only considering Region and Milk features in this example.
The reason for considering Region, because it being a nominal field, I feel it will give a natural sense of how the 
clustering is performing.
"""

import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool,ColumnDataSource,Range1d
from bokeh.layouts import gridplot
from bokeh.models.widgets import Panel,Tabs,Dropdown,DataTable, TableColumn
import numpy as np
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

actual_df = pd.read_csv('Wholesale customers data.csv',na_values=[''])
actual_df_2 = pd.read_csv('Wholesale customers data.csv',na_values=[''])

records=list(actual_df.index)
features=list(actual_df.columns)

def get_color(y_pred): #Returns a list of colors, based on the prediction made by the clustering method
	colors=[]
	for i in range(y_pred[0].shape[0]):
		num=y_pred[0][i]
		if num==-1:
			colors.append("pink")
		elif num==0:
			colors.append("red")
		elif num==1:
			colors.append("blue")
		elif num==2:
			colors.append("green")
		elif num==3:
			colors.append("yellow")
		elif num==4:
			colors.append("cyan")
		elif num==5:
			colors.append("purple")
		elif num==6:
			colors.append("navy")
		elif num==7:
			colors.append("orange")
	return colors

X=np.array([actual_df['Region'],actual_df['Milk']])
X=X.T
y_pred=[0,0] #[(numpy_array_of_labels),no_of_clusters]
hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])

#For KMeans clustering, we need to find the optimal number of clusters
#Here I'm using elbow-method, where I plot distortions vs range and at the first "elbow" bend, I mark the number of clusters

def display_elbow_method(X,y_pred,output_name):
	distortions = []
	K = range(1,8)
	for k in K:
		y_pred[1]=k
		clf=KMeans(n_clusters=y_pred[1])
		clf.fit(X)
		y_pred[0]=clf.fit_predict(X)
		distortions.append(sum(np.min(cdist(X, clf.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

	#print (distortions)

	output_file(output_name+"_Elbow.html")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	px = figure(plot_width=1000, plot_height=800,tools=[hover,'pan','wheel_zoom','box_zoom','box_select'],title="Elbow-method KMeans (for region and milk)",x_axis_label="Range",y_axis_label="Distortions")
	px.line([i for i in range(1,8)],distortions,line_width=2,color="red")
	show(px)

display_elbow_method(X,y_pred,"Region_Vs_Milk")
y_pred[1]=5 #Using the elbow-method... most optimal number of clusters is 5
y_pred[0]=KMeans(n_clusters=y_pred[1]).fit_predict(X) #KMeans clustering with 5 clusters

hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
p_ch_1 = figure(plot_width=1000, plot_height=800,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters in K-Means",x_axis_label="Region",y_axis_label="Milk")
source=ColumnDataSource(data=dict(region=actual_df['Region'],milk=actual_df['Milk']))
p_ch_1.circle('region','milk',source=source,size=10,color=get_color(y_pred))
output_file("Region_Vs_Milk_KMeans.html")
show(p_ch_1)

#When directly used with DBScan, it gives ALL points as noise, 
#because the distribution is varying WIDELY (especially in fields like Milk whose max is 73K and min is 55)
#Hence, I normalized the 'Milk' attribute by bringing it into the same range as Region [1,3]
#Note that, I used the normalization, only for predicting the labels, but the corresponding plots still 
#use the original data, so that the user can draw inferences directly from the actual data

slope_values=[]
min_values=[]

for col in features:
	slope_values.append(2.0/(actual_df[col].max()-actual_df[col].min()))
	min_values.append(actual_df[col].min())
for i in range(len(features)):
	actual_df_2[features[i]]=1.0+(actual_df[features[i]]-min_values[i])*((slope_values[i]))

X=np.array([actual_df_2['Region'],actual_df_2['Milk']])
X=X.T

y_pred[0]=DBSCAN(eps=0.3, min_samples=10).fit_predict(X) #DBScan clustering

hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
p_ch_1 = figure(plot_width=1000, plot_height=800,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters in DBScan",x_axis_label="Region",y_axis_label="Milk")
source=ColumnDataSource(data=dict(region=actual_df['Region'],milk=actual_df['Milk']))
output_file("Region_Vs_Milk_DBScan_3_10.html")
p_ch_1.circle('region','milk',source=source,size=10,color=get_color(y_pred))
show(p_ch_1)

"""
I've tried experimenting with varies values of eps and min_samples (An example follows, in case one wants to experiment further)
The clustering is more or less similar when it comes to DBScan. 
Two probable reasons for this might be because the number of records are too low (440 is too low a sample size)
or maybe even because the range of both variables is too small (b/w 1 to 3) and the density variation is too small 

y_pred[0]=DBSCAN(eps=0.5, min_samples=20).fit_predict(X) #DBScan clustering


p_ch_1 = figure(plot_width=1000, plot_height=800,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters in DBScan",x_axis_label="Region",y_axis_label="Milk")
source=ColumnDataSource(data=dict(region=actual_df['Region'],milk=actual_df['Milk']))
output_file("Region_Vs_Milk_DBScan_5_20.html")
p_ch_1.circle('region','milk',source=source,size=10,color=get_color(y_pred))
show(p_ch_1)
"""


#Now let us remove the outliers (as indicated by pink in DBScan) and rerun our elbow method and KMeans
noise_indices=[]
for i in range(len(y_pred[0])):
	if y_pred[0][i]==-1:
		noise_indices.append(i)

X=np.delete(X,noise_indices,0)

#print (actual_df.shape)
actual_df_3_r=[]
actual_df_3_m=[]
for i in range(actual_df.shape[0]):
	if i not in noise_indices:
		actual_df_3_r.append(actual_df.iloc[i]['Region'])
		actual_df_3_m.append(actual_df.iloc[i]['Milk'])

display_elbow_method(X,y_pred,"Region_Vs_Milk_No_Outliers")

y_pred[1]=3 #Using the elbow-method... most optimal number of clusters is 3
y_pred[0]=KMeans(n_clusters=y_pred[1]).fit_predict(X)
hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
p_ch_1 = figure(plot_width=1000, plot_height=800,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters in K-Means (No outliers)",x_axis_label="Region",y_axis_label="Milk")
p_ch_1.circle(actual_df_3_r,actual_df_3_m,size=10,color=get_color(y_pred))
output_file("Region_Vs_Milk_KMeans_No_Outliers.html")
show(p_ch_1)

#As indicated by this example, KMeans is affected a LOT by outliers 
#(As is expected, because it directly uses feature values to calculate the distances)

"""
Elbow-method reference:
https://pythonprogramminglanguage.com/kmeans-elbow-method/
"""
