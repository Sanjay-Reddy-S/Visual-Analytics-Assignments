"""
Now let us consider, the whole dataset as a whole.
ALL the plots across tabs are linked and selecting few interesting points... 
will highlight the same points across all the plots

As illustrated in aa_Run_Me_First_Two_Feature_clustering we use normalization when it comes to DBScan and
elbow_method to find number of clusters w.r.t to KMeans

Later, using correlation between variables, I remove 'Grocery' (which has very high correlation with both 'Milk' and 'Detergent_Paper') and rerun the whole experiment

Here, many of the .html files are commented out because they might cause the browser to crash. ALL the .html files are already present in the zip folder (Opening them one at a time shouldn't be a problem).
Uncomment the show(.html) commands at your own risk.
"""

import random
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

actual_df = pd.read_csv('Wholesale customers data.csv',na_values=[''])
actual_df_2 = pd.read_csv('Wholesale customers data.csv',na_values=[''])

records=list(actual_df.index)
#print (records)
features=list(actual_df.columns)
#print (features)

y_pred=[0,0] #Labels,no_pf_labels

X=np.array([actual_df['Channel'],actual_df['Region'],actual_df['Fresh'],actual_df['Milk'],actual_df['Grocery'],actual_df['Frozen'],actual_df['Detergents_Paper'],actual_df['Delicassen']])
X=X.T

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
	px = figure(plot_width=1000, plot_height=800,tools=[hover,'pan','wheel_zoom','box_zoom','box_select'],title="Elbow-method KMeans",x_axis_label="Range",y_axis_label="Distortions")
	px.line([i for i in range(1,8)],distortions,line_width=2,color="red")
	show(px)

display_elbow_method(X,y_pred,"All_features")
y_pred[1]=5 #Using the elbow-method... most optimal number of clusters is 5

def draw_graphs(y_pred,output_name): #This function will create all the plots present in the html files. It will be called twice, once by KMeans and next by DBScan. It creates plots considering 2 features of the data at a time and categorizes into tabs
	global actual_df
	source=ColumnDataSource(data=dict(channel=actual_df['Channel'],region=actual_df['Region'],fresh=actual_df['Fresh'],milk=actual_df['Milk'],grocery=actual_df['Grocery'],frozen=actual_df['Frozen'],detergents_paper=actual_df['Detergents_Paper'],delicassen=actual_df['Delicassen']))

	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_ch_1 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Channel",y_axis_label="Fresh")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_ch_2 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Channel",y_axis_label="Region")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_ch_3 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Channel",y_axis_label="Milk")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_ch_4 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Channel",y_axis_label="Grocery")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_ch_5 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Channel",y_axis_label="Frozen")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_ch_6 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Channel",y_axis_label="Detergents_Paper")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_ch_7 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Channel",y_axis_label="Delicassen")

	#p_ch_1.x_range = Range1d(0, 20000) #In case you want to change the range of x-axis and y-axis by default
	#p_ch_2.y_range = Range1d(0, 20000)
	

	
	p_ch_1.circle('channel','fresh',source=source,size=10,color=get_color(y_pred))
	p_ch_2.circle('channel','region',source=source,size=10,color=get_color(y_pred))
	p_ch_3.circle('channel','milk',source=source,size=10,color=get_color(y_pred))
	p_ch_4.circle('channel','grocery',source=source,size=10,color=get_color(y_pred))
	p_ch_5.circle('channel','frozen',source=source,size=10,color=get_color(y_pred))
	p_ch_6.circle('channel','detergents_paper',source=source,size=10,color=get_color(y_pred))
	p_ch_7.circle('channel','delicassen',source=source,size=10,color=get_color(y_pred))

	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_re_1 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Region",y_axis_label="Channel")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_re_2 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Region",y_axis_label="Fresh")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_re_3 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Region",y_axis_label="Milk")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_re_4 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Region",y_axis_label="Grocery")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_re_5 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Region",y_axis_label="Frozen")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_re_6 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Region",y_axis_label="Detergents_Paper")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_re_7 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Region",y_axis_label="Delicassen")

	# p_re_1=.x_range = Range1d(0, 20000) #In case you want to change the range of x-axis and y-axis by default
	# p_re_2.y_range = Range1d(0, 20000)
	

	#for i in range(actual_df['Milk'].shape[0]):
	p_re_1.circle('region','channel',source=source,size=10,color=get_color(y_pred))
	p_re_2.circle('region','fresh',source=source,size=10,color=get_color(y_pred))
	p_re_3.circle('region','milk',source=source,size=10,color=get_color(y_pred))
	p_re_4.circle('region','grocery',source=source,size=10,color=get_color(y_pred))
	p_re_5.circle('region','frozen',source=source,size=10,color=get_color(y_pred))
	p_re_6.circle('region','detergents_paper',source=source,size=10,color=get_color(y_pred))
	p_re_7.circle('region','delicassen',source=source,size=10,color=get_color(y_pred))

	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_fh_1 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Fresh",y_axis_label="Channel")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_fh_2 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Fresh",y_axis_label="Region")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_fh_3 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Fresh",y_axis_label="Milk")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_fh_4 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Fresh",y_axis_label="Grocery")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_fh_5 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Fresh",y_axis_label="Frozen")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_fh_6 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Fresh",y_axis_label="Detergents_Paper")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_fh_7 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Fresh",y_axis_label="Delicassen")

	# p_fh_1.x_range = Range1d(0, 20000) #In case you want to change the range of x-axis and y-axis by default
	# p_fh_2.y_range = Range1d(0, 20000)
	


	p_fh_1.circle('fresh','channel',source=source,size=10,color=get_color(y_pred))
	p_fh_2.circle('fresh','region',source=source,size=10,color=get_color(y_pred))
	p_fh_3.circle('fresh','milk',source=source,size=10,color=get_color(y_pred))
	p_fh_4.circle('fresh','grocery',source=source,size=10,color=get_color(y_pred))
	p_fh_5.circle('fresh','frozen',source=source,size=10,color=get_color(y_pred))
	p_fh_6.circle('fresh','detergents_paper',source=source,size=10,color=get_color(y_pred))
	p_fh_7.circle('fresh','delicassen',source=source,size=10,color=get_color(y_pred))

	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_mi_1 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Milk",y_axis_label="Channel")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_mi_2 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Milk",y_axis_label="Region")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_mi_3 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Milk",y_axis_label="Fresh")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_mi_4 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Milk",y_axis_label="Grocery")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_mi_5 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Milk",y_axis_label="Frozen")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_mi_6 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Milk",y_axis_label="Detergents_Paper")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_mi_7 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Milk",y_axis_label="Delicassen")

	# p_mi_1.x_range = Range1d(0, 20000) #In case you want to change the range of x-axis and y-axis by default
	# p_mi_2.y_range = Range1d(0, 20000)
	


	
	p_mi_1.circle('milk','channel',source=source,size=10,color=get_color(y_pred))
	p_mi_2.circle('milk','region',source=source,size=10,color=get_color(y_pred))
	p_mi_3.circle('milk','fresh',source=source,size=10,color=get_color(y_pred))
	p_mi_4.circle('milk','grocery',source=source,size=10,color=get_color(y_pred))
	p_mi_5.circle('milk','frozen',source=source,size=10,color=get_color(y_pred))
	p_mi_6.circle('milk','detergents_paper',source=source,size=10,color=get_color(y_pred))
	p_mi_7.circle('milk','delicassen',source=source,size=10,color=get_color(y_pred))

	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_gr_1 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Grocery",y_axis_label="Channel")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_gr_2 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Grocery",y_axis_label="Region")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_gr_3 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Grocery",y_axis_label="Fresh")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_gr_4 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Grocery",y_axis_label="Milk")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_gr_5 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Grocery",y_axis_label="Frozen")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_gr_6 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Grocery",y_axis_label="Detergents_Paper")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_gr_7 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Grocery",y_axis_label="Delicassen")

	# p_gr_1.x_range = Range1d(0, 20000) #In case you want to change the range of x-axis and y-axis by default
	# p_gr_2.y_range = Range1d(0, 20000)
	


	
	p_gr_1.circle('grocery','channel',source=source,size=10,color=get_color(y_pred))
	p_gr_2.circle('grocery','region',source=source,size=10,color=get_color(y_pred))
	p_gr_3.circle('grocery','fresh',source=source,size=10,color=get_color(y_pred))
	p_gr_4.circle('grocery','milk',source=source,size=10,color=get_color(y_pred))
	p_gr_5.circle('grocery','frozen',source=source,size=10,color=get_color(y_pred))
	p_gr_6.circle('grocery','detergents_paper',source=source,size=10,color=get_color(y_pred))
	p_gr_7.circle('grocery','delicassen',source=source,size=10,color=get_color(y_pred))

	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_fz_1 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Frozen",y_axis_label="Channel")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_fz_2 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Frozen",y_axis_label="Region")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_fz_3 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Frozen",y_axis_label="Milk")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_fz_4 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Frozen",y_axis_label="Grocery")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_fz_5 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Frozen",y_axis_label="Fresh")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_fz_6 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Frozen",y_axis_label="Detergents_Paper")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_fz_7 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Frozen",y_axis_label="Delicassen")

	# p_fz_1.x_range = Range1d(0, 20000) #In case you want to change the range of x-axis and y-axis by default
	# p_fz_2.y_range = Range1d(0, 20000)
	


	p_fz_1.circle('frozen','channel',source=source,size=10,color=get_color(y_pred))
	p_fz_2.circle('frozen','region',source=source,size=10,color=get_color(y_pred))
	p_fz_3.circle('frozen','milk',source=source,size=10,color=get_color(y_pred))
	p_fz_4.circle('frozen','grocery',source=source,size=10,color=get_color(y_pred))
	p_fz_5.circle('frozen','fresh',source=source,size=10,color=get_color(y_pred))
	p_fz_6.circle('frozen','detergents_paper',source=source,size=10,color=get_color(y_pred))
	p_fz_7.circle('frozen','delicassen',source=source,size=10,color=get_color(y_pred))

	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_dp_1 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Detergents_Paper",y_axis_label="Channel")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_dp_2 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Detergents_Paper",y_axis_label="Region")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_dp_3 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Detergents_Paper",y_axis_label="Milk")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_dp_4 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Detergents_Paper",y_axis_label="Grocery")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_dp_5 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Detergents_Paper",y_axis_label="Fresh")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_dp_6 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Detergents_Paper",y_axis_label="Frozen")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_dp_7 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Detergents_Paper",y_axis_label="Delicassen")

	# p_dp_1.x_range = Range1d(0, 20000) #In case you want to change the range of x-axis and y-axis by default
	# p_dp_2.y_range = Range1d(0, 20000)
	

	#for i in range(actual_df['Milk'].shape[0]):
	p_dp_1.circle('detergents_paper','channel',source=source,size=10,color=get_color(y_pred))
	p_dp_2.circle('detergents_paper','region',source=source,size=10,color=get_color(y_pred))
	p_dp_3.circle('detergents_paper','milk',source=source,size=10,color=get_color(y_pred))
	p_dp_4.circle('detergents_paper','grocery',source=source,size=10,color=get_color(y_pred))
	p_dp_5.circle('detergents_paper','fresh',source=source,size=10,color=get_color(y_pred))
	p_dp_6.circle('detergents_paper','frozen',source=source,size=10,color=get_color(y_pred))
	p_dp_7.circle('detergents_paper','delicassen',source=source,size=10,color=get_color(y_pred))

	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_dc_1 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Delicassen",y_axis_label="Channel")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_dc_2 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Delicassen",y_axis_label="Region")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_dc_3 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Delicassen",y_axis_label="Milk")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_dc_4 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Delicassen",y_axis_label="Grocery")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_dc_5 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Delicassen",y_axis_label="Fresh")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_dc_6 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Delicassen",y_axis_label="Frozen")
	hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
	p_dc_7 = figure(plot_width=300, plot_height=300,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Clusters",x_axis_label="Delicassen",y_axis_label="Detergents_Paper")

	# p_dc_1.x_range = Range1d(0, 20000) #In case you want to change the range of x-axis and y-axis by default
	# p_dc_2.y_range = Range1d(0, 20000)
	


	p_dc_1.circle('delicassen','channel',source=source,size=10,color=get_color(y_pred))
	p_dc_2.circle('delicassen','region',source=source,size=10,color=get_color(y_pred))
	p_dc_3.circle('delicassen','milk',source=source,size=10,color=get_color(y_pred))
	p_dc_4.circle('delicassen','grocery',source=source,size=10,color=get_color(y_pred))
	p_dc_5.circle('delicassen','fresh',source=source,size=10,color=get_color(y_pred))
	p_dc_6.circle('delicassen','frozen',source=source,size=10,color=get_color(y_pred))
	p_dc_7.circle('delicassen','detergents_paper',source=source,size=10,color=get_color(y_pred))

	tab1=Panel(child=gridplot([[p_ch_1, p_ch_2], [p_ch_3, p_ch_4],[p_ch_5, p_ch_6], [p_ch_7, None]]),title="Channel vs rest "+output_name)
	tab2=Panel(child=gridplot([[p_re_1, p_re_2], [p_re_3, p_re_4],[p_re_5, p_re_6], [p_re_7, None]]),title="Region vs rest "+output_name)
	tab3=Panel(child=gridplot([[p_fh_1, p_fh_2], [p_fh_3, p_fh_4],[p_fh_5, p_fh_6], [p_fh_7, None]]),title="Fresh vs rest "+output_name)
	tab4=Panel(child=gridplot([[p_mi_1, p_mi_2], [p_mi_3, p_mi_4],[p_mi_5, p_mi_6], [p_mi_7, None]]),title="Milk vs rest "+output_name)
	tab5=Panel(child=gridplot([[p_gr_1, p_gr_2], [p_gr_3, p_gr_4],[p_gr_5, p_gr_6], [p_gr_7, None]]),title="Grocery vs rest "+output_name)
	tab6=Panel(child=gridplot([[p_fz_1, p_fz_2], [p_fz_3, p_fz_4],[p_fz_5, p_fz_6], [p_fz_7, None]]),title="Frozen vs rest "+output_name)
	tab7=Panel(child=gridplot([[p_dp_1, p_dp_2], [p_dp_3, p_dp_4],[p_dp_5, p_dp_6], [p_dp_7, None]]),title="Detergents_Paper vs rest "+output_name)
	tab8=Panel(child=gridplot([[p_dc_1, p_dc_2], [p_dc_3, p_dc_4],[p_dc_5, p_dc_6], [p_dc_7, None]]),title="Delicassen vs rest "+output_name)

	return (tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8)

y_pred[0]=KMeans(n_clusters=y_pred[1]).fit_predict(X) #KMeans clustering with 5 clusters
tabs_tuple_KMeans=draw_graphs(y_pred,"KMeans")

slope_values=[]
min_values=[]
for col in features:
	slope_values.append(2.0/(actual_df[col].max()-actual_df[col].min()))
	min_values.append(actual_df[col].min())
for i in range(len(features)):
	actual_df_2[features[i]]=1.0+(actual_df[features[i]]-min_values[i])*((slope_values[i]))
X=np.array([actual_df_2['Channel'],actual_df_2['Region'],actual_df_2['Fresh'],actual_df_2['Milk'],actual_df_2['Grocery'],actual_df_2['Frozen'],actual_df_2['Detergents_Paper'],actual_df_2['Delicassen']])
X=X.T
y_pred[0]=DBSCAN(eps=0.3, min_samples=10).fit_predict(X)
tabs_tuple_DBScan=draw_graphs(y_pred,"DBScan")

output_file("Channel_vs_rest.html")
tabs=Tabs(tabs=[tabs_tuple_KMeans[0],tabs_tuple_DBScan[0]])
show(tabs)

output_file("Region_vs_rest.html")
tabs=Tabs(tabs=[tabs_tuple_KMeans[1],tabs_tuple_DBScan[1]])
show(tabs)

output_file("Fresh_vs_rest.html")
tabs=Tabs(tabs=[tabs_tuple_KMeans[2],tabs_tuple_DBScan[2]])
show(tabs)
"""
output_file("Milk_vs_rest.html")
tabs=Tabs(tabs=[tabs_tuple_KMeans[3],tabs_tuple_DBScan[3]])
show(tabs)

output_file("Grocery_vs_rest.html")
tabs=Tabs(tabs=[tabs_tuple_KMeans[4],tabs_tuple_DBScan[4]])
show(tabs)

output_file("Frozen_vs_rest.html")
tabs=Tabs(tabs=[tabs_tuple_KMeans[5],tabs_tuple_DBScan[5]])
show(tabs)

output_file("Detergents_Paper_vs_rest.html")
tabs=Tabs(tabs=[tabs_tuple_KMeans[6],tabs_tuple_DBScan[6]])
show(tabs)

output_file("Delicassen_vs_rest.html")
tabs=Tabs(tabs=[tabs_tuple_KMeans[7],tabs_tuple_DBScan[7]])
show(tabs)
"""
correlations=[]
for i in range(len(features)):
	for j in range(i+1,len(features)):
		corr=pearsonr(actual_df.iloc[:,i],actual_df.iloc[:,j])[0]
		correlations.append((features[i],features[j],corr))
correlations.sort(reverse=True,key=lambda x:x[2])
print (correlations[0:2])



X_nog=np.array([actual_df['Channel'],actual_df['Region'],actual_df['Fresh'],actual_df['Milk'],actual_df['Frozen'],actual_df['Detergents_Paper'],actual_df['Delicassen']])
X_nog=X_nog.T
y_pred[1]=3 #Found the optimal number of clusters as 3
y_pred[0]=KMeans(n_clusters=y_pred[1]).fit_predict(X_nog)
tabs_tuple_KMeans_no_grocery=draw_graphs(y_pred,"KMeans_No_Grocery")

X_nog=np.array([actual_df_2['Channel'],actual_df_2['Region'],actual_df_2['Fresh'],actual_df_2['Milk'],actual_df_2['Frozen'],actual_df_2['Detergents_Paper'],actual_df_2['Delicassen']])
X_nog=X_nog.T
y_pred[0]=DBSCAN(eps=0.3, min_samples=10).fit_predict(X)
tabs_tuple_DBScan_no_grocery=draw_graphs(y_pred,"DBScan_No_Grocery")
"""
output_file("Channel_vs_rest_No_Grocery.html")
tabs=Tabs(tabs=[tabs_tuple_KMeans_no_grocery[0],tabs_tuple_DBScan_no_grocery[0]])
show(tabs)

output_file("Region_vs_rest_No_Grocery.html")
tabs=Tabs(tabs=[tabs_tuple_KMeans_no_grocery[1],tabs_tuple_DBScan_no_grocery[1]])
show(tabs)

output_file("Fresh_vs_rest_No_Grocery.html")
tabs=Tabs(tabs=[tabs_tuple_KMeans_no_grocery[2],tabs_tuple_DBScan_no_grocery[2]])
show(tabs)

output_file("Milk_vs_rest_No_Grocery.html")
tabs=Tabs(tabs=[tabs_tuple_KMeans_no_grocery[3],tabs_tuple_DBScan_no_grocery[3]])
show(tabs)

output_file("Grocery_vs_rest_No_Grocery.html")
tabs=Tabs(tabs=[tabs_tuple_KMeans_no_grocery[4],tabs_tuple_DBScan_no_grocery[4]])
show(tabs)

output_file("Frozen_vs_rest_No_Grocery.html")
tabs=Tabs(tabs=[tabs_tuple_KMeans_no_grocery[5],tabs_tuple_DBScan_no_grocery[5]])
show(tabs)
"""
output_file("Detergents_Paper_vs_rest_No_Grocery.html")
tabs=Tabs(tabs=[tabs_tuple_KMeans_no_grocery[6],tabs_tuple_DBScan_no_grocery[6]])
show(tabs)

output_file("Delicassen_vs_rest_No_Grocery.html")
tabs=Tabs(tabs=[tabs_tuple_KMeans_no_grocery[7],tabs_tuple_DBScan_no_grocery[7]])
show(tabs)

"""
Elbow-method reference:
https://pythonprogramminglanguage.com/kmeans-elbow-method/

"""
