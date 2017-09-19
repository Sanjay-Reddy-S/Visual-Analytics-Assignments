import random
import pandas as pd
from bokeh.layouts import column
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool,ColumnDataSource
from bokeh.models.callbacks import CustomJS
from bokeh.layouts import widgetbox,layout
from bokeh.models.widgets import Panel,Tabs,Dropdown
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.svm import SVR
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def get_color():
	return (random.randrange(0,255),random.randrange(0,255),random.randrange(0,255))

miss_df = pd.read_csv('Wholesale customers data-missing.csv',na_values=[''])
small_df=miss_df.dropna(how='any')
actual_df = pd.read_csv('Wholesale customers data.csv',na_values=[''])

#print miss_df.head()
#print miss_df.shape

#print miss_df.isnull().sum(axis=0)
#print miss_df.isnull().sum(axis=1)

#print miss_df.loc[miss_df['Milk'].isnull()]

records=list(miss_df.index)
features=list(miss_df.columns)

#print features
missing_rows=[]
missing_row_index=[]
missing_row_indices=[]
missing_features=[]
for i in features:
	row=miss_df.loc[miss_df[i].isnull()]
	if row.empty:
		continue
	else:
		missing_row_index.append(list(row.index))
		missing_rows.append(row)
		for j in range(len(list(row.index))):
			missing_features.append(i)


for lst_j in missing_row_index:
	for j in lst_j:
		missing_row_indices.append(j)

#print missing_row_index #List of lists... contining which rows missing
#print missing_row_indices #Single row contining which rows missing
#print missing_rows #All rows which have NULL values
#print missing_features #List of all features having NULL (has duplicates)

actual_rows=[]
actual_values=[]
for lst_j in missing_row_index:
	for j in lst_j:
		actual_rows.append(actual_df.iloc[j])
for i in range(len(missing_features)):
	actual_values.append(actual_rows[i][missing_features[i]])

mean_values=[]
median_values=[]
random_values=[]
interpolate_values=[]
for i in missing_features:
	mean_values.append(miss_df[i].mean())
	median_values.append(miss_df[i].median())
	while(1):
		if(str(miss_df[i][random.randrange(0,len(records))])!='nan'):
			random_values.append(miss_df[i][random.randrange(0,len(records))])
			break

NN=3
for count in range(len(missing_features)):
	feature_index= features.index(missing_features[count])
	total=0.0
	for i in range(-1*NN,NN+1):
		if i!=0:
			total+=miss_df.iloc[missing_row_indices[count]-i,feature_index]
	interpolate_values.append(total/(2*NN))

#print interpolate_values
#print mean_values
#print median_values
#print random_values
#print actual_values

poly_values=[]

#print missing_features
for count in range(len(missing_features)):
	feature_index= features.index(missing_features[count])
	y=small_df.iloc[:,feature_index]
	#print y.shape[0]
	#print y
	x=np.array([i for i in range(y.shape[0]+len(missing_row_indices)) if i not in missing_row_indices])
	#print x.shape
	z=np.polyfit(x,y,1)
	p=np.poly1d(z)
	poly_degree=1
	error=mean_squared_error(y,p(x))
	for i in range(2,11):
		z=np.polyfit(x,y,i)
		p=np.poly1d(z)
		#print "Error: "+str(mean_squared_error(y,p(x)))
		if error>mean_squared_error(y,p(x)):
			poly_degree=i
	z=np.polyfit(x,y,poly_degree)
	p=np.poly1d(z)
	poly_values.append(p(missing_row_indices[count]))

knn_values=[]
svm_values=[]
linr_values=[]
logr_values=[]
svr_values=[]
for count in range(len(missing_features)):
	feature_index= features.index(missing_features[count])
	small_df_x=np.append(small_df.iloc[:,:feature_index],small_df.iloc[:,feature_index+1:],axis=1)
	small_df_y=small_df.iloc[:,feature_index]
	row= np.append(miss_df.iloc[missing_row_indices[count],:feature_index],miss_df.iloc[missing_row_indices[count],feature_index+1:],axis=1)
	
	clf=KNeighborsClassifier()
	clf.fit(small_df_x,small_df_y)
	knn_values.append(int(clf.predict(row)))

	clf=SVC()
	clf.fit(small_df_x,small_df_y)
	svm_values.append(int(clf.predict(row)))

	clf=linear_model.LinearRegression()
	clf.fit(small_df_x,small_df_y)
	linr_values.append(int(clf.predict(row)))

	clf=linear_model.LogisticRegression()
	clf.fit(small_df_x,small_df_y)
	logr_values.append(int(clf.predict(row)))

	clf=SVR(kernel='rbf')
	clf.fit(small_df_x,small_df_y)
	svr_values.append(int(clf.predict(row)))

#print knn_values
#print svm_values
#print linr_values
#print logr_values
#print svr_values

knn_values_2=[]
svm_values_2=[]
linr_values_2=[]
logr_values_2=[]
svr_values_2=[]

for count in range(len(missing_features)):
	feature_index= features.index(missing_features[count])
	small_df_x=small_df.iloc[:,:2]
	small_df_y=small_df.iloc[:,feature_index]
	row= miss_df.iloc[missing_row_indices[count],:2]
	print small_df_x.shape
	print small_df_y.shape
	print row.shape
	#break
	clf=KNeighborsClassifier()
	clf.fit(small_df_x,small_df_y)
	knn_values_2.append(int(clf.predict(row)))

	clf=SVC()
	clf.fit(small_df_x,small_df_y)
	svm_values_2.append(int(clf.predict(row)))

	clf=linear_model.LinearRegression()
	clf.fit(small_df_x,small_df_y)
	linr_values_2.append(int(clf.predict(row)))

	clf=linear_model.LogisticRegression()
	clf.fit(small_df_x,small_df_y)
	logr_values_2.append(int(clf.predict(row)))

	clf=SVR(kernel='rbf')
	clf.fit(small_df_x,small_df_y)
	svr_values_2.append(int(clf.predict(row)))

"""
y=small_df.iloc[:,3]
x=np.array([i for i in range(y.shape[0]+len(missing_row_indices)) if i not in missing_row_indices])
z=np.polyfit(x,y,1)
p=np.poly1d(z)
print p(180)

for i in range(len(features)):
	for j in range(i+1,len(features)):
		corr=pearsonr(small_df.iloc[:,i],small_df.iloc[:,j])[0]
		print "Correlation "+features[i]+","+features[j]+": "+str(corr)

feature_index= features.index('Milk')
#print small_df.iloc[:,:2].shape
#print small_df.iloc[:,4:].shape
small_df_x=np.append(small_df.iloc[:,:feature_index],small_df.iloc[:,feature_index+1:],axis=1)
print small_df_x.shape
small_df_y=small_df.iloc[:,feature_index]
print small_df_y.shape
row= np.append(miss_df.iloc[180,:feature_index],miss_df.iloc[180,feature_index+1:],axis=1)
clf=KNeighborsClassifier()
clf.fit(small_df_x,small_df_y)
print "KNN: "+str(clf.predict(row))

clf=SVC()
clf.fit(small_df_x,small_df_y)
print "SVM: "+str(clf.predict(row))

clf=linear_model.LinearRegression()
clf.fit(small_df_x,small_df_y)
print "Lin_R: "+str(clf.predict(row))

clf=linear_model.LogisticRegression()
clf.fit(small_df_x,small_df_y)
print "Log_R: "+str(clf.predict(row))

clf=SVR(kernel='rbf')
clf.fit(small_df_x,small_df_y)
"""

x_range=[(i+1) for i in range(len(actual_values))]

output_file("difference_lines.html")
hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
p = figure(plot_width=1000, plot_height=800,tools=[hover,'pan','wheel_zoom','box_zoom','box_select'],title="Difference in values",x_axis_label="Different Columns "+str(missing_features),y_axis_label="Number axis")
p.circle(x_range, actual_values,size=20,color=get_color(),legend="Actual Values")

p.circle(x_range, mean_values,size=20,color=get_color(),legend="Mean Values")
p.circle(x_range, median_values,size=20,color=get_color(),legend="Median Values")
p.circle(x_range, random_values,size=20,color=get_color(),legend="Random Values")
p.circle(x_range, interpolate_values,size=20,color=get_color(),legend="Interpolate Values")
p.circle(x_range, poly_values,size=20,color=get_color(),legend="Poly Values")

p.legend.click_policy="hide"
tab1=Panel(child=p,title="Interpolated Values")

p2 = figure(plot_width=1000, plot_height=800,tools=[hover,'pan','wheel_zoom','box_zoom','box_select'],title="Difference in values (considering only CHannel & Region as input)",x_axis_label="Different Columns "+str(missing_features),y_axis_label="Number axis")
p2.circle(x_range, actual_values,size=20,color=get_color(),legend="Actual Values")

p2.circle(x_range, knn_values_2,size=20,color=get_color(),legend="KNN Values")
p2.circle(x_range, svm_values_2,size=20,color=get_color(),legend="SVM Values")
p2.circle(x_range, linr_values_2,size=20,color=get_color(),legend="Linear Values")
p2.circle(x_range, logr_values_2,size=20,color=get_color(),legend="Logistic Values")
p2.circle(x_range, svr_values_2,size=20,color=get_color(),legend="SVR Values")

p2.legend.click_policy="hide"
tab2=Panel(child=p2,title="Channel & Region as inputs")

p3 = figure(plot_width=1000, plot_height=800,tools=[hover,'pan','wheel_zoom','box_zoom','box_select'],title="Difference in values ",x_axis_label="Different Columns "+str(missing_features),y_axis_label="Number axis")
p3.circle(x_range, actual_values,size=20,color=get_color(),legend="Actual Values")

p3.circle(x_range, knn_values,size=20,color=get_color(),legend="KNN Values")
p3.circle(x_range, svm_values,size=20,color=get_color(),legend="SVM Values")
p3.circle(x_range, linr_values,size=20,color=get_color(),legend="Linear Values")
p3.circle(x_range, logr_values,size=20,color=get_color(),legend="Logistic Values")
p3.circle(x_range, svr_values,size=20,color=get_color(),legend="SVR Values")

p3.legend.click_policy="hide"
tab3=Panel(child=p3,title="All except missing as inputs")

tabs=Tabs(tabs=[tab1,tab2,tab3])
#tabs=Tabs(tabs=[tab1,tab3])
show(tabs)

"""
https://stackoverflow.com/questions/19165259/python-numpy-scipy-curve-fitting
https://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html
"""