import random
import numpy as np
import shapefile
#import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool,ColumnDataSource,Range1d,Select, Slider, Label
from bokeh.layouts import layout,column,widgetbox, row
from bokeh.io import output_notebook, curdoc

PATH2='VASTChal2014MC2-20140430/Geospatial/Abila'

sf2=shapefile.Reader(PATH2)
count=0
count2=0

p = figure(plot_width=400, plot_height=400,tools=['pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Map of Abila")
X_bokeh_7=[]
Y_bokeh_7=[]
for shape in list(sf2.iterShapes()):
	if count%500==0:
		print (count)
	#if count>290:
	#	break
	#x_lon_2 = np.zeros((len(shape.points),1))
	#y_lat_2 = np.zeros((len(shape.points),1))
	x_lon_2 = np.zeros((len(shape.points)))
	y_lat_2 = np.zeros((len(shape.points)))
	temp1=[]
	temp2=[]
	for ip in range(len(shape.points)):
		x_lon_2[ip] = shape.points[ip][0]
		y_lat_2[ip] = shape.points[ip][1]
		temp1.append(shape.points[ip][0])
		temp2.append(shape.points[ip][1])
		count2+=1
	X_bokeh_7.append(temp1)
	Y_bokeh_7.append(temp2)
	#X_bokeh_2=[]
	#Y_bokeh_2=[]
	#for i in range(len(x_lon_2)):
	#	print (str(x_lon_2[i])+","+str(y_lat_2[i])),
	#	X_bokeh_2.append(float(x_lon_2[i]))
	#	Y_bokeh_2.append(float(y_lat_2[i]))
	
	count+=1

print ("HERE: ")
print (x_lon_2.shape)
print (len(x_lon_2.tolist()))
print (count)
print (count2)
output_file("remove_7.html")
p.multi_line(X_bokeh_7,Y_bokeh_7,line_width=1,color='blue')
show(p)