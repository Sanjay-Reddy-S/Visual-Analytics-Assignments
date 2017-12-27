import pandas as pd
import numpy as np
import shapefile
from datetime import datetime, time
#import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool,ColumnDataSource,Range1d,Select, Slider, Label
from bokeh.layouts import layout,column,widgetbox, row
from bokeh.io import output_notebook, curdoc

PATH='Geospatial/Kronos_Island'
PATH2='Geospatial/Abila'

FMT='%H:%M:%S'
s1 = '10:33:26'

s3="0:15:00"
s4="0:00:00"
#tdelta1 = datetime.strptime(s2, FMT) - datetime.strptime(s1, FMT)
tdelta_compare = datetime.strptime(s3,FMT)-datetime.strptime(s4,FMT)

sf1=shapefile.Reader(PATH)
sf2=shapefile.Reader(PATH2)
shape_ex=sf1.shape(0)
#x_lon = np.zeros((len(shape_ex.points),1))
#y_lat = np.zeros((len(shape_ex.points),1))
x_lon = np.zeros((len(shape_ex.points)))
y_lat = np.zeros((len(shape_ex.points)))
for ip in range(len(shape_ex.points)):
	x_lon[ip] = shape_ex.points[ip][0]
	y_lat[ip] = shape_ex.points[ip][1]

#output_file("remove.html")

car_id_name={}
car_name_id={}
fp=open("car-assignments.csv",'r')
for line in fp:
	if line[:9]=='LastName,':
		continue
	text=line.split(',')
	full_name=text[1]+" "+text[0]
	if text[2] not in car_id_name:
		car_id_name[text[2]]=full_name
	if full_name not in car_name_id:
		car_name_id[full_name]=text[2]


#print (car_id_name)
#print (car_id_name['35'])

dict_id={}#'35':{'01/06/2014':[[36.07,36.06],[24.87,24.87],[06:28:01,06:28:03],[Full_Name, Full_Name]]} Y followed by X
dict_cars={}
fp=open("gps.csv",'r')
for line in fp:
	if 'id' in line:
		continue
	line=line.replace('\n','')
	text=line.split(',')
	date=text[0][:10]
	time=text[0][11:]
	if text[1] not in dict_cars:
		dict_cars[text[1]]=''
	if text[1] in dict_id:
		if date in dict_id[text[1]]:
			dict_id[text[1]][date][0].append(float(text[2]))
			dict_id[text[1]][date][1].append(float(text[3]))
			dict_id[text[1]][date][2].append(time)
			dict_id[text[1]][date][3].append(car_id_name.get(text[1],"Unknown_Name"))
		else:
			dict_id[text[1]][date]=[[float(text[2])],[float(text[3])],[time],[car_id_name.get(text[1],"Unknown_Name")]]
	else:
		dict_id[text[1]]={}
		dict_id[text[1]][date]=[[float(text[2])],[float(text[3])],[time],[car_id_name.get(text[1],"Unknown_Name")]]

count=0
dict_id_2={} #For Car Stops
for car_id in dict_id:
	for date in dict_id[car_id]:
		lst=dict_id[car_id][date]
		for i in range(len(lst[1])-1):
			tdelta=datetime.strptime(lst[2][i+1],FMT)-datetime.strptime(lst[2][i],FMT)
			#print (tdelta)

			if tdelta>tdelta_compare:
				#print (tdelta),
				#print (" "),
				#print (tdelta_compare)
				if car_id in dict_id_2:
					if date in dict_id_2[car_id]:
						dict_id_2[car_id][date][0].append(lst[0][i])
						dict_id_2[car_id][date][1].append(lst[1][i])
						dict_id_2[car_id][date][2].append(lst[2][i])
						dict_id_2[car_id][date][3].append(lst[3][i])
					else:
						dict_id_2[car_id][date]=[[lst[0][i]],[lst[1][i]],[lst[2][i]],[lst[3][i]]]
				else:
					dict_id_2[car_id]={date:[[lst[0][i]],[lst[1][i]],[lst[2][i]],[lst[3][i]]]}

#print (dict_id_2['35']['01/06/2014'][2])
#print (dict_id_2)
#print (count)

#print (dict_id['35']['01/06/2014'])
#print (dict_id['35']['01/07/2014'])
#print (dict_id['35'].keys())
#print (dict_id.keys())

print (dict_id_2['35']['01/06/2014'])

X_bokeh_3=[]
Y_bokeh_3=[]
T_bokeh_3=[]
N_bokeh_3=[]
for i in range(len(dict_id['35']['01/06/2014'][0])):
	#print (str(x_lon[i])+","+str(y_lat[i])),
	X_bokeh_3.append(float(dict_id['35']['01/06/2014'][1][i])) #in Csv Y followed by X
	Y_bokeh_3.append(float(dict_id['35']['01/06/2014'][0][i]))
	T_bokeh_3.append(dict_id['35']['01/06/2014'][2][i])
	N_bokeh_3.append(dict_id['35']['01/06/2014'][3][i])
	#print (str(x_lon[i])+","+str(y_lat[i]))

X_bokeh_4=[]
Y_bokeh_4=[]
T_bokeh_4=[]
N_bokeh_4=[]
print (T_bokeh_4)
for i in range(len(dict_id_2['35']['01/06/2014'][0])):
	#print (str(x_lon[i])+","+str(y_lat[i])),
	X_bokeh_4.append(float(dict_id_2['35']['01/06/2014'][1][i])) #in Csv Y followed by X
	Y_bokeh_4.append(float(dict_id_2['35']['01/06/2014'][0][i]))
	T_bokeh_4.append(dict_id_2['35']['01/06/2014'][2][i])
	N_bokeh_4.append(dict_id_2['35']['01/06/2014'][3][i])

def region_of_day(time_s):
	hour=int(time_s[0:2])
	if hour>=6 and hour<12:
		return "morning"
	elif hour>=12 and hour<5:
		return "noon"
	elif hour>=5 and hour<=8:
		return "evening"
	else:
		return "night"

def get_color_2(timestamps):
	color=[]
	for i in timestamps:
		color.append("yellow")
	return color
def get_color(timestamps):
	color_lst=[]
	for time_s in timestamps:
		section=region_of_day(time_s)
		if section=="morning":
			color_lst.append("olive")
		elif section=="noon":
			color_lst.append("blue")
		elif section=="evening":
			color_lst.append("green")
		elif section=="night":
			color_lst.append("red")
		else:
			color_lst.append("navy")
		#print (time_s)
		#dt1=datetime.strptime(time_s,FMT)
		#print (dt1)
		#print (dt1.time())
		#curr_time=dt1.time()
		#if (time(6,0)<= curr_time) and (curr_time<=time(11,59)):
		#	color_lst.append("yellow")
		#elif (time(12,0)<=curr_time) and (curr_time<=time(16,30)):
		#	color_lst.append("orange")
		#elif (time(16,31)<=curr_time) and (curr_time<=time(20,0)):
		#	color_lst.append("red")
		#elif (time(20,1)<=curr_time) and (curr_time <=time(23,59)):
		#	color_lst.append("olive")
		#elif (time(0,0)<=curr_time) and (curr_time <=time(5,59)):
		#	color_lst.append("olive")
	print (len(color_lst))
	print (len(timestamps))
	print ("Here: "+color_lst[0])
	print ("THere: "+color_lst[-1])

	return color_lst

source=ColumnDataSource(data=dict(X_axis=X_bokeh_3,Y_axis=Y_bokeh_3,Time=T_bokeh_3, Full_Name=N_bokeh_3,color=get_color(T_bokeh_3)))
hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)"),("time_stamp", "@Time"),("full_name","@Full_Name")])
p = figure(plot_width=400, plot_height=400,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Route of Car 35 of Willem Vasco-Pais on 6 Jan")
source2=ColumnDataSource(data=dict(X_axis=X_bokeh_4,Y_axis=Y_bokeh_4,Time=T_bokeh_4, Full_Name=N_bokeh_4,color=get_color(T_bokeh_4)))
hover2=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)"),("time_stamp", "@Time"),("full_name","@Full_Name")])
p2 = figure(plot_width=400, plot_height=400,tools=[hover2,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Stops > 15 min by Car 35 of Willem Vasco-Pais on 6 Jan")
X_bokeh=[]
Y_bokeh=[]
for i in range(len(x_lon)):
	#print (str(x_lon[i])+","+str(y_lat[i])),
	X_bokeh.append(float(x_lon[i]))
	Y_bokeh.append(float(y_lat[i]))
	#print (str(x_lon[i])+","+str(y_lat[i]))
#output_file("remove.html")

p.line(x_lon.tolist(),(y_lat.tolist()),line_width=1,color='red')
#p.line(x_lon_2.tolist(),y_lat_2.tolist(),line_width=0.5,color='blue')
#print ("Here: "+color_lst[0])
#get_color_2(T_bokeh_3)

X_bokeh_7=[] #For Abila's streets
Y_bokeh_7=[]
for shape in list(sf2.iterShapes()):
	temp1=[]
	temp2=[]
	for ip in range(len(shape.points)):
		temp1.append(shape.points[ip][0])
		temp2.append(shape.points[ip][1])
	X_bokeh_7.append(temp1)
	Y_bokeh_7.append(temp2)

p.multi_line(X_bokeh_7,Y_bokeh_7,line_width=1,color='blue')
#p.line('X_axis','Y_axis',source=source,line_width=1,color="green")
p.circle('X_axis','Y_axis',source=source,size=1,color='color')
p2.line(x_lon.tolist(),(y_lat.tolist()),line_width=1,color='red')
p2.multi_line(X_bokeh_7,Y_bokeh_7,line_width=1,color='blue')
p2.circle('X_axis','Y_axis',source=source2,size=15,color='color',alpha=0.5)

def update_date(attrname, old, new):
	changed_value = date_select.value
	print (changed_value)
	car_value=car_select.value
	full_name=car_id_name.get(car_value,"Unknown_Name")
	#print ("sentiment")
	#print (sentiment)
	#print ("Old y")
	#print (source2.data['y'])
	#print("New y")
	#count=source.data['cur_count']
	#print (count)
	#words=return_words_for_cloud(changed_value,int(count[0]))
	p.title.text = "Route of Car "+car_value+" of "+full_name+" on "+changed_value+" Jan"
	p2.title.text = "Stops > 15 by Car "+car_value+" of "+full_name+" on "+changed_value+" Jan"
	#source.data=dict(cur_count=[str(count[0]) for i in range(len(words))],x=[i for i in range(len( words))],y=[int(i[2].replace('pt','')) for i in words],labels=[i[0] for i in words],cur_year=[changed_value for i in range(len(words))])
	X_bokeh_3=[]
	Y_bokeh_3=[]
	T_bokeh_3=[]
	N_bokeh_3=[]
	if '01/'+changed_value+'/2014' in dict_id[car_value]:
		for i in range(len(dict_id[car_value]['01/'+changed_value+'/2014'][0])):
			X_bokeh_3.append(float(dict_id[car_value]['01/'+changed_value+'/2014'][1][i])) #in Csv Y followed by X
			Y_bokeh_3.append(float(dict_id[car_value]['01/'+changed_value+'/2014'][0][i]))
			T_bokeh_3.append(dict_id[car_value]['01/'+changed_value+'/2014'][2][i])
			N_bokeh_3.append(dict_id[car_value]['01/'+changed_value+'/2014'][3][i])
	source.data=dict(X_axis=X_bokeh_3,Y_axis=Y_bokeh_3,Time=T_bokeh_3, Full_Name=N_bokeh_3,color=get_color(T_bokeh_3))

	X_bokeh_4=[]
	Y_bokeh_4=[]
	T_bokeh_4=[]
	N_bokeh_4=[]
	if '01/'+changed_value+'/2014' in dict_id_2[car_value]:
		for i in range(len(dict_id_2[car_value]['01/'+changed_value+'/2014'][0])):
			X_bokeh_3.append(float(dict_id_2[car_value]['01/'+changed_value+'/2014'][1][i])) #in Csv Y followed by X
			Y_bokeh_3.append(float(dict_id_2[car_value]['01/'+changed_value+'/2014'][0][i]))
			T_bokeh_3.append(dict_id_2[car_value]['01/'+changed_value+'/2014'][2][i])
			N_bokeh_3.append(dict_id_2[car_value]['01/'+changed_value+'/2014'][3][i])
	print (T_bokeh_4)
	source2.data=dict(X_axis=X_bokeh_4,Y_axis=Y_bokeh_4,Time=T_bokeh_4, Full_Name=N_bokeh_4,color=get_color(T_bokeh_4))

	#print (source.data)
	#print (str(sentiment[changed_value]))

def update_car(attrname, old, new):
	changed_value = car_select.value
	print (changed_value)
	full_name=car_id_name.get(changed_value,"Unknown_Name")
	print ("HERE:::")
	print (date_select.value)
	old_date=date_select.value
	print ("There...")
	p.title.text = "Route of Car "+changed_value+" of "+full_name+" on "+date_select.value+" Jan"
	p2.title.text = "Stops > 15 by Car "+changed_value+" of "+full_name+" on "+date_select.value+" Jan"
	X_bokeh_3=[]
	Y_bokeh_3=[]
	T_bokeh_3=[]
	N_bokeh_3=[]
	if '01/'+old_date+'/2014' in dict_id[changed_value]:
		for i in range(len(dict_id[changed_value]['01/'+old_date+'/2014'][0])):
			X_bokeh_3.append(float(dict_id[changed_value]['01/'+old_date+'/2014'][1][i])) #in Csv Y followed by X
			Y_bokeh_3.append(float(dict_id[changed_value]['01/'+old_date+'/2014'][0][i]))
			T_bokeh_3.append(dict_id[changed_value]['01/'+old_date+'/2014'][2][i])
			N_bokeh_3.append(dict_id[changed_value]['01/'+old_date+'/2014'][3][i])

	source.data=dict(X_axis=X_bokeh_3,Y_axis=Y_bokeh_3,Time=T_bokeh_3, Full_Name=N_bokeh_3,color=get_color(T_bokeh_3))

	X_bokeh_4=[]
	Y_bokeh_4=[]
	T_bokeh_4=[]
	N_bokeh_4=[]
	if '01/'+old_date+'/2014' in dict_id[changed_value]:
		for i in range(len(dict_id_2[changed_value]['01/'+old_date+'/2014'][0])):
			X_bokeh_4.append(float(dict_id_2[changed_value]['01/'+old_date+'/2014'][1][i])) #in Csv Y followed by X
			Y_bokeh_4.append(float(dict_id_2[changed_value]['01/'+old_date+'/2014'][0][i]))
			T_bokeh_4.append(dict_id_2[changed_value]['01/'+old_date+'/2014'][2][i])
			N_bokeh_4.append(dict_id_2[changed_value]['01/'+old_date+'/2014'][3][i])
	print (T_bokeh_4)
	source2.data=dict(X_axis=X_bokeh_4,Y_axis=Y_bokeh_4,Time=T_bokeh_4, Full_Name=N_bokeh_4,color=get_color(T_bokeh_4))

def update_name(attrname, old, new):
	changed_value = name_select.value
	print (changed_value)
	full_name=str(changed_value)
	changed_value= car_name_id.get(changed_value,"Unknown_Name")
	print ("HERE!!!")
	print (date_select.value)
	old_date=date_select.value
	print ("There&&&")
	p.title.text = "Route of Car "+changed_value+" of "+full_name+" on "+date_select.value+" Jan"
	p2.title.text = "Stops > 15 by Car "+changed_value+" of "+full_name+" on "+date_select.value+" Jan"
	X_bokeh_3=[]
	Y_bokeh_3=[]
	T_bokeh_3=[]
	N_bokeh_3=[]
	if '01/'+old_date+'/2014' in dict_id[changed_value]:
		for i in range(len(dict_id[changed_value]['01/'+old_date+'/2014'][0])):
			X_bokeh_3.append(float(dict_id[changed_value]['01/'+old_date+'/2014'][1][i])) #in Csv Y followed by X
			Y_bokeh_3.append(float(dict_id[changed_value]['01/'+old_date+'/2014'][0][i]))
			T_bokeh_3.append(dict_id[changed_value]['01/'+old_date+'/2014'][2][i])
			N_bokeh_3.append(dict_id[changed_value]['01/'+old_date+'/2014'][3][i])

	source.data=dict(X_axis=X_bokeh_3,Y_axis=Y_bokeh_3,Time=T_bokeh_3, Full_Name=N_bokeh_3,color=get_color(T_bokeh_3))

	X_bokeh_4=[]
	Y_bokeh_4=[]
	T_bokeh_4=[]
	N_bokeh_4=[]

	if '01/'+old_date+'/2014' in dict_id_2[changed_value]:
		for i in range(len(dict_id_2[changed_value]['01/'+old_date+'/2014'][0])):
			X_bokeh_4.append(float(dict_id_2[changed_value]['01/'+old_date+'/2014'][1][i])) #in Csv Y followed by X
			Y_bokeh_4.append(float(dict_id_2[changed_value]['01/'+old_date+'/2014'][0][i]))
			T_bokeh_4.append(dict_id_2[changed_value]['01/'+old_date+'/2014'][2][i])
			N_bokeh_4.append(dict_id_2[changed_value]['01/'+old_date+'/2014'][3][i])
	print (T_bokeh_4)

	source2.data=dict(X_axis=X_bokeh_4,Y_axis=Y_bokeh_4,Time=T_bokeh_4, Full_Name=N_bokeh_4,color=get_color(T_bokeh_4))	

date_choice=[]
for j in range(6,20):
	if j<10:
		date_choice.append('0'+str(j))
	else:
		date_choice.append(str(j))
date_select = Select(value='6',title='Select Date in January:',width=200,options=date_choice)

car_choice=list(dict_cars.keys())
car_select = Select(value='35',title='Select Car to Follow:',width=200,options=car_choice)

name_choice=list(car_name_id.keys())
name_select = Select(value='Willem Vasco-Pais',title='Select Employee to Follow:',width=200,options=name_choice)

time_choice=['1','5','10','15','30','60']
time_select = Select(value='15',title='Select Stop Time:',width=200,options=time_choice)

date_select.on_change('value', update_date)
car_select.on_change('value',update_car)
name_select.on_change('value',update_name)

l = layout([[row(date_select,car_select,name_select)],
	[row(p,p2)],
])


curdoc().add_root(l)
#show(p)
"""
sf2=shapefile.Reader(PATH2)
count=0
count2=0
for shape in list(sf2.iterShapes()):
	#if count>290:
	#	break
	#x_lon_2 = np.zeros((len(shape.points),1))
	#y_lat_2 = np.zeros((len(shape.points),1))
	x_lon_2 = np.zeros((len(shape.points)))
	y_lat_2 = np.zeros((len(shape.points)))
	for ip in range(len(shape.points)):
		x_lon_2[ip] = shape.points[ip][0]
		y_lat_2[ip] = shape.points[ip][1]
		count2+=1
	
	#X_bokeh_2=[]
	#Y_bokeh_2=[]
	#for i in range(len(x_lon_2)):
	#	print (str(x_lon_2[i])+","+str(y_lat_2[i])),
	#	X_bokeh_2.append(float(x_lon_2[i]))
	#	Y_bokeh_2.append(float(y_lat_2[i]))
	
	#p.line(x_lon_2.tolist(),y_lat_2.tolist(),line_width=0.5,color='blue')
	count+=1
"""


#print ("HERE: ")
#print (x_lon_2.shape)
#print (len(x_lon_2.tolist()))
#print (count)
#print (count2)

#plt.plot(x_lon,y_lat,'k')
#plt.xlim(shape_ex.bbox[0],shape_ex.bbox[2])
#plt.show()


"""
 IMPORT THE SHAPEFILE 

shp_file_base='cb_2015_us_state_20m'
dat_dir='shapefiles/'+shp_file_base +'/'
sf = shapefile.Reader(dat_dir+shp_file_base)

print 'number of shapes imported:',len(sf.shapes())
print ' '
print 'geometry attributes in each shape:'
for name in dir(sf.shape()):
    if not name.startswith('__'):
       print name

plt.figure()
ax = plt.axes()
ax.set_aspect('equal')
shape_ex = sf.shape(5)
x_lon = np.zeros((len(shape_ex.points),1))
y_lat = np.zeros((len(shape_ex.points),1))
for ip in range(len(shape_ex.points)):
    x_lon[ip] = shape_ex.points[ip][0]
    y_lat[ip] = shape_ex.points[ip][1]

plt.plot(x_lon,y_lat,'k') 

# use bbox (bounding box) to set plot limits
plt.xlim(shape_ex.bbox[0],shape_ex.bbox[2])


plt.figure()
ax = plt.axes()
ax.set_aspect('equal')
x_leftmost=10000
x_rightmost=-10000
y_bottom=-10000
y_top=10000
for shape in list(sf.iterShapes()):
    x_lon = np.zeros((len(shape.points),1))
    y_lat = np.zeros((len(shape.points),1))
    for ip in range(len(shape.points)):
        x_lon[ip] = shape.points[ip][0]
        y_lat[ip] = shape.points[ip][1]
        if x_leftmost>shape.bbox[ip][0]:
        	x_leftmost=shape.bbox[ip][0]
        if x_rightmost<shape.bbox[ip][2]:
        	x_rightmost=shape.bbox[ip][2]
		if y_bottom>shape.bbox[ip][1]:
			y_bottom=shape.bbox[ip][1]
		if y_top<shape.bbox[ip][3]:
			y_bottom=shape.bbox[ip][3]
    plt.plot(x_lon,y_lat) 
plt.xlim(-130,-60)
plt.ylim(23,50)



plt.figure()
ax = plt.axes() # add the axes
ax.set_aspect('equal')

for shape in list(sf.iterShapes()):
    npoints=len(shape.points) # total points
    nparts = len(shape.parts) # total parts

    if nparts == 1:
        x_lon = np.zeros((len(shape.points),1))
        y_lat = np.zeros((len(shape.points),1))
        for ip in range(len(shape.points)):
            x_lon[ip] = shape.points[ip][0]
            y_lat[ip] = shape.points[ip][1]
        plt.plot(x_lon,y_lat) 

    else: # loop over parts of each shape, plot separately
        for ip in range(nparts): # loop over parts, plot separately
            i0=shape.parts[ip]
            if ip < nparts-1:
               i1 = shape.parts[ip+1]-1
            else:
               i1 = npoints
            
            seg=shape.points[i0:i1+1]
            x_lon = np.zeros((len(seg),1))
            y_lat = np.zeros((len(seg),1))
            for ip in range(len(seg)):
                x_lon[ip] = seg[ip][0]
                y_lat[ip] = seg[ip][1]
            
            plt.plot(x_lon,y_lat) 

plt.xlim(-130,-60)
plt.ylim(23,50)
plt.show()
"""

"""
# Read the ShapeFile
PATH='VASTChal2014MC2-20140430/Geospatial/Kronos_Island'
#dat = shapefile.Reader("../../data/shp/Shapes/India_State.shp")
sf = shapefile.Reader(PATH)

for i in sf.iterRecords():
	print i

print sf.shapeType

#myshp = open("shapefiles/blockgroups.shp", "rb")


# Create a Unique List of States (Administrative Regions)
states = set([i[2] for i in dat.iterRecords()])

from bokeh.plotting import *
output_file("india_states.html")

hold()

TOOLS="pan,wheel_zoom,box_zoom,reset,previewsave"
figure(title="Map of India", tools=TOOLS, plot_width=900, plot_height=800)


for state_name in states:
	data = getDict(state_name, dat)
    patches(data[state_name]['lat_list'], data[state_name]['lng_list'],fill_color=colors[state_name], line_color="black")


show()
"""
