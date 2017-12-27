import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show, gridplot
from bokeh.models import HoverTool,ColumnDataSource,Range1d,Select, Slider, Label
from bokeh.layouts import layout,column,widgetbox, row
from bokeh.io import output_notebook, curdoc

#fp1=open('cc_data.csv','r')
#fp2=open('loyalty_data.csv','r')

fp1=open('cc_data.csv','r',encoding='latin-1')
fp2=open('loyalty_data.csv','r',encoding='latin-1')

c_dict_places={}
c_dict_names={}

c_set_places=set()
c_set_names=set()

l_dict_places={}
l_dict_names={}

l_set_places=set()
l_set_names=set()

set_places=set()
set_names=set()

#"Brew've Been Served":{'01/06/2014':[[11.34,8.33],[Full_Name, Full_Name]]}
#"Full_Name":{'01/06/2014':[["11.34","13.26"],["Brew've Been Served","Hippokampos"]]}

for line in fp1:
	if 'id' in line:
		continue
	text=line.split(',')
	c_set_names.add(text[3]+" "+text[4])
	c_set_places.add(text[1])
	date_time=text[0].split(' ')
	full_name=text[3]+' '+text[4]
	full_name=full_name.replace('\n','')
	full_name=full_name.replace('\r','')
	set_places.add(text[1])
	set_names.add(full_name)
	if text[1] in c_dict_places:
		if date_time[0] in c_dict_places[text[1]]:
			c_dict_places[text[1]][date_time[0]][0].append(text[2])
			c_dict_places[text[1]][date_time[0]][1].append(full_name)
		else:
			c_dict_places[text[1]][date_time[0]]=[[text[2]],[full_name]]			
	else:
		c_dict_places[text[1]]={date_time[0]:[[text[2]],[full_name]]}

	if full_name in c_dict_names:
		if date_time[0] in c_dict_names[full_name]:
			c_dict_names[full_name][date_time[0]][0].append(text[2])
			c_dict_names[full_name][date_time[0]][1].append(text[1])
		else:
			c_dict_names[full_name][date_time[0]]=[[text[2]],[text[1]]]			
	else:
		c_dict_names[full_name]={date_time[0]:[[text[2]],[text[1]]]}

for line in fp2:
	if 'id' in line:
		continue
	text=line.split(',')
	l_set_names.add(text[3]+" "+text[4])
	l_set_places.add(text[1])
	date_time=text[0].split(' ')
	full_name=text[3]+' '+text[4]
	full_name=full_name.replace('\n','')
	full_name=full_name.replace('\r','')
	set_places.add(text[1])
	set_names.add(full_name)
	if text[1] in l_dict_places:
		if date_time[0] in l_dict_places[text[1]]:
			l_dict_places[text[1]][date_time[0]][0].append(text[2])
			l_dict_places[text[1]][date_time[0]][1].append(full_name)
		else:
			l_dict_places[text[1]][date_time[0]]=[[text[2]],[full_name]]			
	else:
		l_dict_places[text[1]]={date_time[0]:[[text[2]],[full_name]]}
	
	if full_name in l_dict_names:
		if date_time[0] in l_dict_names[full_name]:
			l_dict_names[full_name][date_time[0]][0].append(text[2])
			l_dict_names[full_name][date_time[0]][1].append(text[1])
		else:
			l_dict_names[full_name][date_time[0]]=[[text[2]],[text[1]]]			
	else:
		l_dict_names[full_name]={date_time[0]:[[text[2]],[text[1]]]}

print (c_dict_places["Brew've Been Served"])
#print (l_dict_places)

output_file("remove.html")

def calc_sum(lst,opn="sum"):
	ans=0.0
	if opn=="sum":
		for i in lst:
			ans+=float(i)
	return ans

def name_place_statistics(place,dict_places_1,dict_places_2,flag="length"):
	dates=['1/'+str(j)+'/2014' for j in range(6,20)]
	count_place=[]
	if place not in dict_places_1:
		source1=ColumnDataSource(data=dict(X_axis=[str(j) for j in range(6,20)],Y_axis=[0 for i in range(len(dates))]))
	else:
		date_dict=dict_places_1[place]
		for i in dates:
			count_list=(date_dict.get(i,[[],[]]))
			if flag=="length":
				count_place.append(len(count_list[0]))
			elif flag=="sum":
				count_place.append(calc_sum(count_list[0],"sum"))
		source1=ColumnDataSource(data=dict(X_axis=[str(j) for j in range(6,20)],Y_axis=count_place))	
	
	if len(count_place)>0:
		count_place_all=list(count_place)
	else:
		count_place_all=[0]*len(dates)

	if place not in dict_places_2:
		source2=ColumnDataSource(data=dict(X_axis=[str(j) for j in range(6,20)],Y_axis=[0 for i in range(len(dates))]))
	else:
		date_dict=dict_places_2[place]
		count_place=[]
		for i in range(len(dates)):
			count_list=(date_dict.get(dates[i],[[],[]]))

			if flag=="length":
				count_place.append(len(count_list[0]))
				count_place_all[i]+=(len(count_list[0]))
			elif flag=="sum":
				count_place.append(calc_sum(count_list[0],"sum"))
				count_place_all[i]+=(calc_sum(count_list[0],"sum"))
			#count_place_all[i]+=len(count_list[0])
		source2=ColumnDataSource(data=dict(X_axis=[str(j) for j in range(6,20)],Y_axis=count_place))	
	source3=ColumnDataSource(data=dict(X_axis=[str(j) for j in range(6,20)],Y_axis=count_place_all))
	return (source1,source2,source3)	

#source=ColumnDataSource(data=dict(X_axis=X_bokeh_3,Y_axis=Y_bokeh_3,Time=T_bokeh_3, Full_Name=N_bokeh_3))
source1,source2,source3=name_place_statistics("Brew've Been Served",c_dict_places,l_dict_places)
source4,source5,source6=name_place_statistics("Brew've Been Served",c_dict_places,l_dict_places,flag="sum")
#source1,source2,source3=name_place_statistics("Edvard Vann",c_dict_names,l_dict_names)
hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
p1 = figure(plot_width=400, plot_height=400,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Brew've Been Served shop Number of Users")
p2 = figure(plot_width=400, plot_height=400,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Brew've Been Served shop total Revenue")

p1.line("X_axis","Y_axis",source=source1,line_width=1,color='red',legend="Credit_card_Users")
p1.line("X_axis","Y_axis",source=source2,line_width=1,color='blue',legend="Loyalty_card_Users")
p1.line("X_axis","Y_axis",source=source3,line_width=1,color='green',legend="Total_Users")

p2.line("X_axis","Y_axis",source=source4,line_width=1,color='red',legend="Credit_card_Users")
p2.line("X_axis","Y_axis",source=source5,line_width=1,color='blue',legend="Loyalty_card_Users")
p2.line("X_axis","Y_axis",source=source6,line_width=1,color='green',legend="Total_Users")

source7,source8,source9=name_place_statistics("Edvard Vann",c_dict_names,l_dict_names)
source10,source11,source12=name_place_statistics("Edvard Vann",c_dict_names,l_dict_names,flag="sum")
#source1,source2,source3=name_place_statistics("Edvard Vann",c_dict_names,l_dict_names)
hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
p3 = figure(plot_width=400, plot_height=400,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Edvard Vann Card Usage")
p4 = figure(plot_width=400, plot_height=400,tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'],title="Edvard Vann Card Spending")

p3.line("X_axis","Y_axis",source=source7,line_width=1,color='red',legend="Number_Credit_Card")
p3.line("X_axis","Y_axis",source=source8,line_width=1,color='blue',legend="Number_Loyalty_card")
p3.line("X_axis","Y_axis",source=source9,line_width=1,color='green',legend="Total_Card_Usage")

p4.line("X_axis","Y_axis",source=source10,line_width=1,color='red',legend="Amount_Credit_Card")
p4.line("X_axis","Y_axis",source=source11,line_width=1,color='blue',legend="Amount_Loyalty_Card")
p4.line("X_axis","Y_axis",source=source12,line_width=1,color='green',legend="Total_Amount")

#source2=name_place_statistics("Edvard Vann",c_dict_names)
#hover2=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
#p.line("X_axis","Y_axis",source=source2,line_width=1,color='blue')


p1.legend.click_policy="hide"
p2.legend.click_policy="hide"
p3.legend.click_policy="hide"
p4.legend.click_policy="hide"

#p = gridplot([[p1, p2], [p3, p4]])
#show(p)


def update_card(attrname, old, new):
	changed_value = card_select.value
	print (changed_value)
	if changed_value=='Credit_card':
		temp_source=name_place_statistics(place_select.value,c_dict_places)
	elif changed_value=='Loyalty_card':
		temp_source=name_place_statistics(place_select.value,l_dict_places)
	
	source.data=temp_source.data

def update_name(attrname, old, new):
	changed_value = name_select.value
	print (changed_value)
	p3.title.text = changed_value+" Card Usage"
	p4.title.text = changed_value+" Card spending"
	t_source1,t_source2,t_source3=name_place_statistics(changed_value,c_dict_names,l_dict_names)
	t_source4,t_source5,t_source6=name_place_statistics(changed_value,c_dict_names,l_dict_names,flag="sum")
	source7.data=t_source1.data
	source8.data=t_source2.data
	source9.data=t_source3.data
	source10.data=t_source4.data
	source11.data=t_source5.data
	source12.data=t_source6.data

def update_place(attrname, old, new):
	changed_value = place_select.value
	print (changed_value)
	p1.title.text = changed_value+" Served shop Number of Users"
	p2.title.text = changed_value+" Served shop total Revenue"
	t_source1,t_source2,t_source3=name_place_statistics(changed_value,c_dict_places,l_dict_places)
	t_source4,t_source5,t_source6=name_place_statistics(changed_value,c_dict_places,l_dict_places,flag="sum")
	source1.data=t_source1.data
	source2.data=t_source2.data
	source3.data=t_source3.data
	source4.data=t_source4.data
	source5.data=t_source5.data
	source6.data=t_source6.data

#card_choice=["Credit_card","Loyalty_card"]
#card_select = Select(value='Credit_card',title='Select Card to Follow:',width=200,options=card_choice)

place_choice=list(set_places)
place_select = Select(value="Brew've Been Served",title='Select Shop for statistics:',width=200,options=place_choice)

name_choice=list(set_names)
name_select = Select(value="Edvard Vann",title='Select Name for statistics:',width=200,options=name_choice)

#card_select.on_change('value',update_card)
name_select.on_change('value',update_name)
place_select.on_change('value',update_place)


l = layout([[row(place_select)],
	[row(p1,p2)],
	[row(name_select)],
	[row(p3,p4)],
])


curdoc().add_root(l)
show(p)