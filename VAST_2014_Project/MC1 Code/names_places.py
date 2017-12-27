from bokeh.io import output_notebook, curdoc
from bokeh.layouts import layout,column, widgetbox, row
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, Select, Slider, Label, HoverTool
from collections import defaultdict
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn, TextInput, RadioGroup
import random

import sys
import os
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.chunk import conlltags2tree, tree2conlltags
import nltk

PATH="articles"
entities=["ORGANIZATION","PERSON","LOCATION","FACILITY","GPE"]


years={}
def read_data(): #years['1996']=['f1.txt','f2.txt']
	#os.path.join(PATH,"")
	count=0
	for f in os.listdir(PATH):
		if f=='696.txt':
			continue
		with open(os.path.join(PATH,f),'r',encoding = "latin-1") as doc:
			data=doc.read()
			match=re.search(r'(\d+/\d+/\d+)',data)
			match2=re.search(r'(\d+\s*(:?January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',data)
			try:
				data=match.group(1)
				year=data[:4]
				#print (f,year)
				if year in years:
					years[year].append(f)
				else:
					years[year]=[f]
			except AttributeError:
				#print("Here:"+f)
				match2=re.search(r'(\d+\s*(:?January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',data)
				data=match2.group(1)
				year=data[-4:]
				print (f,year)
				if year in years:
					years[year].append(f)
				else:
					years[year]=[f]
			except AttributeError:
				count+=1
			#print date
	#print ("THis: "+str(count)) #No. of Files with no date of form xxxx/xx/xx
	return years

years=read_data()
names_year_dict={}
for year in years:
	dic={}
	for j in entities:
		dic[j]=set()
		#dic[j]=[]
	for f in years[year]:
		fp=open(PATH+'/'+str(f),'r',encoding = "latin-1")
		for line in fp:
			sentence=line.replace('\n','')
			ne_tree = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence)))
			for chunk in ne_tree:
				try:
					dic[chunk.label()].add(' '.join(c[0] for c in chunk.leaves()))
					#dic[chunk.label()].append(' '.join(c[0] for c in chunk.leaves()))
				except:
					continue
	names_year_dict[year]=dic

"""
fp=open("0.txt",'r')
for line in fp:
	sentence=line.replace('\n','')
	ne_tree = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence)))
	for chunk in ne_tree:
		try:
			dic[chunk.label()].append(' '.join(c[0] for c in chunk.leaves()))
		except:
			continue

"""
for year in names_year_dict:
	print (year)
	print(names_year_dict[year])
	for key in names_year_dict[year]:
			print (key)
			print (names_year_dict[year][key])
			print ("")

source1=ColumnDataSource(data=dict(names=list(names_year_dict['2013']['PERSON'])))
source2=ColumnDataSource(data=dict(orgs=list(names_year_dict['2013']['ORGANIZATION'])))
source3=ColumnDataSource(data=dict(places=list(names_year_dict['2013']['GPE'])))

#source=ColumnDataSource(data=dict(text=ans,year=years_of_msgs,files=files))
#columns=[TableColumn(field="text",title="sentences"),TableColumn(field="year",title="Year"),TableColumn(field="files",title="file_names"),]
#output_file("remove.html")
#data_table=DataTable(source=source,columns=columns,width=1000,height=300)
columns1=[TableColumn(field="names",title="Person Names"),]
data_table1=DataTable(source=source1,columns=columns1,width=1000,height=250)

columns2=[TableColumn(field="orgs",title="Organizations"),]
data_table2=DataTable(source=source2,columns=columns2,width=1000,height=200)

columns3=[TableColumn(field="places",title="Places"),]
data_table3=DataTable(source=source3,columns=columns3,width=1000,height=200)

def update_year(attrname, old, new):
	changed_value = year_select.value
	print (changed_value)
	source1.data=dict(names=list(names_year_dict[changed_value]['PERSON']))
	source2.data=dict(orgs=list(names_year_dict[changed_value]['ORGANIZATION']))
	source3.data=dict(places=list(names_year_dict[changed_value]['GPE']))

year_choice=list(years.keys())
year_select = Select(value='2013',title='Select Year:',width=200,options=year_choice)
year_select.on_change('value', update_year)

inputs=widgetbox(year_select,data_table1,data_table3,data_table2)
curdoc().add_root(row(inputs))
#output_file("remove.html")
#show(widgetbox(data_table1,data_table2,data_table3))