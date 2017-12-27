from __future__ import division

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

PLOT_WIDTH=1000
PLOT_HEIGHT=1000
PATH="articles" #CHANGE THIS ACCORDINGLY!!! word_cloud_2.py file and articles folder should be on the same level


#PATH="testing_remove"
def give_bow(lst):
	stop_words1=set(stopwords.words('english'))
	punc=['.',',','!','?',"'","''","``","-","*","%"]
	bow = defaultdict(float)
	tot_count=0.0
	for f in lst:
		with open(os.path.join(PATH,f),'r',encoding="latin-1") as doc:
			#fp=open(filename,'r')
			data=doc.read()
			tokens=word_tokenize(data)
			lowered_tokens = map(lambda t: t.lower(), tokens)
			all_tokens = [w for w in lowered_tokens if not w in stop_words1]
			all_tokens = [w for w in all_tokens if not w in punc]
			tot_count+=len(all_tokens)
			for token in all_tokens:
				bow[token] += 1.0
	return (dict(bow),tot_count)

def top_words(bow,num):
	tup_wc=[]
	for i in bow.items():
		tup_wc.append(i)
	tup_wc.sort(key=lambda tup: tup[1],reverse=True)
	return tup_wc[0:num]

words_test = [
	('hello', (10, 100), '12pt', 0.8),
	('how', (20, 10), '14pt', 0.8),
	('are', (100, 20), '16pt', 0.8),
	('you', (50, 30), '18pt', 0.8),
	('???', (100, 100), '18pt', 0.8),
]

years={}
def read_data():
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
				print (f,year)
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
	print (count) #No. of Files with no date of form xxxx/xx/xx
	return years

#print ("HERE GOES!!:")
#print give_bow(['test1.txt','test2.txt'])

dic=read_data()

def sent_seed(word,years,context=3,flag=0): #flag==0 dont ignore case
	ans=[]
	files=[]
	years_of_msgs=[]
	punc=['.',',','!','?',"'","''","``","-","*","%",'\n']
	for year in years:
		for f in years[year]:
			fp=open(PATH+'/'+str(f),'r',encoding = "latin-1")
			for line in fp:
				if flag==1:
					word=word.lower()
					line=line.lower()
				#line=line.lower()
				if " "+word+" " in line:
					for k in punc:
						line=line.replace(k,'')
					words=line.split(' ')
					try:
						idx=words.index(word)
					except ValueError:
						print (line)
						return
					txt=""
					for i in range(-1*context,context+1):
						if i+idx<0 or i+idx>=len(words):
							continue
						else:
							txt+=words[i+idx]+" "
					ans.append(txt)
					years_of_msgs.append(year)
					files.append(f)
	return (ans,years_of_msgs,files)

ans,years_of_msgs,files=sent_seed("leader",dic)
print (ans)
print (len(ans))
x=["Trying out","new things"]
source=ColumnDataSource(data=dict(text=ans,year=years_of_msgs,files=files))

columns=[TableColumn(field="text",title="sentences"),TableColumn(field="year",title="Year"),TableColumn(field="files",title="file_names"),]
#output_file("remove.html")
data_table=DataTable(source=source,columns=columns,width=1000,height=300)
#show(widgetbox(data_table))


#_choice=[str(i) for i in range(1992,2015) if i not in [2006,2008]]
#year_select = Select(value='2001',title='Select year:',width=200,options=year_choice)

text_input=TextInput(value="leader",title="Seed Word: ")
context_len=Slider(title="context_len", value=3, start=0, end=7, step=1)
radio_group=RadioGroup(labels=["Consider Case","Ignore Case"],active=0)


def update_seed(attrname, old, new):
	new_seed=text_input.value
	print (new_seed)
	ans,years_of_msgs,files=sent_seed(new_seed,dic,context_len.value,radio_group.active)
	source.data=dict(text=ans,year=years_of_msgs,files=files)

def update_context_len(attrname,old,new):
	new_len=context_len.value
	print (new_len)
	ans,years_of_msgs,files=sent_seed(text_input.value,dic,context=new_len,flag=radio_group.active)
	source.data=dict(text=ans,year=years_of_msgs,files=files)

def update_case(new):
	flag=str(new)
	print (flag)
	ans,years_of_msgs,files=sent_seed(text_input.value,dic,context_len.value,flag=int(flag))
	source.data=dict(text=ans,year=years_of_msgs,files=files)

#year_select.on_change('value', update_year)
#count_select.on_change('value', update_count)
#curdoc().add_root(column(year_select, count_select,p))
text_input.on_change('value', update_seed)
context_len.on_change('value',update_context_len)
radio_group.on_click(update_case)
inputs = widgetbox(text_input,context_len,radio_group,data_table)
curdoc().add_root(row(inputs))


#print (dic.keys())
#bow,tot_count=give_bow("data.txt")
#top_words_list=top_words(bow,5)
def return_words_for_cloud(year,num):
	words=[]
	bow,tot_count=give_bow(dic[year])
	top_words_list=top_words(bow,int(num))
	for i in top_words_list:
		loc=(random.randrange(0,PLOT_WIDTH-100),random.randrange(0,PLOT_HEIGHT-100))
		words.append((i[0],loc,str(int(1000*i[1]/tot_count))+'pt',0.8))
	print (words)
	return words



"""
fig = figure(plot_width=PLOT_WIDTH,plot_height=PLOT_HEIGHT, title='word_cloud')
words=return_words_for_cloud('2001',10)
for word, loc, size, alpha in words:
	w = Label(x=loc[0], y=loc[1], x_units='screen', y_units='screen',text=word, render_mode='css', text_alpha=alpha, text_font_size=size,text_color=(255,0,0))
	fig.add_layout(w)

#print bow("data.txt")
words=return_words_for_cloud('2001',10)
source=ColumnDataSource(data=dict(x=[i for i in range(len( words))],y=[int(i[2].replace('pt','')) for i in words],labels=[i[0] for i in words],cur_year=['2001' for i in range(len( words))],cur_count=['10' for i in range(len( words))]))
hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)"),("word", "@labels")])
p=figure(plot_width=PLOT_WIDTH,plot_height=PLOT_HEIGHT,tools=[hover,'pan','wheel_zoom','box_zoom','box_select'],title="Top 10 words in year 2001")
#source=ColumnDataSource(data=dict(x=[i[0] for i in words],y=[i[2] for i in words]))

p.vbar(x='x',top='y',width=0.5,color="firebrick",source=source)

#output_file("remove.html")
#show(p)

year_choice=[str(i) for i in range(1992,2015) if i not in [2006,2008]]
year_select = Select(value='2001',title='Select year:',width=200,options=year_choice)

count_choice=['5','7','10','15']
count_select = Select(value='10',title='Select count:',width=200,options=count_choice)

def update_year(attrname, old, new):
	changed_value = year_select.value
	print (changed_value)
	count=source.data['cur_count']
	print (count)
	words=return_words_for_cloud(changed_value,int(count[0]))
	p.title.text = "Top 10 words in year "+changed_value
	source.data=dict(cur_count=[str(count[0]) for i in range(len(words))],x=[i for i in range(len( words))],y=[int(i[2].replace('pt','')) for i in words],labels=[i[0] for i in words],cur_year=[changed_value for i in range(len(words))])

def update_count(attrname, old, new):
	changed_count = count_select.value
	print (changed_count)
	years_temp=source.data['cur_year']
	words=return_words_for_cloud(years_temp[0],int(changed_count))
	source.data=dict(cur_year=[years_temp[0] for i in range(len(words))],x=[i for i in range(len( words))],y=[int(i[2].replace('pt','')) for i in words],labels=[i[0] for i in words],cur_count=[changed_count for i in range(len(words))])
	#words=return_words_for_cloud(changed_value)
	#p.title.text = "Top 10 words in year "+changed_value
	#source.data=dict(x=[i for i in range(len( words))],y=[int(i[2].replace('pt','')) for i in words],labels=[i[0] for i in words])


year_select.on_change('value', update_year)
count_select.on_change('value', update_count)
curdoc().add_root(column(year_select, count_select,p))
"""