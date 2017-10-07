import pandas as pd
from scipy.stats import pearsonr
import numpy as np
from sklearn.metrics import matthews_corrcoef
from sklearn.cluster import KMeans, DBSCAN
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool,ColumnDataSource, FactorRange
from bokeh.layouts import gridplot,widgetbox
from bokeh.models.widgets import Panel,Tabs,Dropdown,DataTable, TableColumn
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 


def get_color(y_pred): #Returns a list of colors,
	colors=[]
	for i in range(y_pred[0].shape[0]):
		num=y_pred[0][i]
		if num==-1:
			colors.append("cyan")
		elif num==0:
			colors.append("red")
		elif num==1:
			colors.append("blue")
		elif num==2:
			colors.append("green")
		elif num==3:
			colors.append("yellow")
		elif num==4:
			colors.append("purple")
		elif num==5:
			colors.append("orange")
		elif num==6:
			colors.append("navy")
	return colors

food_df=pd.read_csv('nutrition_raw_anonymized_data.csv',na_values=[''])

records=list(food_df.index)
features=list(food_df.columns)

missing_row_index=[]
for i in features:
	row=food_df.loc[food_df[i].isnull()]
	if row.empty:
		continue
	else:
		missing_row_index.append(list(row.index))


print (missing_row_index) #No Missing Values! :)

features_to_numeric=['cancer','diabetes','heart_disease',"ever_smoked","currently_smoke",'smoke_often',"smoke_rarely","never_smoked","quit_smoking",'left_hand','cat','dog','Dems','Jewish','atheist'] #Initial Categorical Values being converted to binary data

for j in features_to_numeric:
	idx=features.index(j)
	for i in range((food_df[j].shape[0])):
		if food_df.iloc[i][j]=='Yes':
			food_df.iloc[i,idx]=1.0
		else:
			food_df.iloc[i,idx]=0.0
	food_df[j]=food_df[j].apply(pd.to_numeric)

idx=features.index('belly')
for i in range((food_df['belly'].shape[0])):
	if food_df.iloc[i]['belly']=='Outie':
		food_df.iloc[i,idx]=1.0
	else:
		food_df.iloc[i,idx]=0.0
food_df['belly']=food_df['belly'].apply(pd.to_numeric)	

features_to_quantize=[]
skew_features=set(['VITAMINDQUAN','CALCIUMQUAN','WATERQUAN','VITAMINEQUAN']) #Their numerical value is too high as compared to other food_items. So removing them
for i in features:
	if 'QUAN' in i and i not in skew_features:
		features_to_quantize.append(i)

for j in features_to_quantize:
	idx=features.index(j)
	for i in range((food_df[j].shape[0])):
		food_df.iloc[i,idx]=food_df.iloc[i,idx]*food_df.iloc[i,idx-1]

def has_pet(x):
	if x>=1:
		return 1.0
	elif x==0:
		return 0.0


food_df=food_df.assign(temp_pet = lambda x: (x.cat+x.dog))
food_df["has_pet"] = food_df["temp_pet"].apply(has_pet) #Creating has_pet feature from cat and dog fields


def calc_support(lst,food_df): #How frequently the itemset lst appears
	features=list(food_df.columns)
	indices=[]
	tr_count=0.0
	to_count=0.0
	for i in lst:
		indices.append(features.index(i))
	#print (indices)
	for i in range(food_df['cancer'].shape[0]):
		flag=1
		for j in lst:
			#print str(i)+" : "+str(j)
			if food_df.iloc[i][j]==0:
				flag=-1
		if flag==1:
			tr_count+=1
		to_count+=1
	return tr_count/to_count
	#return (tr_count,to_count)

def calc_confidence(left_lst,right_lst,food_df): #How confident we are with the rule
	denom=calc_support(left_lst,food_df)
	nume=calc_support(left_lst+right_lst,food_df)
	return nume/denom

def calc_lift(left_lst,right_lst,food_df): #Lift=1 implies independence. More in value the better.
	denom=calc_support(left_lst,food_df) * calc_support(right_lst,food_df) 
	nume=calc_support(left_lst+right_lst,food_df)
	return nume/denom

def calc_conviction(left_lst,right_lst,food_df): #How much better than random chance is it. (For % subtract 1)
	nume=1.0-calc_support(right_lst,food_df)
	denom=1.0-calc_confidence(left_lst,right_lst,food_df)
	return nume/denom

#print matthews_corrcoef(food_df['belly'], food_df['diabetes']) #Wanted to use, but didn't


print ("Smoke_Often -> cancer. Confidence: "+str(calc_confidence(['smoke_often'],['cancer'],food_df))+", Lift: "+str(calc_lift(['smoke_often'],['cancer'],food_df)))
print ("cancer -> quit_smoking. Confidence: "+str(calc_confidence(['cancer'],['quit_smoking'],food_df))+", Lift: "+str(calc_lift(['cancer'],['quit_smoking'],food_df)))
print ("cancer -> quit_smoking. Confidence: "+str(calc_confidence(['cancer'],['quit_smoking'],food_df))+", Lift: "+str(calc_lift(['cancer'],['quit_smoking'],food_df)))
#print "has_pet -> smoke_rarely. Confidence: "+str(calc_confidence(['has_pet'],['smoke_rarely'],food_df))+", Lift: "+str(calc_lift(['has_pet'],['smoke_rarely'],food_df))) #0.05, 1.02
print ("has_pet -> quit_smoking. Confidence: "+str(calc_confidence(['has_pet'],['quit_smoking'],food_df))+", Lift: "+str(calc_lift(['has_pet'],['quit_smoking'],food_df))) #0.286, 1.4
#print "Innie belly -> cancer. Confidence: "+str(calc_confidence(['belly'],['cancer'],food_df))+", Lift: "+str(calc_lift(['belly'],['cancer'],food_df))) #0.551, 1.06
#print "Innie belly -> heart_disease. Confidence: "+str(calc_confidence(['belly'],['heart_disease'],food_df))+", Lift: "+str(calc_lift(['belly'],['heart_disease'],food_df))) #0.367, 0.99
#print "Innie belly -> diabetes. Confidence: "+str(calc_confidence(['belly'],['diabetes'],food_df))+", Lift: "+str(calc_lift(['belly'],['diabetes'],food_df))) #0.245, 0.88
print ("Outie belly -> cancer. Confidence: "+str(calc_confidence(['belly'],['cancer'],food_df))+", Lift: "+str(calc_lift(['belly'],['cancer'],food_df))) #0.52, 0.386
print ("Outie belly -> heart_disease. Confidence: "+str(calc_confidence(['belly'],['heart_disease'],food_df))+", Lift: "+str(calc_lift(['belly'],['heart_disease'],food_df))) #0.4, 1.08
print ("Outie belly -> diabetes. Confidence: "+str(calc_confidence(['belly'],['diabetes'],food_df))+", Lift: "+str(calc_lift(['belly'],['diabetes'],food_df))) #0.6, 2.16
print ("Jewish -> Dems. Confidence: "+str(calc_confidence(['Jewish'],['Dems'],food_df))+", Lift: "+str(calc_lift(['Jewish'],['Dems'],food_df))) #1.0, 1.32
print ("Dems -> Jewish. Confidence: "+str(calc_confidence(['Dems'],['Jewish'],food_df))+", Lift: "+str(calc_lift(['Dems'],['Jewish'],food_df))) #0.04, 1.31
print ("atheist -> Dems. Confidence: "+str(calc_confidence(['atheist'],['Dems'],food_df))+", Lift: "+str(calc_lift(['atheist'],['Dems'],food_df))) #0.8, 1.06
print ("Dems -> atheist. Confidence: "+str(calc_confidence(['Dems'],['atheist'],food_df))+", Lift: "+str(calc_lift(['Dems'],['atheist'],food_df))) #0.7, 1.06
print ("has_pet -> atheist. Confidence: "+str(calc_confidence(['has_pet'],['atheist'],food_df))+", Lift: "+str(calc_lift(['has_pet'],['atheist'],food_df))) #0.714, 1.07
print ("atheist -> has_pet. Confidence: "+str(calc_confidence(['atheist'],['has_pet'],food_df))+", Lift: "+str(calc_lift(['atheist'],['has_pet'],food_df))) #0.694, 1.07
#print calc_confidence(['cancer'],['smoke_often'],food_df)


def correlate(lft_lst,ryt_lst):
	temp_dict={}
	for i in lft_lst:
		temp_dict[i]=[]
		for j in ryt_lst:
			yes_c=0.0
			no_c=0.0
			one_c=0.0
			for (k1,k2) in zip(food_df[i],food_df[j]):
				#print str(j)+" : "+str(i)
				if k1==1  and k2==1:
					yes_c+=1
				elif k1==0 and k2==0:
					no_c+=1
				else:
					one_c+=1
			temp_dict[i].append([j,yes_c,no_c,one_c])
	return temp_dict

def return_col(dict_oils,field,idx):
	temp_lst=[]
	for i in dict_oils[field]:
		temp_lst.append(i[idx])
	return temp_lst

ryt_lst=["COOKINGFATPAMORNONE","COOKINGFATBUTTER","COOKINGFATHALF","COOKINGFATSTICKMARG","COOKINGFATSOFTTUBMARG","COOKINGFATLOWFATMARG","COOKINGFATOLIVE","COOKINGFATCANOLA","COOKINGFATCORN","COOKINGFATPEANUT","COOKINGFATLARD","COOKINGFATCRISCO","COOKINGFATOTHER"]
dict_oils=correlate(["cancer","diabetes","heart_disease"],ryt_lst)

source_c = ColumnDataSource(data=dict(x=[(i+1) for i in range(len(ryt_lst))], y_c = return_col(dict_oils,'cancer',1),y_nc=return_col(dict_oils,'cancer',2),oil=ryt_lst))
hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)"),("Oil", "@oil")])

p_oil_c = figure(plot_width=1000, plot_height=800,tools=[hover,'pan','wheel_zoom','box_zoom','box_select'],title="Oil trends of people with Cancer (Hover for actual values)",x_axis_label="Oil type",y_axis_label="No. of people",x_range=[i.replace('COOKING','') for i in ryt_lst])
p_oil_c.line(x ='x', y ='y_c',source=source_c,line_width=2,legend='Cancer',line_color="red")
p_oil_c.line(x ='x', y ='y_nc',source=source_c,line_width=2,legend='No_Cancer',line_color="green")


source_d = ColumnDataSource(data=dict(x=[(i+1) for i in range(len(ryt_lst))], y_d = return_col(dict_oils,'diabetes',1),y_nd=return_col(dict_oils,'diabetes',2),oil=ryt_lst))
hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)"),("Oil", "@oil")])

p_oil_d = figure(plot_width=1000, plot_height=800,tools=[hover,'pan','wheel_zoom','box_zoom','box_select'],title="Oil trends of people with Diabetes (Hover for actual values)",x_axis_label="Oil type",y_axis_label="No. of people",x_range=[i.replace('COOKING','') for i in ryt_lst])
p_oil_d.line(x ='x', y ='y_d',source=source_d,line_width=2,legend='Diabetes',line_color="red")
p_oil_d.line(x ='x', y ='y_nd',source=source_d,line_width=2,legend='No_Diabetes',line_color="green")

source_h = ColumnDataSource(data=dict(x=[(i+1) for i in range(len(ryt_lst))], y_h = return_col(dict_oils,'heart_disease',1),y_nh=return_col(dict_oils,'heart_disease',2),oil=ryt_lst))
hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)"),("Oil", "@oil")])

p_oil_h = figure(plot_width=1000, plot_height=800,tools=[hover,'pan','wheel_zoom','box_zoom','box_select'],title="Oil trends of people with Heart Disease (Hover for actual values)",x_axis_label="Oil type",y_axis_label="No. of people",x_range=[i.replace('COOKING','') for i in ryt_lst])
p_oil_h.line(x ='x', y ='y_h',source=source_h,line_width=2,legend='heart_disease',line_color="red")
p_oil_h.line(x ='x', y ='y_nh',source=source_h,line_width=2,legend='No_heart_disease',line_color="green")

tab1=Panel(child=p_oil_c,title="Cancer patient Oil trends")
tab2=Panel(child=p_oil_d,title="Diabetes patient Oil trends")
tab3=Panel(child=p_oil_h,title="Heart patient oil trends")

output_file("Oil Trends of people.html")
tabs=Tabs(tabs=[tab1,tab2,tab3])
show(tabs)


dual_mean={}
cancer_mean=[]
no_cancer_mean=[]
for k in features_to_quantize:
	cancer_s=0.0
	cancer_c=0.0
	non_cancer_s=0.0
	non_cancer_c=0.0
	for (i,j) in zip(food_df['cancer'],food_df[k]):
		if i==1:
			cancer_s+=j
			cancer_c+=1
		elif i==0:
			non_cancer_s+=j
			non_cancer_c+=1
	mean_cancer=cancer_s/cancer_c
	mean_non_cancer=non_cancer_s/non_cancer_c
	#if mean_cancer<400 and mean_non_cancer<400:
	dual_mean[k]=[mean_cancer,mean_non_cancer]
	cancer_mean.append((k,mean_cancer))
	no_cancer_mean.append((k,mean_non_cancer))
	#else:
	#skew_features.add(k)

#print dual_mean['RIBSQUAN']
cancer_top_bottom=sorted(cancer_mean,key=lambda x: x[1],reverse=True)
no_cancer_top_bottom=sorted(no_cancer_mean,key=lambda x: x[1],reverse=True)
cancer_top_bottom=cancer_top_bottom[:7]+cancer_top_bottom[-7:]
no_cancer_top_bottom=no_cancer_top_bottom[:7]+no_cancer_top_bottom[-7:]

#print cancer_top_bottom
#print no_cancer_top_bottom

cancer_food=set([])
for i in cancer_top_bottom:
	cancer_food.add(i[0])
for i in no_cancer_top_bottom:
	cancer_food.add(i[0])


types=['cancer','no_cancer']
food_items=[]
data={'food_items':[],'non_cancer_values':[],'cancer_values':[]}
for key in cancer_food:
	key=key.replace('QUAN','')
	data['food_items'].append(key)
	data['cancer_values'].append(dual_mean[key+'QUAN'][0])
	data['non_cancer_values'].append(dual_mean[key+'QUAN'][1])

x = [ (food_item.replace('QUAN',''), category) for food_item in cancer_food for category in types ]
counts = sum(zip(data['cancer_values'], data['non_cancer_values']), ()) 
get_color_vals=[]
for i in range(len(counts)):
	if i%2==0:
		get_color_vals.append(["blue"])
	else:
		get_color_vals.append(["green"])	

source = ColumnDataSource(data=dict(x=x, counts=counts))
hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
p = figure(x_range=FactorRange(*x), plot_height=1000, plot_width=1600,title="Mean food quantity for cancer patients",tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'])
p.vbar(x='x', top='counts', width=0.9, source=source,color=get_color_vals)
#p.x_range.range_padding = 0.1
p.xaxis.major_label_orientation = 1
p.xgrid.grid_line_color = None

tab1=Panel(child=p,title="Cancer patient Food trends")

dual_mean={}
heart_mean=[]
no_heart_mean=[]
for k in features_to_quantize:
	heart_s=0.0
	heart_c=0.0
	non_heart_s=0.0
	non_heart_c=0.0
	for (i,j) in zip(food_df['heart_disease'],food_df[k]):
		if i==1:
			heart_s+=j
			heart_c+=1
		elif i==0:
			non_heart_s+=j
			non_heart_c+=1
	mean_heart=heart_s/heart_c
	mean_non_heart=non_heart_s/non_heart_c
	#if mean_cancer<400 and mean_non_cancer<400:
	dual_mean[k]=[mean_heart,mean_non_heart]
	heart_mean.append((k,mean_heart))
	no_heart_mean.append((k,mean_non_heart))
	#else:
	#skew_features.add(k)

#print dual_mean['RIBSQUAN']
heart_top_bottom=sorted(heart_mean,key=lambda x: x[1],reverse=True)
no_heart_top_bottom=sorted(no_heart_mean,key=lambda x: x[1],reverse=True)
heart_top_bottom=heart_top_bottom[:7]+heart_top_bottom[-7:]
no_heart_top_bottom=no_heart_top_bottom[:7]+no_heart_top_bottom[-7:]

#print cancer_top_bottom
#print no_cancer_top_bottom

heart_food=set([])
for i in heart_top_bottom:
	heart_food.add(i[0])
for i in no_heart_top_bottom:
	heart_food.add(i[0])

heart_types=['heart','no_heart']
heart_food_items=[]
data_heart={'heart_food_items':[],'non_heart_values':[],'heart_values':[]}
for key in heart_food:
	key=key.replace('QUAN','')
	data_heart['heart_food_items'].append(key)
	data_heart['heart_values'].append(dual_mean[key+'QUAN'][0])
	data_heart['non_heart_values'].append(dual_mean[key+'QUAN'][1])

x = [ (food_item.replace('QUAN',''), category) for food_item in heart_food for category in heart_types ]

counts = sum(zip(data_heart['heart_values'], data_heart['non_heart_values']), ()) 
get_color_vals=[]
for i in range(len(counts)):
	if i%2==0:
		get_color_vals.append(["blue"])
	else:
		get_color_vals.append(["green"])	

source = ColumnDataSource(data=dict(x=x, counts=counts))
p2 = figure(x_range=FactorRange(*x), plot_height=1000, plot_width=1600,title="Mean food quantity for heart patients",tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'])
p2.vbar(x='x', top='counts', width=0.9, source=source,color=get_color_vals)
#p.x_range.range_padding = 0.1
p2.xaxis.major_label_orientation = 1
p2.xgrid.grid_line_color = None

tab2=Panel(child=p2,title="Heart patient Food trends")

dual_mean={}
diabetes_mean=[]
no_diabetes_mean=[]
for k in features_to_quantize:
	diabetes_s=0.0
	diabetes_c=0.0
	non_diabetes_s=0.0
	non_diabetes_c=0.0
	for (i,j) in zip(food_df['diabetes'],food_df[k]):
		if i==1:
			diabetes_s+=j
			diabetes_c+=1
		elif i==0:
			non_diabetes_s+=j
			non_diabetes_c+=1
	mean_diabetes=diabetes_s/diabetes_c
	mean_non_diabetes=non_diabetes_s/non_diabetes_c
	#if mean_cancer<400 and mean_non_cancer<400:
	dual_mean[k]=[mean_diabetes,mean_non_diabetes]
	diabetes_mean.append((k,mean_diabetes))
	no_diabetes_mean.append((k,mean_non_diabetes))
	#else:
	#skew_features.add(k)

#print dual_mean['RIBSQUAN']
diabetes_top_bottom=sorted(diabetes_mean,key=lambda x: x[1],reverse=True)
no_diabetes_top_bottom=sorted(no_diabetes_mean,key=lambda x: x[1],reverse=True)
diabetes_top_bottom=diabetes_top_bottom[:7]+diabetes_top_bottom[-7:]
no_diabetes_top_bottom=no_diabetes_top_bottom[:7]+no_diabetes_top_bottom[-7:]

#print cancer_top_bottom
#print no_cancer_top_bottom

diabetes_food=set([])
for i in diabetes_top_bottom:
	diabetes_food.add(i[0])
for i in no_diabetes_top_bottom:
	diabetes_food.add(i[0])

diabetes_types=['diabet','no_diabet']
diabetes_food_items=[]
data_diabetes={'diabetes_food_items':[],'non_diabetes_values':[],'diabetes_values':[]}
for key in diabetes_food:
	key=key.replace('QUAN','')
	data_diabetes['diabetes_food_items'].append(key)
	data_diabetes['diabetes_values'].append(dual_mean[key+'QUAN'][0])
	data_diabetes['non_diabetes_values'].append(dual_mean[key+'QUAN'][1])

x = [ (food_item.replace('QUAN',''), category) for food_item in diabetes_food for category in diabetes_types ]

counts = sum(zip(data_diabetes['diabetes_values'], data_diabetes['non_diabetes_values']), ()) 
get_color_vals=[]
for i in range(len(counts)):
	if i%2==0:
		get_color_vals.append(["blue"])
	else:
		get_color_vals.append(["green"])	

source = ColumnDataSource(data=dict(x=x, counts=counts))
p3 = figure(x_range=FactorRange(*x), plot_height=1000, plot_width=1600,title="Mean food quantity for diabetes patients",tools=[hover,'pan','wheel_zoom','box_zoom','box_select','lasso_select'])
p3.vbar(x='x', top='counts', width=0.9, source=source,color=get_color_vals)
#p.x_range.range_padding = 0.1
p2.xaxis.major_label_orientation = 1
p3.xgrid.grid_line_color = None

tab3=Panel(child=p3,title="Diabetes patient Food trends")


output_file("Food Trends of people.html")
tabs=Tabs(tabs=[tab1,tab2,tab3])
show(tabs)


lin_alg_X=[]
for i in features_to_quantize[:54]:
	lin_alg_row=[]
	for j in food_df[i]:
		lin_alg_row.append(j)
	lin_alg_X.append(lin_alg_row)
 

lin_alg_Y=food_df["VITAMINDQUAN"]
ans_vitd= np.linalg.solve(lin_alg_X, lin_alg_Y)


lin_alg_Y=food_df["VITAMINEQUAN"]
ans_vite= np.linalg.solve(lin_alg_X, lin_alg_Y)


lin_alg_Y=food_df["WATERQUAN"]
ans_water= np.linalg.solve(lin_alg_X, lin_alg_Y)


lin_alg_Y=food_df["DT_NITROGEN"]
ans_nitrogen= np.linalg.solve(lin_alg_X, lin_alg_Y)


output_file("nutrition_table.html")
data=dict(food_items=features_to_quantize[:54],vitd_values=ans_vitd,vite_values=ans_vite,water_values=ans_water,nitro_values=ans_nitrogen)
source=ColumnDataSource(data)
columns = [TableColumn(field="food_items", title="Food Item"),TableColumn(field="water_values", title="Water Quantity"),TableColumn(field="vitd_values", title="Vitamin D"),TableColumn(field="vite_values", title="Vitamin E"),TableColumn(field="nitro_values", title="Nitrogen Quantity")]
data_table = DataTable(source=source, columns=columns, width=1000, height=800)
tab1=Panel(child=data_table, title="Nutrition Summary (table)")

data2=dict(food_items=features_to_quantize[:10],vitd_values=ans_vitd[:10],vite_values=ans_vite[:10],water_values=ans_water[:10],nitro_values=ans_nitrogen[:10],x_range_numbers=[i for i in range(1,11)])
source2=ColumnDataSource(data2)
hover=HoverTool(tooltips=[("index","$index"),("(x,y)","($x,$y)")])
p2=figure(plot_width=1100, plot_height=1000,tools=[hover,'pan','wheel_zoom','box_zoom','box_select'], title="Plot showing (approx) nutrient values for a SAMPLE of food types (Hover for values)",x_axis_label="Food items",y_axis_label="Value (Info. missing about Units)",x_range=[i.replace('QUAN','') for i in features_to_quantize[:10]])
#p2.x_range.range_padding = 0.1
p2.circle('x_range_numbers','nitro_values' ,source=source2,color=(255,0,0),legend="Nitrogen",size=10,alpha=0.7)
p2.square('x_range_numbers','water_values' ,source=source2,color=(0,255,0),legend="Water",size=10,alpha=0.7)
p2.diamond('x_range_numbers','vitd_values' ,source=source2,color=(0,0,255),legend="Vitamin D",size=10,alpha=0.7)
p2.inverted_triangle('x_range_numbers','vite_values' ,source=source2,color="olive",legend="Vitamin E",size=10,alpha=0.7)

tab2=Panel(child=p2, title="Nutrition Summary (plot)")

tabs=Tabs(tabs=[tab1,tab2])

show(tabs)
#print food_df["OXALIC_ACID"]

"""
http://bokeh.pydata.org/en/latest/docs/gallery/bar_nested.html

"""
