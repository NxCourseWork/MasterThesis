from tkinter import *
import time
import tkinter
import random
import numpy as np
from itertools import  combinations, chain
import xlwt 
from xlwt import Workbook 
from random import randint


wb = Workbook() 
sheet1 = wb.add_sheet('Sheet 1') 
total =[] 
counter = 0
store_row = 0


def count():
	global counter, store_row 
	counter += 1
	if counter>1:
		print("\n ////////////////////////////////////////////////////")
		print( " Attempt:     " +str(counter))
		store_row +=1
	else:
		print("\n /////////////////////////////////////////////////////")
		print( " Attempt:     " +str(counter))
	print(counter)
	top.after(1000000, count)


def create_circle(x, y, r, canvasName, color): #center coordinates, radius
    x0 = x - r
    y0 = y - r
    x1 = x + r
    y1 = y + r
    return canvasName.create_oval(x0, y0, x1, y1, fill = color)


def Main():
	global random_words,top, opened
	top = Tk()
	myCanvas = Canvas(top,  width=550, height=250)
	myCanvas.pack()  
	myCanvas.create_text(270, 10, text = "Choose one of the “most preferable” color from three colors shown below", font= ('Times', '13', 'bold italic'))
	word_list = [['Lavender', 'Gray', 'Blue'],
[ 'Gray', 'Blue', 'Lavender'],
['Blue', 'Gray', 'Lavender'],
['Gray', 'Lavender', 'Purple'],
['Pink', 'Lavender', 'Blue'],
['Lavender', 'Gray', 'Pink'],
['Lavender', 'Pink', 'Purple'],
['Pink', 'Lavender', 'Purple'],
['Purple', 'Pink', 'Lavender'],
[ 'Gold','Cyan', 'Yellow'],
['Cyan', 'Yellow', 'Gold'],
['Lavender', 'Yellow', 'Gold'],
['Gray', 'Yellow', 'Gold'],
['Cyan', 'Yellow', 'Gray'],
[ 'Yellow', 'Gold', 'Cyan'],
[ 'Yellow', 'Gold', 'purple'],
[ 'Red', 'Light Salmon', 'Dark Orange'],
[ 'Dark Orange', 'Light Salmon', 'Red'],
[  'Light Salmon', 'Red', 'Dark Orange'],
['Orange', 'Lavender', 'Cyan'],
['Orange', 'Lavender', 'Black'],
['Dark Orange', 'Lavender', 'Orange'],
[ 'Blue' , 'Orange Red', 'Dark Orange'],
[ 'Orange' , 'Red', 'Dark Orange'],
[ 'Orange' , 'Orange Red', 'Dark Orange'],
[ 'Orange' , 'Orange Red', 'purple'], 
[ 'Green' , 'Orange Red', 'yellow'], 
['Dark Orange', 'Orange' , 'Orange Red'],
[ 'Yellow', 'Gold', 'Light Salmon'],
[ 'Light Salmon', 'Yellow', 'Gold'],
['Purple', 'Green', 'Pink'],
['Purple', 'Orange', 'Pink'],
['Green', 'Purple',  'Pink'],
['Purple',  'Pink', 'Green'],
['Red',  'Pink', 'Green'],
['Purple',  'Pink', 'Blue'],
['Purple',  'Pink', 'Cyan'],
[ 'Black', 'Purple', 'Tomato'], 
[ 'Orange', 'Pink', 'Tomato'], 
[ 'Black', 'Light Salmon', 'Tomato'], 
[ 'Black', 'Orange', 'Tomato'], 
[ 'Orange', 'Light Salmon', 'Tomato'], 
['Maroon', 'Dark Olive Green', 'Green'],
['Green', 'Dark Olive Green', 'Yellow'],
['Blue', 'Green', 'Yellow'],
['Orange', 'Green', 'Yellow'],
['Dark Turquoise', 'Coral', 'Blue'],
['Blue', 'Coral', 'Dark Turquoise'],
['Blue', 'Coral', 'Gray'],
['Blue', 'Coral', 'Cyan'],
['Cyan', 'Coral', 'Blue'],
['Green', 'Tomato', 'Yellow'],
['Turquoise', 'Tomato', 'Pink'],
[ 'Tomato', 'Pink', 'Turquoise'],
['Turquoise', 'steel blue', 'Pink'],
['navy', 'steel blue', 'Pink'],
['Turquoise', 'steel blue', 'Pink'],
['Green', 'Tomato', 'Yellow'],
['Red', 'Blue', 'Yellow'],
['Coral', 'Dark Turquoise', 'Blue'],
['Pink','Pale Violet Red', 'salmon'],
['Yellow','Pale Violet Red', 'purple'],
['Pale Violet Red','Orange', 'gray'],
['Purple','Pink', 'gray'],
['green','Pink', 'gray'],
['black','green', 'gray'],
['salmon','green', 'gray'],
['black','Pink', 'salmon'],
['Pink','Pale Violet Red', 'gold'],
['Purple','Pink', 'Pale Violet Red'],
['Orange','Pink', 'Pale Violet Red']]
	random_words_1  = random.sample(word_list, 1)
	random_words = random_words_1[0]
	#print("Random_Words:  " + str(random_words))
	create_circle(100, 100, 60, myCanvas, random_words[0])
	button_1 =tkinter.Button(top, height=1, width=13, bg= random_words[0], fg = 'black',command=store_data_0) 
	button_1.place(x=50, y=180)
	create_circle(250, 100, 60, myCanvas, random_words[1])
	button_2 =tkinter.Button(top, height=1, width=13, bg= random_words[1], fg = 'black',command=store_data_1) 
	button_2.place(x=200, y=180)
	create_circle(400, 100, 60, myCanvas, random_words[2])
	button_3 =tkinter.Button(top, height=1, width=13, bg= random_words[2], fg = 'black',command=store_data_2) 
	button_3.place(x=355, y=180)
	opened = True
	top.mainloop() 
    
def window_open_close():
	global opened,top
	if opened == True:	
		top.deiconify()
	else: 
		time.sleep(3)
		opened = True
		Main()
        
def combo_set(data):
	odd =[]
	l = []
	k = []
	n = 0
	combo_out = []
	for j in [data]:
		combo = list(combinations(j,2))
		for i in  combo:
			for p in data:
				if p not in i:
					odd.append(p)
	for i in combo:
		k = []
		for j in i:
			l = j
			k.append(l)
		k.append(odd[n])
		n+=1
		combo_out.append(k)
	return combo_out
		
	
def user_group_up(data):
	group_1 = []
	combo = data[0:3]
	chosen = data[3]
	if chosen in combo:
		c = combo.index(chosen)
		group_1 = combo[0:c]
		v  =combo[c+1:len(combo)]
		group_1[len(group_1):] = v
	return group_1, chosen

def compareset_make_binary(all_combo, g1,g2):
	bin_out = [i for i in range(0,len(all_combo))]
	current_combo=[]
	current_combo = g1
	current_combo.append(g2)
	for i in all_combo:
		j = all_combo.index(i)
		if i == current_combo:
			bin_out[j] = 1
		else:
			bin_out[j] = 0
			
	return bin_out
			
def store_in_excel(comboset):
	out = [] 
	col = 0
	if counter == 1:
		row = 0
		#print("I am in You") 
		sheet1.write(row, col, comboset[0])
		sheet1.write(row, col+1, comboset[1])
		sheet1.write(row, col+2, comboset[2])
		sheet1.write(row, col+3, comboset[3])                   
		wb.save('Irene_3.xls') 
	else:
		#print("Noooo I am in You")   
		print(store_row)  
		#print("I am in You") 
		sheet1.write(store_row, col, comboset[0])
		sheet1.write(store_row, col+1, comboset[1])
		sheet1.write(store_row, col+2, comboset[2])
		sheet1.write(store_row, col+3, comboset[3])                   
		wb.save('Irene_3.xls') 


# Finish everything here
def data_collection(out):
	
	global opened, top, random_words
	a=[]
	a  = random_words
	a.append(out)
	store_in_excel(a)
	top.withdraw()
	opened = False
	window_open_close()
	
		

def store_data_0():
	global random_words
	output = random_words[0]
	count()
	data_collection(output)
	
def store_data_1():
	global random_words
	output =random_words[1]
	count()
	data_collection(output)

	
def store_data_2():
	global random_words
	output =random_words[2]
	count()
	data_collection(output)

    
Main()