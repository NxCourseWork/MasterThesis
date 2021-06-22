#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

"""

@Author: Narmada Ambigapathy

@Email: narmada.ambika@gmail.com

@Github:

"""

#########################################################
#---------------------START-OF-CODE---------------------#
#########################################################


# Importing required packages
import tkinter as tk
import tkinter.ttk as ttk
import time
import random
import numpy as np
from itertools import combinations, chain
import xlwt
from xlwt import Workbook
from random import randint


wb = Workbook()
sheet1 = wb.add_sheet('Sheet 1')
total = []
counter = 0
store_row = 0


def count():
    global counter, store_row
    counter += 1
    if counter > 1:
        print("\n ////////////////////////////////////////////////////")
        print(" Attempt:     " + str(counter))
        store_row += 3
    else:
        print("\n /////////////////////////////////////////////////////")
        print(" Attempt:     " + str(counter))
    print(counter)
    top.after(1000000, count)


#########################################################
#---------------CLASSES USED IN THE GUI-----------------#
#########################################################

# For the idea and implementation for the Placeholder c.f. https://blog.teclado.com/tkinter-placeholder-entry-field/

class Date_Entry(ttk.Entry):
    def __init__(self, root, *args, format="en", delimeter=None,**kwargs,):
        super().__init__(root,*args, style="Placeholder.TEntry", **kwargs)
        accepted_formats = ["en", "de", "US"]
        self.root = root
        self.delete_mode = False
        self.set_placeholder_mode = False
        if delimeter!=None:
            try:
                len(delimeter) == 1
            except:
                print("Delimeter must be a single character")
                print("Setting delimeter to default")
                delimeter = None

        try:
            format in accepted_formats
        except:
            print("Format is not definded. Setting format to english (en)")
            format = "en"

        self.format = format

        if format == "en":
            if delimeter == None:
                delimeter = "-"
            self.insert("end", "DD" + delimeter +
                              "MM" + delimeter + "YYYY")
        elif format == "de":
            if delimeter == None:
                delimeter = "."
            self.insert("end", "DD" + delimeter +
                              "MM" + delimeter + "YYYY")
        elif format == "US":
            if delimeter == None:
                delimeter = "/"
            self.insert("end", "MM" + delimeter +
                              "DD" + delimeter + "YYYY")

        style = ttk.Style(root)
        style.configure("Placeholder.TEntry", foreground="#d5d5d5")
        self.placeholder = self.get()

        self.delimeter = delimeter

        validate_command = (self.register(self.on_validate), '%S','%d')
        self.config(validate='key', validatecommand=validate_command)

        self.bind('<KeyPress>', self.datemask)
        self.bind('<FocusIn>', self._clear_placeholder)
        self.bind("<FocusOut>", self._add_placeholder)

    def _clear_placeholder(self, e):
        if self["style"] == "Placeholder.TEntry":
            self.delete("0", "end")
            self["style"] = "TEntry"

    def _add_placeholder(self, e):
        if self.get()=="":
            self.set_placeholder_mode = True
            self.insert("0", self.placeholder)
            self["style"] = "Placeholder.TEntry"
            self.set_placeholder_mode = False


    # https://stackoverflow.com/questions/4140437/interactively-validating-entry-widget-content-in-tkinter/4140988#4140988
    def on_validate(self,inStr,acttyp):
        text = self.get()
        if acttyp == '1': #insert
            self.delete_mode=False
            if len(text) > 9:
                return False
            if len(text) in [2,5]:
                if not inStr == self.delimeter:
                    return False
            else:
                if (not inStr.isdigit()) and (not self.set_placeholder_mode):
                    return False
        if acttyp== '0': #deletion
            self.delete_mode=True
        return True

    # https://stackoverflow.com/questions/13242970/tkinter-entry-box-formatted-for-date
    def datemask(self, event):
        if not self.delete_mode:
            if len(self.get()) == 2:
                self.insert("end", self.delimeter)
            elif len(self.get()) == 5:
                self.insert("end", self.delimeter)
            elif len(self.get()) >= 10:
                self.delete(10, "end")




#---------------------------------------------------------#
#                     NEW CLASS                           #
#---------------------------------------------------------#




class circle():
    """Class that defines the clickable circle canvas objects
    """

    def __init__(self, x, y, r, fill_col, canv):
        """Init of the class

        Args:
                x (Integer): X-coordinate of the of the circle
                y (Integer): Y-coordinate of the center of the circle
                r (Integer): Radius of the circle
                fill_col (String): String defining giving the Tkinter filling color
                canv (Tkinter.Canvas): Canvas of our Tk() object
        """
        self.x = x
        self.y = y
        self.radius = r
        self.fill = fill_col
        self.canvas = canv
        self.print = self.canvas.create_circle(x, y, r, fill=self.fill)

# Command for drawing a circel with a ceratin radius r
# around some point (x,y)


def _create_circle(self, x, y, r, **kwargs):
    """Create a circle of radius r with center at (x,y) on a tkinter.canvas

    Args:
            x (Integer): X-coordinate of the center
            y (Integer): Y-coordinate of the center
            r (Integer): Radius of the circle

    Returns:
            Tkinter.canvas.create_oval object: A circle as canvas object
    """
    return self.create_oval(x-r, y-r, x+r, y+r, **kwargs)


# "monkey patching" of that new command
tk.Canvas.create_circle = _create_circle

# function to change coulour of the object instead of creating a new one


def change_color(canvas_object, new_color, canvas):
    """Change the fill color of the given 'canvas_object' of 'canvas' to 'new_color'

    Args:
            canvas_object (tkinter.canvas object): The canvas object, whose filling color should get changed
            new_color (String): String giving the tkinter name of the new filling color
            canvas (tkinter.canvas, optional): The tkinter.canvas the given canvas object belongs to.
                    Defaults to MyCanvas.
    """
    canvas.itemconfig(canvas_object, fill=new_color)

# function that should be bounded to the mouce-click event


def place_colored_circles(centers, radius, canvas, colors):
    circles = []
    for i in range(len(centers)):
        circles.append(circle(centers[i][0], centers[i][1], radius, colors[i], canvas))
    return circles

def give_new_option_sample(word_list, canvas):
    global random_words, circles
    random_words = random.sample(word_list, 3)
    print("Random_Words:  " + str(random_words))
    for i in range(len(circles)):
        change_color(circles[i].print,random_words[i],canvas)
    return random_words


def create_experiment_GUI():
    global top, opened, word_list, myCanvas, circles, random_words

    def callback(event):
        """Callback function of given 'event'

        Args:
                event (tkinter event): The event we are going to bind the callback to
        """
        i = -1
        for circ in circles:
            i += 1
            if np.linalg.norm(np.array([event.x, event.y])-np.array([circ.x, circ.y])) <= radius:
                store_data(i)

    top = tk.Tk()
    top.title('Experiment')
    myCanvas = tk.Canvas(top,  width=550, height=250)
    myCanvas.pack()
    myCanvas.create_text(270, 10, text="Choose one of the “most preferable” color from three colors shown below", font=(
        'Times', '13', 'bold italic'))
    word_list = ['Orange Red', 'Light Salmon', 'Dark Orange', 'Green', 'Pink', 'White', 'Cyan', 'Yellow', 'Gold', 'Purple', 'Dark Turquoise', 'Peach Puff', 'Tomato', 'Red', 'Coral', 'Lavender', 'Dark Olive Green', 'Orange', 'Pale Violet Red', 'Blue', 'Maroon',
                'Hot Pink', 'steel blue', 'Salmon', 'turquoise', 'deep sky blue', 'Turquoise', 'light steel blue', 'Aquamarine', 'Blue Violet', 'Yellow Green', 'Magenta2', 'Navy', 'Gray']
    random_words = random.sample(word_list, 3)
    print("Random_Words:  " + str(random_words))
    centers = [
        [120,130],
        [270,130],
        [420,130]
    ]
    print(centers[0][0])
    radius = 60
    circles = place_colored_circles(centers, radius, myCanvas, random_words)
    myCanvas.bind("<Button-1>", callback)
    myCanvas.pack()
    top.resizable(False,False)
    top.eval('tk::PlaceWindow . center')
    print("Created")
    opened = True
    top.mainloop()



def create_Login_GUI():
    global filename
    def _Login_filled(event):
        global filename
        start_allowed = True
        _name = name_entry.get()
        _birthday = birthday_entry.get()
        try:
            int(str.replace(_birthday,birthday_entry.delimeter,""))
        except:
            print("Please enter a valid date and/or Name.")
            start_allowed = False
        if (start_allowed) and (_name !="") and (len(_birthday)==10):
            filename = _name + "_" + _birthday
            filename = str.replace(filename," ","_")
            filename = str.replace(filename,birthday_entry.delimeter,"_")
            master.destroy()
            create_experiment_GUI()

    master = tk.Tk()
    master.geometry('220x100')
    master.resizable(False, False)
    master.title('Log in')
    master.eval('tk::PlaceWindow . center')

    # The Login Frane
    Login = ttk.Frame(master)
    Login.pack(padx=10, pady=10, fill="x", expand=True)

    # entry and label for entering the participants name
    name_label = ttk.Label(Login, text="Full Name:")
    name_label.pack(fill="x", expand=True)
    name = tk.StringVar
    name_entry = ttk.Entry(Login, textvariable=name)
    name_entry.pack(fill="x",expand=True)
    name_entry.focus()

    # label and entry for entering the participants birthday
    # required to create a more unique file name for the data sample

    birthday_label = ttk.Label(Login, text="Date of Birth:")
    birthday_label.pack(fill="x", expand=True)
    birthday = tk.StringVar
    birthday_entry = Date_Entry(Login, textvariable=birthday)
    birthday_entry.pack(fill="x", expand=True)
    master.bind('<Return>', _Login_filled)
    name_entry.focus()
    master.mainloop()


#########################################################
#---------------------END OF GUI------------------------#
#########################################################


def Run_Experiment(experiment_running):
    if not experiment_running:
        create_Login_GUI()
    else:
        # try:
        #     #top.winfo_exists()
        #     window_open_close
        # except:
        #     opened = True
        create_experiment_GUI()



def Main():
    """The main function of the GUI
    """
    global random_words, top, opened, word_list, myCanvas
    experiment_running = False
    Run_Experiment(experiment_running)



def window_open_close():
    """Function that hides 'top' for brief moment and shows it again with
    a new sample of color options.
    """
    global opened, top, random_words, word_list,circles,myCanvas
    # if opened == True:
    #     top.withdraw()
    # else:

    top.withdraw()
    time.sleep(0.5)
    opened = True
    top.deiconify()
    give_new_option_sample(word_list,myCanvas)


def combo_set(data):
    odd = []
    l = []
    k = []
    n = 0
    combo_out = []
    for j in [data]:
        combo = list(combinations(j, 2))
        for i in combo:
            for p in data:
                if p not in i:
                    odd.append(p)
    for i in combo:
        k = []
        for j in i:
            l = j
            k.append(l)
        k.append(odd[n])
        n += 1
        combo_out.append(k)
    return combo_out


def user_group_up(data):
    group_1 = []
    combo = data[0:3]
    chosen = data[3]
    if chosen in combo:
        c = combo.index(chosen)
        group_1 = combo[0:c]
        v = combo[c+1:len(combo)]
        group_1[len(group_1):] = v
    return group_1, chosen


def compareset_make_binary(all_combo, g1, g2):
    bin_out = [i for i in range(0, len(all_combo))]
    current_combo = []
    current_combo = g1
    current_combo.append(g2)
    for i in all_combo:
        j = all_combo.index(i)
        if i == current_combo:
            bin_out[j] = 1
        else:
            bin_out[j] = 0

    return bin_out


def store_in_excel(comboset, bin_o):
    global filename
    out = []
    for i in range(0, len(bin_o)):
        comboset[i].append(bin_o[i])
    print(comboset)
    if counter == 1:
        print("I am in You")
        for i, l in enumerate(comboset):
            for j, col in enumerate(l):
                sheet1.write(i, j, col)
                wb.save(filename + '.xls')
    else:
        print("Noooo I am in You")
        for i, l in enumerate(comboset):
            for j, col in enumerate(l):
                sheet1.write(i+store_row, j, col)
                wb.save(filename + '.xls')


# Finish everything here
def data_collection(out):

    global opened, top, random_words
    a = []
    a = random_words
    a.append(out)
    g_1, g_2 = user_group_up(a)
    print("group_1:  " + str(g_1) + "     \n chosen: " + str(g_2))
    comboset_ou = combo_set(random_words[0:3])
    binary_out = compareset_make_binary(comboset_ou, g_1, g_2)
    store_in_excel(comboset_ou, binary_out)
    # top.withdraw()
    # opened = False
    window_open_close()


def store_data(chosen_circle):
    global random_words
    output = random_words[chosen_circle]
    count()
    data_collection(output)


# def store_data_0():
#     global random_words
#     output = random_words[0]
#     count()
#     data_collection(output)


# def store_data_1():
#     global random_words
#     output = random_words[1]
#     count()
#     data_collection(output)


# def store_data_2():
#     global random_words
#     output = random_words[2]
#     count()
#     data_collection(output)


if __name__ == "__main__":
    Main()
