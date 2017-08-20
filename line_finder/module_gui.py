# ref: http://www.pyimagesearch.com/2016/05/23/opencv-with-tkinter/
from tkinter import *
from tkinter import filedialog
from tkinter import ttk

from PIL import Image
from PIL import ImageTk

from line_finder.thresholding import Channel, Gradient, Module, Thresholding
import numpy as np

import cv2


def select_image():
    # grab a reference to the image panels
    global panelA_content, img

    # open a file chooser dialog and allow the user to select an input
    # image
    path = filedialog.askopenfilename()

    # ensure a file path was selected
    if len(path) > 0:
        img = cv2.imread(path)

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        # if the panels are None, initialize them
        if panelA_content is None:
            # the first panel will store our original image
            panelA_content = ttk.Label(panelA, image=image)
            panelA_content.image = image
            panelA_content.grid(column=0, row=0)
        # otherwise, update the image panels
        else:
            # update the pannels
            panelA_content.configure(image=image)
            panelA_content.image = image
    change_dropdown()
    update_panelC()


def change_dropdown(*args):
    channel = channel_var.get()
    gradient = gradient_var.get()

    channel_functor = Channel.gray
    if channel == 'blue':
        channel_functor = Channel.blue_channel
    elif channel == 'green':
        channel_functor = Channel.red_channel
    elif channel == 'hue':
        channel_functor = Channel.h_channel
    elif channel == 'lightness':
        channel_functor = Channel.l_channel
    elif channel == 'saturation':
        channel_functor = Channel.s_channel

    gradient_functor = None
    if gradient == 'x':
        gradient_functor = Gradient.sobel_x
    elif gradient == 'y':
        gradient_functor = Gradient.sobel_y
    elif gradient == 'magnitude':
        gradient_functor = Gradient.sobel_magnitude
    elif gradient == 'direction':
        gradient_functor = Gradient.sobel_direction

    global panelB_content, img, processed
    name, processed = Module.transform(img, channel_functor, gradient_functor)

    image = cv2.resize(processed, (0, 0), fx=scale, fy=scale)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)

    if panelB_content is None:
        panelB_content = ttk.Label(panelB, image=image)
        panelB_content.image = image
        panelB_content.grid(column=0, row=0)
    else:
        panelB_content.configure(image=image)
        panelB_content.image = image

    min_scale.config(from_=np.amin(processed))
    max_scale.config(to=np.amax(processed))
    update_panelC()


def update_panelC():
    min = min_scale_var.get()
    max = max_scale_var.get()

    global panelC_content, processed

    threshed = Thresholding.range(processed, (min, max))
    threshed = cv2.resize(threshed, (0, 0), fx=scale, fy=scale)

    image = Image.fromarray(threshed * 255)
    image = ImageTk.PhotoImage(image)

    if panelC_content is None:
        panelC_content = ttk.Label(panelC, image=image)
        panelC_content.image = image
        panelC_content.grid(column=0, row=0)
    else:
        panelC_content.configure(image=image)
        panelC_content.image = image


def change_min_scale(val):
    # max_scale.config(from_=val)
    min_label.config(text="{:.2f}".format(float(val)))

    update_panelC()


def change_max_scale(val):
    # min_scale.config(to=val)
    max_label.config(text="%.2d" % float(val))
    update_panelC()


scale = 0.3
image_width = 1280
image_height = 720
img = None
processed = None

# initialize the window toolkit
root = Tk()
root.title("image processing")

mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

# button to
btn = ttk.Button(mainframe, text="Select an image", command=select_image)
btn.grid(column=0, row=0, columnspan=4, sticky=(W, E))

# image panel A
panelA = ttk.Frame(mainframe, borderwidth=5, relief="sunken", width=image_width * scale, height=image_height * scale)
panelA.grid(column=0, row=1, columnspan=4, rowspan=1, sticky=(N, S, E, W))
panelA_content = None

# drop down channel to select channel
channel_var = StringVar(root)
channel_choices = ['gray', 'blue', 'green', 'red', 'hue', 'lightness', 'saturation']
# channel_var.set('gray')  # set the default option

ttk.Label(mainframe, text="Choose Channel").grid(column=0, row=2)
popupMenu = ttk.OptionMenu(mainframe, channel_var, channel_choices[0], *channel_choices)
popupMenu.grid(column=1, row=2)
channel_var.trace('w', change_dropdown)

# drop down channel to select gradient
gradient_var = StringVar(root)
gradient_choices = ['none', 'x', 'y', 'magnitude', 'direction']
# gradient_var.set('none')  # set the default option

ttk.Label(mainframe, text="Choose Gradient").grid(column=2, row=2)
popupMenu = ttk.OptionMenu(mainframe, gradient_var, gradient_choices[0], *gradient_choices)
popupMenu.grid(column=3, row=2)
gradient_var.trace('w', change_dropdown)

# image panel B
panelB = ttk.Frame(mainframe, borderwidth=5, relief="sunken", width=image_width * scale, height=image_height * scale)
panelB.grid(column=0, row=4, columnspan=4, rowspan=1, sticky=(N, S, E, W))
panelB_content = None


# scrollbar
min = 0
max = 255

min_scale_var = DoubleVar(value=min)
max_scale_var = DoubleVar(value=max)

min_scale = ttk.Scale(mainframe, from_=min, to=max, orient='horizontal', command=change_min_scale, variable=min_scale_var)
max_scale = ttk.Scale(mainframe, from_=min, to=max, orient='horizontal', command=change_max_scale, variable=max_scale_var)
min_scale.grid(column=0, row=5, columnspan=2)
max_scale.grid(column=2, row=5, columnspan=2)

min_label = ttk.Label(mainframe, text=min)
min_label.grid(column=0, row=6)
max_label = ttk.Label(mainframe, text=max)
max_label.grid(column=2, row=6)

# image pannel C
panelC = ttk.Frame(mainframe, borderwidth=5, relief="sunken", width=image_width * scale, height=image_height * scale)
panelC.grid(column=0, row=7, columnspan=4, rowspan=1, sticky=(N, S, E, W))
panelC_content = None

# run
for child in mainframe.winfo_children():
    child.grid_configure(padx=5, pady=5)

# kick off the GUI
root.mainloop()
