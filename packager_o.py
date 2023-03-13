#!/usr/bin/env python
# coding: utf-8

# # Main

# In[1]:


# video_capture():
#     import numpy as np
#     import cv2
#     cap=cv2.VideoCapture(0)
#     while (cap.isOpened()):
#         ret,frame=cap.read()
#         cv2.imshow('output',frame)
#         if (cv2.waitKey(1) & 0xFF==ord('q')):
#             break
#     cap.release()
#     cv2.destroyAllWindows()


# In[2]:


import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
from tkinter import *
# from tkinter.messagebox import askyesno
import cv2
import mediapipe as mp

import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, BatchNormalization, Dense, Flatten, LayerNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.applications as appl
from sklearn.model_selection import train_test_split
import os
import cv2
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from tqdm import tqdm
from tensorflow.keras.utils import Sequence
from keras import callbacks  
from keras.models import load_model
from keras.utils import np_utils


# In[3]:


load_path='E:/VA/onehandtwohand/128/106words_DSLR_FH/'

model_name1 = '15layer_lr0.00001_106words_dslr128-99.79'

CATEGORIES=np.load(load_path+'cat_106.npy', allow_pickle=True)
IMG_SIZE=128
cat_len=len(CATEGORIES)
print(cat_len)


# In[4]:


# #load saved history
history_const=np.load(load_path+model_name1+'_history.npy',allow_pickle='TRUE').item()

# #load saved model
model1=load_model(load_path+model_name1+'_model.h5')

# model1.summary()


# In[5]:


import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results
    
def draw_landmarks(image, results):   
    #face
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
#     #pose
#     mp_drawing.draw_landmarks(
#         image,
#         results.pose_landmarks,
#         mp_holistic.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles
#         .get_default_pose_landmarks_style())
    
    #left hand
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
#         landmark_drawing_spec=None,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # right hand
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
#         landmark_drawing_spec=None,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())


# In[6]:


def evaluate_model(img_array):
    with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:    
        image, results = mediapipe_detection(img_array, holistic)
        draw_landmarks(image, results)
        if not (results.left_hand_landmarks or results.right_hand_landmarks):
            pass

        #white background
        img = np.zeros([480,480,3],dtype=np.uint8)
        img.fill(255) 
        draw_landmarks(img, results)

        # for prediction
        IMG_SIZE=128
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X = np.array(img).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        X = X.astype('float32')
        X /= 255
        X = np.array(X)
        Y = model1.predict(X,verbose=0)
    return Y,image


# In[ ]:





# In[7]:


# visualization live test code
############################################### for display#########################
from tkinter import *
from PIL import Image
from PIL import ImageTk
import cv2
#import imutils


def start():
   global cap
   cap = cv2.VideoCapture(0)
   visualizar()
   
# def show_accuracy():
#     label_acc = ZButton(f4, text=" Score: "+ str(np.argmax(Y))+'%',width=12,
#                   anchor = 'center',font=('bold',24), bg="cyan")
#     label_acc.place(relx=1, rely=0.28,anchor = 'e')
   
# def show_prediction():
#     label_pred = Button(f4, text=" Predicted: "+ CATEGORIES[np.argmax(Y)],width=12,
#                   anchor = 'center',font=('bold',24), bg="cyan")
#     label_pred.place(relx=0, rely=0.28,anchor = 'w')
   
def visualizar():
   global cap
   if cap is not None:
       ret, img_array = cap.read()
       if ret == True:
           #frame = imutils.resize(frame, width=frame.winfo_width())
           #cv2_video()
           img_array = cv2.flip(img_array, 1)
               #webcam
#             img_array = img_array[:,80:560, :]    
#             img_array = img_array[:, 224:800, :]
   #               #dslr
           img_array = cv2.resize(img_array[:, 224:800, :],(480,480))
           
       # evaluate CNN
           Y,image=evaluate_model(img_array)
#             display  

           display = cv2.resize(img_array, (750, 750))
           img_array = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

           im = Image.fromarray(img_array)
           img = ImageTk.PhotoImage(image=im)

           lblVideo.configure(image=img)
           lblVideo.image = img
           lblVideo.after(10, visualizar)
           
           #labelFrame
#             f4 = LabelFrame(f3, height=40, width=40) #background="pink"
#             frame_null.pack(expand='yes', fill='both', anchor ='center',ipadx=30,ipady=30)
           label_results = Label(f3, text="Report:",font=('black',24),bg="bisque")
           label_results.place(relx=0.5,rely=0.15,anchor='center')
           
           #button to display results
                       #button to display results
           btn5 = Button(f3, text=" Prediction: "+ CATEGORIES[np.argmax(Y)],width=40,
                         anchor = 'center',font=('bold',30), bg="cyan")#command= lambda:show_prediction()
           btn5.place(relx=0.5, rely=0.3,anchor = 'center')
           #float
           btn6 = Button(f3, text=" Accuracy: "+ str(float("%.2f" % np.max(Y))*100)+'%',width=40,
                         anchor = 'center',font=('bold',30), bg="cyan")#command= lambda:show_accuracy()
           btn6.place(relx=0.5, rely=0.4,anchor = 'center')
           #numerical
#             btn6 = Button(f3, text=" Accuracy: "+ str(int(np.max(Y)*100)+'%',width=40,
#                           anchor = 'center',font=('bold',30), bg="cyan")#command= lambda:show_accuracy()
#             btn6.place(relx=0.5, rely=0.4,anchor = 'center')
           
           
#             l5 = Label(btn5, text=" Prediction: "+ CATEGORIES[np.argmax(Y)], anchor = 'center',font=('bold',30), bg="cyan")#command= lambda:show_prediction()
#             l5.place(relx=0.5, rely=0.3,anchor = 'center')
#             l6 = Label(btn6, text=" Accuracy: "+ str(float("%.4f" % np.max(Y))*100)+'%',anchor = 'center',font=('bold',30), bg="cyan")#command= lambda:show_accuracy()
#             l6.place(relx=0.5, rely=0.4,anchor = 'center')
       else:
           lblVideo.image = ""
           cap.release()

def stop():
   global cap
   cap. release()
   cap=None
   return 0
   
################################################################### Date time and countdown #########   
def date_timee():
   from datetime import datetime
   now = datetime.now()
   date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
   return date_time

def countdown():
   for count in range(10,0,-1):
       print (count)

##################################################### show and save static frame ############################
def show_frame(img,frame2):
   name_time=date_timee()
   filename=image_path+name_time+'/.jpg'
   cv2.imshow(filename, frame)

def save_lastframe():
   name_time=date_timee()
   filename=image_path+name_time+'/.jpg'
   cv2.imwrite(filename, frame)
   
   

   
   
   


# In[8]:


# Import the library tkinter
from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
import cv2
import sys
# from tkvideo import tkvideo    
    # Create a GUI app
app = Tk()

# Give a title to your app
app.title("Vishal GUI")

# app.geometry('720x1000')
# app.configure(bg='white')
#getting screen width and height of display
width= app.winfo_screenwidth()*0.7
height= app.winfo_screenheight()*0.8
#setting tkinter window size
app.geometry("%dx%d" % (width, height))
app.resizable(True, True)

# show a label

# # ################################## FRAME null, column1 #################################################
# # Constructing the second frame, frame2
frame_null = LabelFrame(app, height=60, width=width) #background="pink"
frame_null.pack(expand='yes', fill='both', anchor ='center',ipadx=30,ipady=30)

# frame_null.place(x=200, y=10)
label_null1 = Label(frame_null, text=" Realtime Indian Sign Language Recognition",font=('bold',42))
label_null1.place(rely=0.5,relx=0.5,anchor='center')

# # ############################### FRAME 1, column 0 #################################################

# # Constructing the first frame, frame1


# # Buttons

# # ################################## FRAME 2, column1 #################################################
# # Constructing the second frame, frame2
labelframe2 = LabelFrame(app,height=600, width=width, text="Realtime",font=('bold',16)) #, bg="white",fg="black", padx=400, pady=250
labelframe2.pack(expand='yes', fill='both')


# three frame within
f1 = LabelFrame(labelframe2, height=height,text='Menu',font=('bold',16),background="bisque") #background="bisque"
f2 = LabelFrame(labelframe2,height=height,text="Click button on left panel to for testing",font=('bold',16)) # #background="pink"
f3 = LabelFrame(labelframe2, height=height,text='Result',font=('bold',16),background="bisque")

# three grid values 
f1.grid(row=0, column=0, sticky="nsew")
f2.grid(row=0, column=1, sticky="nsew")
f3.grid(row=0, column=2, sticky="nsew")
#configure for column
labelframe2.grid_columnconfigure(0, weight=1)
labelframe2.grid_columnconfigure(1, weight=2)
labelframe2.grid_columnconfigure(2, weight=1)

#put video in f2
cap = None
lblVideo = Label(f2)
lblVideo.place(relx=0.5, rely=0.5,anchor='center')


#my name
# label_null2 = Label(f3, text="Created by : Vishal & Agrima ")#font=('bold',16)
# label_null2.place(rely=1,relx=1,anchor='se')
#--------------------------------------------------------------------------------------------------------------
            #button to display results
btn9 = Button(f3,width=40,font=('bold',30), bg="cyan")#command= lambda:show_prediction()
btn9.place(relx=0.5, rely=0.3,anchor = 'center')
#float
btn10 = Button(f3,width=40,font=('bold',30), bg="cyan")#command= lambda:show_accuracy()
btn10.place(relx=0.5, rely=0.4,anchor = 'center')

labelframe2.grid_columnconfigure(0, weight=1)
#labels
label_m1 = Label(f1, text="Instructions:",font=('black',24),bg="bisque")
label_m1.place(rely=0.15,relx=0.5,anchor='center')

t1="1. Place your upper half within the frame. \n" 
t2="2. Keep your face close the center of the frame.\n"
t3="3. Sign properly.\n"

t=t1+t2+t3
label_ins = Label(f1, text=t,font=('black',16),bg="bisque")
label_ins.place(rely=0.25,relx=0.5,anchor='center')

#buttons
btn1 = Button(f1, text='Live test',width=20,command = lambda:start(),font=('bold',16))
btn1.place(relx=0.5, rely=0.45,anchor='center')
# btn3 = Button(f1, text="Stop", width=30, command=stop)
# btn3.place(relx=0.1, rely=0.6)
btn2 = Button(f1, text='Exit Program',width=20,command = app.destroy,font=('bold',16))
btn2.place(relx=0.5, rely=0.5,anchor='center')


# Make the loop for displaying app

app.mainloop()


# In[18]:


get_ipython().system('pip install imutils')


# In[ ]:


import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk

def get_image(frame):
    my=frame
#     my = tk.Tk()
#     my.geometry("410x300")  # Size of the window 
#     my.title('Upload image')
    my_font1=('times', 18, 'bold')
    l1 = tk.Label(my,text='Upload Files & display',width=30,font=my_font1)  
    l1.grid(row=1,column=1,columnspan=4)
    b5 = tk.Button(my, text='Upload Files', 
       width=20,command = lambda:upload_file())
    b5.grid(row=2,column=1,columnspan=4)

    def upload_file():
        f_types = [('Jpg Files', '*.jpg'),
        ('PNG Files','*.png')]   # type of files to select 
        filename = tk.filedialog.askopenfilename(multiple=True,filetypes=f_types)
        col=1 # start from column 1
        row=3 # start from row 3 
        for f in filename:
            img=Image.open(f) # read the image file
            img=img.resize((128,128)) # new width & height
            img=ImageTk.PhotoImage(img)
            e1 =tk.Label(my)
            e1.grid(row=row,column=col)
            e1.image = img # keep a reference! by attaching it to a widget attribute
            e1['image']=img # Show Image  
            if(col==3): # start new line after third column
                row=row+1# start wtih next row
                col=1    # start with first column
            else:       # within the same row 
                col=col+1
#         return img            # increase to next column                 
#     my.mainloop()
#     return img


# In[ ]:


from tkinter import *
from PIL import Image
from PIL import ImageTk
import cv2
import imutils

def start():
    global cap
    cap = cv2.VideoCapture(0)
    visualizar()
    
def visualizar():
    global cap
    if cap is not None:
        ret, frame = cap.read()
        if ret == True:
            frame = imutils.resize(frame, width=frame.width)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            im = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=im)

            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(10, visualizar)
        else:
            lblVideo.image = ""
            cap.release()

def stop():
    global cap
    cap. release()
    
cap = None
root = Tk()

btnIniciar = Button(root, text="Start", width=45, command=start)
btnIniciar.grid(column=0, row=0, padx=5, pady=5)

btnFinalizar = Button(root, text="Stop", width=45, command=stop)
btnFinalizar.grid(column=1, row=0, padx=5, pady=5)

lblVideo = Label(root)
lblVideo.grid(column=0, row=1, columnspan=2)

root.mainloop()    


# In[ ]:


import tkinter as tk

root = tk.Tk()
root.title('Steady State Data Processing')
root.geometry('{}x{}'.format(900, 500))


topFrame = tk.Frame(root, bg = 'lavender', width = 900, height=100, relief = 'raised') # , padx = 100, pady=100
topFrame.grid(row = 0, column = 0, columnspan = 3,  sticky="w")



labelCps = tk.Label(root, text='Cps', width = 0, height = 0, padx = 10, pady = 10) 
labelIgn = tk.Label(root, text='Ign', width = 0, height = 0, padx = 10, pady = 10) 
labelInj = tk.Label(root, text='Inj', width = 0, height = 0, padx = 10, pady = 10)


labelCps.grid(row = 1, column = 0, sticky='we')
labelIgn.grid(row = 1, column = 1, sticky='we')
labelInj.grid(row = 1, column = 2, sticky='we')

cpsFrame = tk.Frame(root, width = 300, height = 100, relief = 'raised') # , padx = 100, pady=100
cpsFrame.grid(row = 2, column = 0,  sticky="nsew")

ignFrame = tk.Frame(root, width = 300, height = 100, relief = 'raised') # , padx = 100, pady=100
ignFrame.grid(row = 2, column = 1,  sticky="nsew")

injFrame = tk.Frame(root, width = 300, height = 100, relief = 'raised') # , padx = 100, pady=100
injFrame.grid(row = 2, column = 2,  sticky="nsew")


labelAdv = tk.Label(cpsFrame, anchor = 'center', text='Cps adv threshold:') 
labelAdv.grid(row = 0, column = 0, sticky = 'w')

entryAdv = tk.Entry(cpsFrame)
entryAdv.grid(row = 0, column = 1, sticky = 'e')

labelIgn = tk.Label(ignFrame, justify = 'left', text = 'Dwell start threshold:') 
labelIgn.grid(row = 0, column = 0, sticky = 'w')
entryIgn = tk.Entry(ignFrame)
entryIgn.grid(row = 0, column = 1)
labelIgn = tk.Label(ignFrame, anchor = 'center', text = 'Dwell end threshold:') 
labelIgn.grid(row = 1, column = 0)
entryIgn = tk.Entry(ignFrame)
entryIgn.grid(row = 1, column = 1)

labelInj = tk.Label(injFrame, anchor = 'center', text = 'Inj start threshold:') 
labelInj.grid(row = 0, column = 0)
entryInj = tk.Entry(injFrame)
entryInj.grid(row = 0, column = 1)
labelInj = tk.Label(injFrame, anchor = 'center', text = 'Inj end threshold:') 
labelInj.grid(row = 1, column = 0)
entryInj = tk.Entry(injFrame)
entryInj.grid(row = 1, column = 1)



root.grid_rowconfigure(3, pad = 50)

applyButton = tk.Button(root, text = 'Apply', padx = 30, pady = 15)
applyButton.grid(row = 3, columnspan = 3)


text = ['Plot raw data', 'Plot tooth rpm', 'Plot cycle rpm', 'Plot ign data']
count = 0

# Button frame
frame = tk.Frame(root)
frame.grid(row=4, column=0, sticky='news', columnspan=4)
for t in text:

    # Expand the column widths as required by the window cavity.
    frame.grid_columnconfigure(count, weight=1)
    dataButton = tk.Button(frame ,text = t, width = 5, height = 5 ,anchor = 'center', padx = 30, pady = 15)
    dataButton.grid(row = 0, column = count, sticky = 'news', padx=30)
    # dataButton.grid_columnconfigure(count, weight = 2)    
    count = count + 1

root.mainloop()


# Capture live photo

# Upload image and display

# Two panels in a window

# In[ ]:


import tkinter as tk

root = tk.Tk()
root.geometry("200x100")

f1 = tk.LabelFrame(root, background="bisque", width=10, height=100)
f2 = tk.LabelFrame(root, background="pink", width=10, height=100)

f1.grid(row=0, column=0, sticky="nsew")
f2.grid(row=0, column=1, sticky="nsew")

root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

root.mainloop()


# In[ ]:




