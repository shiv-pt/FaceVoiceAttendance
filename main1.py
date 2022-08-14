############################################# IMPORTING ################################################
import csv
import datetime
from sqlite3 import Timestamp
import time
import tkinter as tk
import tkinter.simpledialog as tsd
from tkinter import messagebox as mess
from tkinter import ttk

import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sounddevice as sd
import wavio as wv
from PIL import Image
from numpy import newaxis
from scipy.io.wavfile import write

import sounddevice as sd
from scipy.io.wavfile import write
from PIL import Image,ImageTk
from keras.models import load_model


import speech_recognition as s_r


############################################# FUNCTIONS ################################################

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

##################################################################################

def tick():
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200,tick)

###################################################################################

def contact():
    mess._show(title='Contact us', message="Please contact us on : 'xxxxxxxxxxxxx@gmail.com' ")

###################################################################################

def check_haarcascadefile():
    exists = os.path.isfile("haarcascade_frontalface_default.xml")
    if exists:
        pass
    else:
        mess._show(title='Some file missing', message='Please contact us for help')
        window.destroy()

###################################################################################

def save_pass():
    assure_path_exists("TrainingImageLabel/")
    exists1 = os.path.isfile("TrainingImageLabel\psd.txt")
    if exists1:
        tf = open("TrainingImageLabel\psd.txt", "r")
        key = tf.read()
    else:
        master.destroy()
        new_pas = tsd.askstring('Old Password not found', 'Please enter a new password below', show='*')
        if new_pas == None:
            mess._show(title='No Password Entered', message='Password not set!! Please try again')
        else:
            tf = open("TrainingImageLabel\psd.txt", "w")
            tf.write(new_pas)
            mess._show(title='Password Registered', message='New password was registered successfully!!')
            return
    op = (old.get())
    newp= (new.get())
    nnewp = (nnew.get())
    if (op == key):
        if(newp == nnewp):
            txf = open("TrainingImageLabel\psd.txt", "w")
            txf.write(newp)
        else:
            mess._show(title='Error', message='Confirm new password again!!!')
            return
    else:
        mess._show(title='Wrong Password', message='Please enter correct old password.')
        return
    mess._show(title='Password Changed', message='Password changed successfully!!')
    master.destroy()

###################################################################################

def change_pass():
    global master
    master = tk.Tk()
    master.geometry("400x160")
    master.resizable(False,False)
    master.title("Change Password")
    master.configure(background="white")
    lbl4 = tk.Label(master,text='    Enter Old Password',bg='white',font=('times', 12, ' bold '))
    lbl4.place(x=10,y=10)
    global old
    old=tk.Entry(master,width=25 ,fg="black",relief='solid',font=('times', 12, ' bold '),show='*')
    old.place(x=180,y=10)
    lbl5 = tk.Label(master, text='   Enter New Password', bg='white', font=('times', 12, ' bold '))
    lbl5.place(x=10, y=45)
    global new
    new = tk.Entry(master, width=25, fg="black",relief='solid', font=('times', 12, ' bold '),show='*')
    new.place(x=180, y=45)
    lbl6 = tk.Label(master, text='Confirm New Password', bg='white', font=('times', 12, ' bold '))
    lbl6.place(x=10, y=80)
    global nnew
    nnew = tk.Entry(master, width=25, fg="black", relief='solid',font=('times', 12, ' bold '),show='*')
    nnew.place(x=180, y=80)
    cancel=tk.Button(master,text="Cancel", command=master.destroy ,fg="black"  ,bg="red" ,height=1,width=25 , activebackground = "white" ,font=('times', 10, ' bold '))
    cancel.place(x=200, y=120)
    save1 = tk.Button(master, text="Save", command=save_pass, fg="black", bg="#3ece48", height = 1,width=25, activebackground="white", font=('times', 10, ' bold '))
    save1.place(x=10, y=120)
    master.mainloop()

#####################################################################################

######################################################################################

def clear():
    txt.delete(0, 'end')
    res = "1)Take Images  >>>  2)Save Profile"
    message1.configure(text=res)


def clear2():
    txt2.delete(0, 'end')
    res = "1)Take Images  >>>  2)Save Profile"
    message1.configure(text=res)

#######################################################################################

def TakeImages():
    check_haarcascadefile()
    columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")
    serial = 0
    exists = os.path.isfile("StudentDetails\StudentDetails.csv")
    if exists:
        with open("StudentDetails\StudentDetails.csv", 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            for l in reader1:
                serial = serial + 1
        serial = (serial // 2)
        csvFile1.close()
    else:
        with open("StudentDetails\StudentDetails.csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(columns)
            serial = 1
        csvFile1.close()
    Id = (txt.get())
    name = (txt2.get())
    if ((name.isalpha()) or (' ' in name)):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0

        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # incrementing sample number
                sampleNum = sampleNum + 1
                # saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage\ " + name + "." + str(serial) + "." + Id + '.' + str(sampleNum) + ".jpg",
                            gray[y:y + h, x:x + w])
                # display the frame
                cv2.imshow('Taking Images', img)
            # wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum > 100:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "Images Taken for ID : " + Id
        row = [serial, '', Id, '', name]
        with open('StudentDetails\StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message1.configure(text=res)
    else:
        if (name.isalpha() == False):
            res = "Enter Correct name"
            message.configure(text=res)

########################################################################################

def TrainImages():
    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel/")
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, ID = getImagesAndLabels("TrainingImage")
    try:
        recognizer.train(faces, np.array(ID))
    except:
        mess._show(title='No Registrations', message='Please Register someone first!!!')
        return
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Profile Saved Successfully"
    message1.configure(text=res)
    message.configure(text='Total Registrations till now  : ' + str(ID[0]))

############################################################################################3

def getImagesAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create empth face list
    faces = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(ID)
    return faces, Ids

###########################################################################################
def storeAttendance(name):
    col_names = ['Name', '', 'Date', '', 'Time']
    i=0
    ts = time.time()
    sub=(txtsub.get())
    print(sub)
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    path="Attendance\ " + date
    exists = os.path.isdir(path)
    print(exists)
    if exists == False:
        print("run")
        os.makedirs(path)
    path2 = path + "\ " + sub + ".csv"
    exists=os.path.isfile(path2)
    attendance = [name, '', str(date), '', str(timeStamp)]
    if exists:
        with open(path2, 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(attendance)
        csvFile1.close()
    else:
        with open(path2, 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(col_names)
            writer.writerow(attendance)
        csvFile1.close()
    with open(path2, 'r') as csvFile1:
        reader1 = csv.reader(csvFile1)
        for lines in reader1:
            i = i + 1
            if (i > 1):
                if (i % 2 != 0):
                    iidd = str(lines[0]) + '   '
                    tv.insert('', 0, text=str(i), values=(str(lines[0]), str(lines[2]), str(lines[4])))
    csvFile1.close()

def combineInput():
    print("Hello1")
    print(trackImg)
    print(type(trackImg))
    model = load_model("Model84Class26.h5")
    assure_path_exists("Attendance/")
    assure_path_exists("StudentDetails/")
    for k in tv.get_children():
        tv.delete(k)
    x, sr = librosa.load("recording0.wav", sr=44100)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    print("Hello2")
    audio_array = np.array(Xdb)
    audio_array = cv2.resize(audio_array, dsize=(500, 500), interpolation=cv2.INTER_CUBIC)
    audio_array = audio_array + 100
    aud = audio_array[:,:,newaxis]
    print("Hello3")
    im = Image.open("crop.jpg")
    image_array = np.array(im)
    #image_array = np.array(trackImg)
    image_array = cv2.resize(image_array, dsize=(500, 500), interpolation=cv2.INTER_CUBIC) #learn about interpolation parameter
    #image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    image_arra = image_array[:,:,newaxis]
    print("Hello4")
    y = image_arra
    y = np.dstack((y,aud))
    pr = (np.rint(model.predict(y[newaxis,:,:,:])).astype('int32'))
    st = ""
    for i in pr[0]:
        st+=str(i)
    print(st,map[st])
    ch=printname(map[st])
    if ch==1:
        frames[0].tkraise()
        storeAttendance(map[st])
    elif ch==2:
        frames[0].tkraise()
    else:
        frames[4].tkraise()
    print("Hello5")
    #plt.figure(figsize=(14, 5))
    #librosa.display.waveplot(x, sr=sr)
   # X = librosa.stft(x)
    #Xdb = librosa.amplitude_to_db(abs(X))
    #plt.figure(figsize=(14, 5))
    #librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    #plt.colorbar()
    #plt.figure(figsize=(14, 5))
    #librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    #plt.colorbar()
   # print(Xdb.shape)
    #audio_array = np.array(Xdb)
    #audio_array = cv2.resize(audio_array, dsize=(1000, 1000), interpolation=cv2.INTER_CUBIC)
    #print(type(audio_array))
    #plt.figure(figsize=(14, 5))
    #librosa.display.specshow(audio_array, sr=sr, x_axis='time', y_axis='log')
    #audio_array = audio_array + 100
    #plt.imshow(audio_array)
    #aud_array = audio_array[:,:,newaxis]
    #y = np.dstack((y,aud_array))

###########################################################################################

def TrackImages():
    check_haarcascadefile()
    assure_path_exists("Attendance/")
    assure_path_exists("StudentDetails/")
    for k in tv.get_children():
        tv.delete(k)
    msg = ''
    i = 0
    j = 0
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    exists3 = os.path.isfile("TrainingImageLabel\Trainner.yml")
    if exists3:
        recognizer.read("TrainingImageLabel\Trainner.yml")
    else:
        mess._show(title='Data Missing', message='Please click on Save Profile to reset data!!')
        return
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);

    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', '', 'Name', '', 'Date', '', 'Time']
    exists1 = os.path.isfile("StudentDetails\StudentDetails.csv")
    if exists1:
        df = pd.read_csv("StudentDetails\StudentDetails.csv")
    else:
        mess._show(title='Details Missing', message='Students details are missing, please check!')
        cam.release()
        cv2.destroyAllWindows()
        window.destroy()


    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
            
            ############ we can just pass gray[y:y+h,x:x+w] to our model

            print(conf)
            if (conf < 50):
                print("In")
                ts = time.time()
                # faces = im[y:y + h, x:x + w]
                # cv2.imwrite("cropped\sahil.jpg",gray[y:y + h, x:x + w])
                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = df.loc[df['SERIAL NO.'] == serial]['NAME'].values
                ID = df.loc[df['SERIAL NO.'] == serial]['ID'].values
                ID = str(ID)
                ID = ID[1:-1]
                bb = str(aa)
                bb = bb[2:-2]
                attendance = [str(ID), '', bb, '', str(date), '', str(timeStamp)]
                cv2.imwrite("crop.jpg",gray[y:y+h,x:x+w])
                return gray[y:y+h,x:x+w] 
            else:
                Id = 'Unknown'
                bb = str(Id)
            #cv2.putText(im, str(bb), (x, y + h), font, 1, (255, 255, 255), 2)
        cv2.imshow('Taking Attendance (Press Q to finish)', im)
        if (cv2.waitKey(1) == ord('q')):
            break
    ts = time.time()
    sub=(txtsub.get())
    print(sub)
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    path="Attendance\ " + date
    exists = os.path.isdir(path)
    print(exists)
    if exists == False:
        print("run")
        os.makedirs(path)
    path2 = path + "\ " + sub + ".csv"
    exists=os.path.isfile(path2)

    if exists:
        with open(path2, 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(attendance)
        csvFile1.close()
    else:
        with open(path2, 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(col_names)
            writer.writerow(attendance)
        csvFile1.close()
    with open(path2, 'r') as csvFile1:
        reader1 = csv.reader(csvFile1)
        for lines in reader1:
            i = i + 1
            if (i > 1):
                if (i % 2 != 0):
                    iidd = str(lines[0]) + '   '
                    #tv.insert('', 0, text=iidd, values=(str(lines[2]), str(lines[4]), str(lines[6])))
    csvFile1.close()
    cam.release()
    cv2.destroyAllWindows()


def TrackImages2():
    check_haarcascadefile()
    assure_path_exists("Attendance/")
    assure_path_exists("StudentDetails/")
    for k in tv.get_children():
        tv.delete(k)
    msg = ''
    i = 0
    j = 0
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    exists3 = os.path.isfile("TrainingImageLabel\Trainner.yml")
    if exists3:
        recognizer.read("TrainingImageLabel\Trainner.yml")
    else:
        mess._show(title='Data Missing', message='Please click on Save Profile to reset data!!')
        return
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);

    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = [ 'Name', '', 'Date', '', 'Time']
    exists1 = os.path.isfile("StudentDetails\StudentDetails.csv")
    if exists1:
        df = pd.read_csv("StudentDetails\StudentDetails.csv")
    else:
        mess._show(title='Details Missing', message='Students details are missing, please check!')
        cam.release()
        cv2.destroyAllWindows()
        window.destroy()


    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
        
            print(conf)
            if (conf < 50):
                print("In")
                ts = time.time()
                # faces = im[y:y + h, x:x + w]
                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = df.loc[df['SERIAL NO.'] == serial]['NAME'].values
                bb = str(aa)
                bb = bb[2:-2]
                attendance = [bb, '', str(date), '', str(timeStamp)]
                cv2.imwrite("crop.jpg",gray[y:y+h,x:x+w])
            else:
                Id = 'Unknown'
                bb = str(Id)
            cv2.putText(im, str(bb), (x, y + h), font, 1, (255, 255, 255), 2)
        cv2.imshow('Taking Attendance (Press Q to finish)', im)
        if (cv2.waitKey(1) == ord('q')):
            break
    ts = time.time()
    sub=(txtsubn.get())
    print(sub)
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    path="Attendance\ " + date
    exists = os.path.isdir(path)
    print(exists)
    if exists == False:
        print("run")
        os.makedirs(path)
    path2 = path + "\ " + sub + ".csv"
    exists=os.path.isfile(path2)

    if exists:
        with open(path2, 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(attendance)
        csvFile1.close()
    else:
        with open(path2, 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(col_names)
            writer.writerow(attendance)
        csvFile1.close()
    with open(path2, 'r') as csvFile1:
        reader1 = csv.reader(csvFile1)
        for lines in reader1:
            i = i + 1
            if (i > 1):
                if (i % 2 != 0):
                    iidd = str(lines[0]) + '   '
                    tvn.insert('', 0, text=str(i), values=(str(lines[0]), str(lines[2]), str(lines[4])))
    csvFile1.close()
    cam.release()
    cv2.destroyAllWindows()

######################################## USED STUFFS ############################################
    
global key
key = ''

ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
day,month,year=date.split("-")

mont={'01':'January',
      '02':'February',
      '03':'March',
      '04':'April',
      '05':'May',
      '06':'June',
      '07':'July',
      '08':'August',
      '09':'September',
      '10':'October',
      '11':'November',
      '12':'December'
      }

def record_audio():
    print("recording started")
    freq = 44100
    duration = 6
    recording = sd.rec(int(duration * freq),
                       samplerate=freq, channels=2)
    temp=6
    while temp>0:
        frames[2].update()
        temp = temp - 1
        tk.Label(frames[2], text=f"{str(temp)}", font="ariel 40", width=4, bg="black", fg="white").place(x=65, y=200)
        time.sleep(1)
        if temp==0:
            mess.showinfo("Time countdown","Time's up")
            frames[0].tkraise()
    sd.wait()
    write("recording0.wav", freq, recording)
    print(recording)
    combineInput()

def voice_frame():
    frames[2].tkraise()
    

######################################## GUI FRONT-END ###########################################

################# LOGIN FORM ####################


def login():
    #getting form data

    global lsg
    lsg=0
    uname=username.get()
    pwd=password.get()
    #applying empty validation
    if uname=='' or pwd=='':
        msg.set("fill the empty field!!!")
    else:
      if uname=="abcd" and pwd=="12345":
        msg.set("Login success")
        login_screen.destroy()
        lsg=1
      else:
        msg.set("Wrong username or password!!!")

#defining loginform function
def Loginform():
    global login_screen
    login_screen = tk.Tk()

    login_screen.resizable(True, False)
    login_screen.title("Attendance System")
    width = login_screen.winfo_screenwidth()
    height = login_screen.winfo_screenheight()

    login_screen.geometry("%dx%d" % (width, height))

    tk.Label(login_screen,width="300", text="Enter credentials to Access Attendance portal", bg="orange",fg="white").pack()

    lgframe = tk.Frame(login_screen).place()

    global  msg
    global username
    global password
    username = tk.StringVar()
    password = tk.StringVar()
    msg=tk.StringVar()
    tk.Label(lgframe,text="Username * ",font=(28)).place(x=(width/2)-35,y=height/2-200)

    tk.Entry(lgframe, textvariable=username,width=30,font=(28)).place(x=(width/2)-150,y=height/2-160)

    tk.Label(lgframe, text="Password * ",font=(28)).place(x=(width/2)-35,y=height/2-120)

    tk.Entry(lgframe, textvariable=password ,show="*",width=30,font=(28)).place(x=(width/2)-150,y=height/2-80)

    tk.Label(lgframe, text="",textvariable=msg).place(x=(width/2)-55,y=height/2-40)
    tk.Button(lgframe, text="Login", width=20, height=2, bg="orange",command=login,cursor="hand2").place(x=(width/2)-55,y=height/2)
    login_screen.mainloop()


Loginform()

######### LOGIN END ############


def printname(name):
    frames[3].tkraise()
    var = tk.IntVar()
    txtname.set(name)
    tk.Label(frames[3], textvariable=txtname, font="ariel 20", bg="black", fg="white").place(x=20, y=30)
    a=tk.Button(frames[3], cursor="hand2", text="Done", command=lambda:var.set(1), bg="green", width=4,font=('times', 15, ' bold ')).place(x=20, y=200)
    b=tk.Button(frames[3], cursor="hand2", text="Retry", command=lambda:var.set(2), bg="green", width=4,
              font=('times', 15, ' bold ')).place(x=120, y=200)
    c=tk.Button(frames[3], cursor="hand2", text="Method 2", command=lambda:var.set(3), bg="green", width=7,
              font=('times', 15, ' bold ')).place(x=210, y=200)

    frames[3].wait_variable(var)
    print(var.get())
    return var.get()


map={'00000000001000000000000000': 'Shivam_Prajapati', '00100000000000000000000000': 'Atharv_Wani', '00000000000000000000000100': 'sahil_sharma', '00000000000000100000000000': 'ameya_mahadev_gonal', '00000000000000000000000001': 'vaibhav_porwal', '00000000000000000000001000': 'rishabh_sharma', '00000000000000010000000000': 'anurag_ashish_khot', '00000000000001000000000000': 'akshay_a_kumar', '00000000000000001000000000': 'ayush_gupta', '00000000000000000100000000': 'charan_sai', '00000000000000000010000000': 'chirag_baliga', '00000000000000000001000000': 'harshit_handa', '00000000000000000000010000': 'ketan_vaish', '00000001000000000000000000': 'Rajot_Saha', '00000100000000000000000000': 'OS_Sumukh', '00000000000100000000000000': 'TarunSrivatsa_VS', '00000000010000000000000000': 'Shakthi_SagarM', '00000010000000000000000000': 'Prinson_Fernandes', '00000000100000000000000000': 'Ronit_Agarwal', '00000000000010000000000000': 'Uday_AS', '00010000000000000000000000': 'NM_Nishant', '01000000000000000000000000': 'Animesh_Singh', '00000000000000000000000010': 'sri_vishnu', '00001000000000000000000000': 'Nachiketa_Nalin', '00000000000000000000100000': 'k_a_sumukh', '10000000000000000000000000': 'Akash_H'}

window = tk.Tk()

width= window.winfo_screenwidth()
height= window.winfo_screenheight()

window.geometry("%dx%d" % (width, height))
window.resizable(True,False)
window.title("Attendance System")

frames={}

frames[0]=tk.Frame(window, bg="#b3cccc") #Light blue
frames[0].place(relx=0.4, rely=0.17, relwidth=0.33, relheight=0.80)

frames[1]=tk.Frame(window, bg="#b3cccc")
frames[1].place(relx=0.4, rely=0.17, relwidth=0.33, relheight=0.80)

###########  code for recording animation #################

frames[2]=tk.Frame(window, bg="black")
frames[2].place(relx=0.5, rely=0.3, relwidth=0.2, relheight=0.4)


frames[3]=tk.Frame(window,bg="black")
frames[3].place(relx=0.45, rely=0.3, relwidth=0.25, relheight=0.4)

frames[4]=tk.Frame(window, bg="#b3cccc") #Light blue
frames[4].place(relx=0.4, rely=0.17, relwidth=0.33, relheight=0.80)


img= (Image.open("voice.png"))

resized_image= img.resize((90,90), Image.ANTIALIAS)

photo=ImageTk.PhotoImage(resized_image)
myimg=tk.Label(frames[2],image=photo,background="black")
myimg.pack(padx=5,pady=5)

btn=tk.Button(frames[2],cursor="hand2",command=record_audio, text="Record",height=2,width=15,font=('times', 12, ' bold ')).pack()

frames[0].tkraise()

txtname=tk.StringVar()

# frame1 = tk.Frame(window, bg="#00aeff") #Light blue
# frame1.place(relx=0.11, rely=0.17, relwidth=0.39, relheight=0.80)
#
# frame2 = tk.Frame(window, bg="#00aeff")
# frame2.place(relx=0.51, rely=0.17, relwidth=0.38, relheight=0.80)

message3 = tk.Label(window, text="Face and Voice Recognition Based Attendance System" ,fg="white",bg="#262523" ,width=70 ,height=1,font=('times', 29, ' bold '))
message3.place(x=10, y=10)

frame3 = tk.Frame(window, bg="#c4c6ce") #gray
frame3.place(relx=0.52, rely=0.09, relwidth=0.09, relheight=0.07)

frame4 = tk.Frame(window, bg="#c4c6ce")
frame4.place(relx=0.36, rely=0.09, relwidth=0.16, relheight=0.07)

datef = tk.Label(frame4, text = day+"-"+mont[month]+"-"+year+"  |  ", fg="black" ,width=55 ,height=1,font=('times', 22, ' bold '))
datef.pack(fill='both',expand=1)

clock = tk.Label(frame3,fg="black",width=55 ,height=1,font=('times', 22, ' bold '))
clock.pack(fill='both',expand=1)
tick()

head2 = tk.Label(frames[1], text="                       For New Registrations                       ", fg="black",bg="#3ece48" ,font=('times', 17, ' bold ') )
head2.grid(row=0,column=0)

head1 = tk.Label(frames[0], text="                       For Already Registered                       ", fg="black",bg="#3ece48" ,font=('times', 17, ' bold ') )
head1.place(x=0,y=0)

head1n = tk.Label(frames[4],text="                                   Method 2                               ", fg="black",bg="#3ece48" ,font=('times', 17, ' bold ') )
head1n.place(x=0,y=0)

lbl = tk.Label(frames[1], text="Enter ID",width=20  ,height=1  ,fg="black"  ,bg="#b3cccc" ,font=('times', 17, ' bold ') )
lbl.place(x=80, y=55)

txt = tk.Entry(frames[1],width=32 ,fg="black",font=('times', 15, ' bold '))
txt.place(x=30, y=88)

lbl2 = tk.Label(frames[1], text="Enter Name",width=20  ,fg="black"  ,bg="#b3cccc" ,font=('times', 17, ' bold '))
lbl2.place(x=80, y=140)

txt2 = tk.Entry(frames[1],width=32 ,fg="black",font=('times', 15, ' bold ')  )
txt2.place(x=30, y=173)

message1 = tk.Label(frames[1], text="1)Take Images  >>>  2)Save Profile" ,bg="#b3cccc" ,fg="black"  ,width=39 ,height=1, activebackground = "yellow" ,font=('times', 15, ' bold '))
message1.place(x=7, y=230)

message = tk.Label(frames[1], text="" ,bg="#b3cccc" ,fg="black"  ,width=39,height=1, activebackground = "yellow" ,font=('times', 16, ' bold '))
message.place(x=7, y=450)

lbl3 = tk.Label(frames[0], text="Attendance",width=20  ,fg="black"  ,bg="#b3cccc"  ,height=1 ,font=('times', 17, ' bold '))
lbl3.place(x=100, y=380)

lblsub = tk.Label(frames[0], text="Select course:",width=20  ,fg="black"  ,bg="#b3cccc"  ,height=1 ,font=('times', 14, ' bold '))
lblsub.place(x=15, y=45)

txtsub = tk.Entry(frames[0],width=20,fg="black",font=('times', 15, ' bold '))
txtsub.insert(0,"aiml")
txtsub.place(x=200, y=45)

lbl3n = tk.Label(frames[4], text="Attendance",width=20  ,fg="black"  ,bg="#b3cccc"  ,height=1 ,font=('times', 17, ' bold '))
lbl3n.place(x=100, y=380)

lblsubn = tk.Label(frames[4], text="Select course:",width=20  ,fg="black"  ,bg="#b3cccc"  ,height=1 ,font=('times', 14, ' bold '))
lblsubn.place(x=15, y=45)

txtsubn = tk.Entry(frames[4],width=20,fg="black",font=('times', 15, ' bold '))
txtsubn.insert(0,"aiml")
txtsubn.place(x=200, y=45)

############### code for selecting a frame ################

def select_f1():
    frames[1].tkraise()

def select_f2():
    frames[0].tkraise()

def select_f3():
    frames[4].tkraise()

f1 = tk.Button(window,cursor="hand2", text="New Registration", command=select_f1   ,bg="yellow"  ,width=18,font=('times', 15, ' bold '))
f1.place(x=100, y=260)

f1 = tk.Button(window,cursor="hand2", text="Take attendance", command=select_f2  ,bg="yellow"  ,width=18,font=('times', 15, ' bold '))
f1.place(x=100, y=320)

f3 = tk.Button(window,cursor="hand2", text="Method 2", command=select_f3  ,bg="yellow"  ,width=18,font=('times', 15, ' bold '))
f3.place(x=100, y=380)




# options = [
#     "Monday",
#     "Tuesday",
#     "Wednesday",
#     "Thursday",
#     "Friday",
#     "Saturday",
#     "Sunday"
# ]
# clicked = tk.StringVar()
# clicked.set("Monday")
# drop = tk.OptionMenu(frame1, clicked, *options).place(x=200,y=45)


res=0
exists = os.path.isfile("StudentDetails\StudentDetails.csv")
if exists:
    with open("StudentDetails\StudentDetails.csv", 'r') as csvFile1:
        reader1 = csv.reader(csvFile1)
        for l in reader1:
            res = res + 1
    res = (res // 2) - 1
    csvFile1.close()
else:
    res = 0
message.configure(text='Total Registrations till now  : '+str(res))

##################### MENUBAR #################################

menubar = tk.Menu(window,relief='ridge')
filemenu = tk.Menu(menubar,tearoff=0)
filemenu.add_command(label='Change Password', command = change_pass)
filemenu.add_command(label='Contact Us', command = contact)
filemenu.add_command(label='Exit',command = window.destroy)
menubar.add_cascade(label='Help',font=('times', 29, ' bold '),menu=filemenu)

################## TREEVIEW ATTENDANCE TABLE ####################

tv= ttk.Treeview(frames[0],height =10,columns = ('name','date','time'))
tv.column('#0',width=82)
tv.column('name',width=130)
tv.column('date',width=133)
tv.column('time',width=133)
tv.grid(row=2,column=0,padx=(0,0),pady=(150,0),columnspan=4)
tv.heading('#0',text ='ID')
tv.heading('name',text ='NAME')
tv.heading('date',text ='DATE')
tv.heading('time',text ='TIME')

###################### SCROLLBAR ################################

scroll=ttk.Scrollbar(frames[0],orient='vertical',command=tv.yview)
scroll.grid(row=2,column=4,padx=(0,100),pady=(150,0),sticky='ns')
tv.configure(yscrollcommand=scroll.set)

###################### BUTTONS ##################################

clearButton = tk.Button(frames[1], text="Clear", command=clear  ,fg="black"  ,bg="#ea2a2a"  ,width=11 ,activebackground = "white" ,font=('times', 11, ' bold '))
clearButton.place(x=335, y=86)
clearButton2 = tk.Button(frames[1], text="Clear", command=clear2  ,fg="black"  ,bg="#ea2a2a"  ,width=11 , activebackground = "white" ,font=('times', 11, ' bold '))
clearButton2.place(x=335, y=172)
takeImg = tk.Button(frames[1], text="Take Images", command=TakeImages  ,fg="white"  ,bg="blue"  ,width=34  ,height=1, activebackground = "white" ,font=('times', 15, ' bold '))
takeImg.place(x=30, y=320)
trainImg = tk.Button(frames[1], text="Save Profile", command=TrainImages ,fg="white"  ,bg="blue"  ,width=34  ,height=1, activebackground = "white" ,font=('times', 15, ' bold '))
trainImg.place(x=30, y=380)
trackImg = tk.Button(frames[0],cursor="hand2", text="Take image", command=TrackImages  ,fg="black"  ,bg="yellow"  ,width=16  ,height=1, activebackground = "white" ,font=('times', 15, ' bold '))
trackImg.place(x=30,y=95)
aud = tk.Button(frames[0],cursor="hand2", text="Record audio", command=voice_frame  ,fg="black"  ,bg="yellow"  ,width=16  ,height=1, activebackground = "white" ,font=('times', 15, ' bold '))
aud.place(x=260,y=95)
quitWindow = tk.Button(frames[0], text="Quit", command=window.destroy  ,fg="black"  ,bg="red"  ,width=35 ,height=1, activebackground = "white" ,font=('times', 15, ' bold '))
quitWindow.place(x=30, y=450)

##################### END ######################################


################## TREEVIEW ATTENDANCE TABLE ####################

tvn= ttk.Treeview(frames[4],height =10,columns = ('name','date','time'))
tvn.column('#0',width=82)
tvn.column('name',width=130)
tvn.column('date',width=133)
tvn.column('time',width=133)
tvn.grid(row=2,column=0,padx=(0,0),pady=(150,0),columnspan=4)
tvn.heading('#0',text ='ID')
tvn.heading('name',text ='NAME')
tvn.heading('date',text ='DATE')
tvn.heading('time',text ='TIME')

###################### SCROLLBAR ################################

scrolln=ttk.Scrollbar(frames[4],orient='vertical',command=tv.yview)
scrolln.grid(row=2,column=4,padx=(0,100),pady=(150,0),sticky='ns')
tvn.configure(yscrollcommand=scrolln.set)


trackImgn = tk.Button(frames[4],cursor="hand2", text="Take image", command=TrackImages2  ,fg="black"  ,bg="yellow"  ,width=16  ,height=1, activebackground = "white" ,font=('times', 15, ' bold '))
trackImgn.place(x=100,y=95)

quitWindown = tk.Button(frames[4], text="Quit", command=window.destroy  ,fg="black"  ,bg="red"  ,width=35 ,height=1, activebackground = "white" ,font=('times', 15, ' bold '))
quitWindown.place(x=30, y=450)

window.configure(menu=menubar)
#model()

if lsg == 1:
    window.mainloop()



####################################################################################################


