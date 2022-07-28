#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tkinter.ttk as ttk
import tkinter as tk


# In[2]:


def counting_data_size(file):
    if file == "Basic_Training.txt":
        word_num = 3
        row_size = 12
        column_size = 9
        
    else:
        word_num = 15
        row_size = 10
        column_size = 10
        
    return word_num,row_size, column_size


# In[3]:


def data_processing(file,word_num):
    data_processed_1=np.genfromtxt(file, delimiter=1, filling_values=-1)  #將資料每列切出array 空白以-1代替
    data_processed_2 = np.split(data_processed_1, word_num) #依字數分開為不同array
    data_processed = np.array(data_processed_2) #再把全部array組在一array
    #print("training data after processing",data_processed)
    
    return data_processed


# In[4]:


def calculate_weight(train_data,column,row,word_num):
    sum_x_product=np.zeros([column*row,column*row])
    
    for i in range(word_num):
        x = train_data[i].reshape(column*row,1)
        x_product=np.dot(x,x.T)
        sum_x_product+= x_product
    weight = (1/column*row)*sum_x_product - (word_num/(column*row))* np.identity(column*row)

    return(weight)


# In[5]:


def calculate_treshold(column,row,weight):
    treshold = np.zeros([column*row,1])
    for i in range(column*row):
        treshold[i]=sum(weight[i])
    
    return(treshold)


# In[6]:


def corresponding_test_data(file):
    if file == 'Basic_Training.txt':
        test_file = 'Basic_Testing.txt'
    else:
        test_file = 'Bonus_Testing.txt'
    return test_file


# In[7]:


def recall(train_data,test_data,column,row,word_num,weight,treshold,Epoch):
    for i in range(word_num):
        test_data_trained = np.copy(test_data[i].reshape(column*row,1))
        test_data_recalled = np.zeros((column*row,1))
        print("第{0:d}個圖 原輸入".format(i+1))
        print_data(test_data_trained,column,row)
        for j in range(Epoch):
            for k in range(column*row):
                W_Xproduct=np.dot(weight,test_data_trained)
                #if W_Xproduct[k]>treshold[k]:
                if W_Xproduct[k]>0:
                    test_data_trained[k]=1
                #elif W_Xproduct[k]<treshold[k]:
                elif W_Xproduct[k]<0:
                    test_data_trained[k]=-1
                else:
                    pass
            print("第{0:d}次學習 recall結果".format(j+1))
            print_data(test_data_trained,column,row)
            if(test_data_recalled==test_data_trained).all():
                break
            else:
                test_data_recalled=np.copy(test_data_trained)
            
        #print("學習{0:d}後 最終recall結果".format(Epoch))
        #print_data(test_data_trained,column,row)
        print("**********************************************")


# In[8]:


def print_data(data,column,row):
    for i in range(column*row):
        if data[i]==-1:
            print(' ',end='')
        else:
            print('1',end='')
        if(i+1)%column == 0:
            print('')
    print('--------------------------')


# In[9]:


def Hopfield_Training():    
    #traning data Processing
    Epoch=int(entry_Epoch.get())
    file = str(var_filename.get())
    train_word_num, train_row,train_column = counting_data_size(file)
    train_processed = data_processing(file,train_word_num)
    
    #testing data Processing
    test_file=corresponding_test_data(file)
    test_processed = data_processing(test_file,train_word_num)
    
    #Class Example traning data
    #train_processed = np.array([[[1,-1,1]],[[-1,1,-1]]])
    #train_word_num=2
    #train_row=1
    #train_column=3
    #test_processed = np.array([[[1,1,-1]]])
    
    #Calculate Weight & Θ
    weight = calculate_weight(train_processed,train_column,train_row,train_word_num)
    treshold = calculate_treshold(train_column,train_row,weight)
    #print("W=",weight,"\n Θ=",treshold)
    recall(train_processed,test_processed,train_column,train_row,train_word_num,weight,treshold,Epoch)
    
    


# In[ ]:


#GUI
#介面基本設定
window= tk.Tk()
window.geometry('200x200')
window.title('NN_HW3_Hopflied-Irene')

label_top = tk.Label(window,text = "Choose training file")
label_top.grid(column=0, row=0)

#檔案選擇設定
file_option=('Basic_Training.txt','Bonus_Training.txt')
var_filename=tk.StringVar()

#下拉選單
##callback Funtion
def callbackFunc_showselect(event):
    selected_file=event.widget.get()
    label_selectfile.config(text=selected_file)
    
##下拉選單設定
combobox=ttk.Combobox(window,values=file_option,textvariable=var_filename)
combobox.grid(column=0, row=1)
combobox.current(0)

##顯示已選檔案
combobox.bind('<<ConboboxSelected>>',callbackFunc_showselect) 
label_selectfile = tk.Label(window)

#輸入學習次數
label_Epoch=tk.Label(window,text='輸入學習次數')
label_Epoch.grid(column=0, row=2)
entry_Epoch = tk.Entry(window)
entry_Epoch.grid(column=0, row=3)

#開始訓練按鈕
button_start = tk.Button(window,text='開始',command=Hopfield_Training)
button_start.grid(column=0, row=4)
window.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




