from cgitb import text
#from distutils.cmd import Command
from sqlite3 import Row
from tkinter import filedialog
from tkinter import*
import tkinter as tk
import numpy as np
import sklearn
from sklearn import datasets
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
class perceptron:
    def __init__(self):
        self.Xdata = []
        self.ydata = []
        self.kind = []
        self.ep = 0
        self.ln = 0
        self.d = 0
        self.w = [0, 0, 0]
        self.arr = []
    def show(self):
        file_path = filedialog.askopenfilename()
        self.file = file_path
    def put_file(self):
        self.enter_epoch()
        self.enter_learningrate()
        f = open(self.file, 'r')
        for line in f.readlines():
            self.arr = line.split()
            self.arr[0] = float(self.arr[0])
            self.arr[1] = float(self.arr[1])
            self.Xdata.append(self.arr[:2])
            self.ydata.append(self.arr[2])
        X_train, X_test, y_train, y_test = train_test_split(self.Xdata, self.ydata, test_size = 0.33, random_state=1)
        self.training(X_train, y_train, X_test, y_test)
        plt.show()
    def enter_epoch(self):
        self.ep = en_epoch.get()
    def enter_learningrate(self):
        self.ln = en_lr.get()
    def training(self, X_train, y_train, X_test, y_test):
        X_trainx = []
        X_trainy = []
        for i in range(len(X_train)):
            X_trainx.append(X_train[i][0])
            X_trainy.append(X_train[i][1])
            if(y_train[i] == '1'):
                ax1.scatter(X_train[i][0], X_train[i][1], marker='+', c='red')
            else:
                ax1.scatter(X_train[i][0], X_train[i][1], marker='+', c='blue')
        ax1.set_xlim(-5, 5)
        ax1.set_ylim(-5, 5)
        w = [-1.0, 0, 1.0]
        for i in range(int(self.ep)):
            sgn = 0
            for j in range(len(X_train)):
                v = 0
                x = [-1, X_train[j][0], X_train[j][1]]
                if(y_train[j] == '1'):
                    self.d = 1
                else:
                    self.d = -1
                v = np.dot(w, x)
                if(v >= 0):
                    sgn = 1
                else:
                    sgn = -1
                if(sgn != self.d):
                    if(v >= 0):
                        for k in range(len(w)):
                            w[k] = w[k] - float(self.ln)*x[k]
                    else:
                        for k in range(len(w)):
                            w[k] = w[k] + float(self.ln)*x[k]
        weights.config(text=w)
        x = np.linspace(-5, 5, 500)
        y = (w[0] - w[1] * x)/w[2]
        ax1.plot(x, y, c="black")
        self.training_accuracy(w, X_train, y_train)
        self.testing(w, X_test, y_test)
    def training_accuracy(self, w, X_train, y_train):
        sum = 0
        for i in range(len(X_train)):
            value = w[1]*X_train[i][0] + w[2]*X_train[i][1]
            if((value > w[0] and y_train[i] == '1') or (value < w[0] and y_train[i] == '0') or (value < w[0] and y_train[i] == '2')):
                sum += 1
        train_acc.config(text=sum/len(X_train))
    def testing(self, w, X_test, y_test):
        X_testx = []
        X_testy = []
        for i in range(len(X_test)):
            X_testx.append(X_test[i][0])
            X_testy.append(X_test[i][1])
            if(y_test[i] == '1'):
                ax2.scatter(X_test[i][0], X_test[i][1], marker='+', c='red')
            else:
                ax2.scatter(X_test[i][0], X_test[i][1], marker='+', c='blue')
        ax2.set_xlim(-5, 5)
        ax2.set_ylim(-5, 5)
        x = np.linspace(-5, 5, 500)
        y = (w[0] - w[1] * x)/w[2]
        ax2.plot(x, y, c="black")
        self.test_accuracy(w, X_test, y_test)
    def test_accuracy(self, w, X_test, y_test):
        sum = 0
        for i in range(len(X_test)):
            value = w[1]*X_test[i][0] + w[2]*X_test[i][1]
            if((value > w[0] and y_test[i] == '1') or (value < w[0] and y_test[i] == '0') or (value < w[0] and y_test[i] == '2')):
                sum += 1
        test_acc.config(text=sum/len(X_test))


import tkinter as tk
from tkinter import Label, Entry, Button
import matplotlib.pyplot as plt

# 創建 perceptron 實例
per = perceptron()

# 創建主視窗
win = tk.Tk()
win.title("109502007 張原鳴")
win.geometry("500x600")
win.resizable(False, False)

# 上方區域：數據集和參數設置
frame_top = tk.Frame(win)
frame_top.pack(pady=10)

data = tk.Button(frame_top, text="Dataset", command=per.show)
data.grid(row=0, column=2, padx=10)

lb_epoch = Label(frame_top, text="Epoch:")
lb_epoch.grid(row=0, column=0, padx=5, sticky="e")
en_epoch = Entry(frame_top)
en_epoch.grid(row=0, column=1, padx=5)

lb_lr = Label(frame_top, text="Learning rate:")
lb_lr.grid(row=1, column=0, padx=5, sticky="e")
en_lr = Entry(frame_top)
en_lr.grid(row=1, column=1, padx=5)

# 中間區域：訓練與測試結果顯示
frame_middle = tk.Frame(win)
frame_middle.pack(pady=10)

lb_train_acc = Label(frame_middle, text="Training accuracy:")
lb_train_acc.grid(row=0, column=0, padx=5, sticky="e")
train_acc = Label(frame_middle, text=".")
train_acc.grid(row=0, column=1, padx=5)

lb_test_acc = Label(frame_middle, text="Test accuracy:")
lb_test_acc.grid(row=1, column=0, padx=5, sticky="e")
test_acc = Label(frame_middle, text="")
test_acc.grid(row=1, column=1, padx=5)

lb_weights = Label(frame_middle, text="w:")
lb_weights.grid(row=2, column=0, padx=5, sticky="e")
weights = Label(frame_middle, text=".")
weights.grid(row=2, column=1, padx=5)

# 下方區域：按鈕與圖形
frame_bottom = tk.Frame(win)
frame_bottom.pack(pady=20)

train_button = Button(frame_bottom, text="Training", command=per.put_file)
train_button.grid(row=0, column=0, padx=10)

# Matplotlib 圖形
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Perceptron Simulator')

win.mainloop()
