import sys
import matplotlib.pyplot as plt
import numpy as np

def showImage(imgs):
    for i in range(len(imgs)):
        plt.subplot(1, len(imgs), i + 1)
        img = plt.imread(imgs[i])
        plt.imshow(img)
        plt.axis('off')
        plt.title(imgs[i].split("/")[-1][:-4])
    plt.show()

def TLU():
    plt.figure(figsize=(10, 10))
    showImage(['figures/TLU.png'])

def Perceptron():
    plt.figure(figsize=(10, 10))
    showImage(['figures/Perceptron.png'])

def MLP():
    plt.figure(figsize=(10, 10))
    showImage(['figures/MLP.png'])
