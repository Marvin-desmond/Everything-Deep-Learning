import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2

def read_img(paths, size=550):
    imgs = []
    for path in paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, img.shape[0]))
        imgs.append(img)
    f = np.vstack((imgs))
    return f



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

def LeNet5():
    plt.figure(figsize=(10, 10))
    LeNet = read_img(["figures/LeNet5_A.png", "figures/LeNet5_B.png"], size=500)
    plt.imshow(LeNet)
    plt.xticks([])
    plt.yticks([])
    plt.title("LeNet5")
    plt.show()

def AlexNet():
    plt.figure(figsize=(14, 10))
    AlexNet = read_img(["figures/AlexNet_A.png", "figures/AlexNet_B.png"], size=600)
    plt.imshow(AlexNet)
    plt.xticks([])
    plt.yticks([])
    plt.title("AlexNet")
    plt.show()


def VGG():
    plt.figure(figsize=(11, 11))
    VGG = read_img(["figures/VGG.png", "figures/VGG_B.png"], size=550)
    plt.imshow(VGG)
    plt.xticks([])
    plt.yticks([])
    plt.title("VGG architecture")
    plt.show()

def InceptionModule():
    plt.figure(figsize=(8, 8))
    img = plt.imread("figures/InceptionModule.png")
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def GoogleNet():
    plt.figure(figsize=(10, 10))
    img = plt.imread("figures/GoogleNet.png")
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def GoogleNetAux():
    plt.figure(figsize=(12, 40))
    img = plt.imread("figures/GoogleNetAux.png")
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()
