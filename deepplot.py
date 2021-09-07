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

def PlainVsRes():
    plt.figure(figsize=(45, 15))
    img = plt.imread("figures/PlainvsResidual.png")
    img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def ResidualBlock():
    plt.figure(figsize=(12, 8))
    img = plt.imread("figures/Residual_blocks.png")
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def ResTypes():
    plt.figure(figsize=(13, 10))
    img = plt.imread("figures/ResNetsLayers.png")
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def NOfP():
    plt.figure(figsize=(10, 8))
    img = plt.imread("figures/NumberOfParameters.png")
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def ResNextVar():
    plt.figure(figsize=(30, 15))
    img = plt.imread('figures/ResNextVariants.png')
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def ResNextArch():
    plt.figure(figsize=(13, 10))
    img = plt.imread('figures/ResNextArchitecture.jpg')
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def InvertedResidual():
    plt.figure(figsize=(10, 8))
    img = plt.imread('figures/InvertedResidual.jpg')
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def LBottleNeck():
    plt.figure(figsize=(8, 8))
    img = plt.imread('figures/MobilenetBlock.jpg')
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def MBnetArch():
    plt.figure(figsize=(12, 10))
    img = plt.imread('figures/MobilenetArch.jpg')
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def Densenetview():
    plt.figure(figsize=(20, 15))
    img = plt.imread('figures/DenseNetOverview2.jpg')
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def Densenetarch():
    plt.figure(figsize=(16, 16))
    img = plt.imread('figures/DenseNetArch.jpg')
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def Densenetarch():
    plt.figure(figsize=(16, 16))
    img = plt.imread('figures/DenseNetArch.jpg')
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def Densenetarch2():
    plt.figure(figsize=(15, 5))
    img = plt.imread('figures/DenseNetArch2.jpg')
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()
