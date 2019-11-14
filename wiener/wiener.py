import os
import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import gaussian, convolve2d
import matplotlib.pyplot as plt

def blur(img,kernel_size=3):
    img_ruido= np.copy(img)
    h=np.eye(kernel_size)/kernel_size
    img_ruido=convolve2d(img_ruido,h,mode='valid')
    return img_ruido

def add_gaussian_noise(img,sigma):
    gauss=np.random.normal(0,sigma,np.shape(img))
    noise = img + gauss
    noise[noise < 0]=0
    noise[noise > 255]=255
    return noise



def filtro_wiener(img,kernel, k):
    kernel /= np.sum(kernel)
    img_ruido = np.copy(img)
    img_ruido = fft2(img_ruido)
    kernel = fft2(kernel,s=img.shape)
    kernel= np.conj(kernel)/(np.abs(kernel) ** 2 + k)
    img_ruido = img_ruido * kernel
    img_ruido = np.abs(ifft2(img_ruido))
    return img_ruido

def gauss_kernel(kernel_size=3):
    h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    h = np.dot(h, h.transpose())
    h /= np.sum(h)
    return h

def rgb2gray(rgb):

    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


if __name__ == "__main__":
    file_name = os.path.join('/home/boaro/Área de Trabalho/teste.jpg')

    img = rgb2gray(plt.imread(file_name))

    ruido= blur(img,kernel_size=7)

    img_ruido=add_gaussian_noise(ruido, sigma=20)


    kernel = gauss_kernel(5)
    img_pos=filtro_wiener(img_ruido,kernel,k=10)

    display=[img,img_ruido,img_pos]
    label=['Imagem Original','Imagem após a aplicação de ruído','Imagem após a filtragem']

    fig=plt.figure(figsize=(12,10))

    for i in range(len(display)):
        fig.add_subplot(2,2,i+1)
        plt.imshow(display[i],cmap='gray')
        plt.title(label[i])

    plt.show()

