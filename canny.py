import numpy as np
import math
import cv2
import glob

class CannyEdgeDetection(object):
    def __init__(self,kernel_size=3,sigma=1,lowthreshold=0.05, highthreshold=0.15,weak=75, strong=255):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.lowthreshold = lowthreshold
        self.highthreshold = highthreshold
        self.weak = weak
        self.strong = strong
        self.kernel = self.gaussian_kernel()
        self.M = 0
        self.N = 0
    def gaussian_kernel(self):
        x = [[-1+k for i in range(self.kernel_size)] for k in range(self.kernel_size)]
        y = [[-1,0,1] for k in range(self.kernel_size)]
        g = [[0,0,0] for k in range(self.kernel_size)]
        normal = 1 / (2.0 * math.pi * self.sigma**2)
        for i in range(self.kernel_size):
            for k in range(self.kernel_size):
                g[i][k] = math.exp(-((x[i][k]**2 + y[i][k]**2) / (2.0*self.sigma**2))) * normal
        return g
    def convolve(self,img,kernel):
        size = len(kernel)
        smallWidth = len(img) - (size-1)
        smallHeight = len(img[0]) - (size-1)
        output = [([0]*smallHeight) for i in range(smallWidth)]
        for i in range(smallWidth):
            for j in range (smallHeight):
                value = 0
                for n in range(size):
                    for m in range(size):
                        value = value + img[i+n][j+m]*kernel[n][m]
                output[i][j] = int(value)      
        return output   
    def sobel_filters(self,img):
        Kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        Ky = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

        Ix = self.convolve(img, Kx)
        Iy = self.convolve(img, Ky)
        
        G = [[0 for k in range(len(Ix[0]))] for i in range(len(Ix))]
        theta = [[0 for k in range(len(Ix[0]))] for i in range(len(Ix))]
        for i in range(len(Ix)):
            for k in range(len(Ix[0])):
                G[i][k] = math.hypot(Ix[i][k],Iy[i][k])
                theta[i][k] = math.atan2(Ix[i][k],Iy[i][k])
        g_max = max(map(max, G))
        G = [[ G[i][k]/g_max * 255 for k in range(len(Ix[0]))] for i in range(len(Ix))]
        return (G, theta)            
    def non_max_suppression(self,img, D):
        Z = [[0 for k in range(self.N)] for i in range(self.M)]
        for i in range(len(D)):
            for k in range(len(D[0])):
                D[i][k] = D[i][k]*180/math.pi
                if(D[i][k]<0):
                    D[i][k] += 180

        for i in range(1,self.M-1):
            for j in range(1,self.N-1):
                try:
                    q = 255
                    r = 255

                    #angle 0
                    if (0 <= D[i][j] < 22.5) or (157.5 <= D[i][j] <= 180):
                        q = img[i][j+1]
                        r = img[i][j-1]
                    #angle 45
                    elif (22.5 <= D[i][j]< 67.5):
                        q = img[i+1][j-1]
                        r = img[i-1][j+1]
                    #angle 90
                    elif (67.5 <= D[i][j] < 112.5):
                        q = img[i+1][j]
                        r = img[i-1][j]
                    #angle 135
                    elif (112.5 <= D[i][j] < 157.5):
                        q = img[i-1][j-1]
                        r = img[i+1][j+1]

                    if (img[i][j] >= q) and (img[i][j] >= r):
                        Z[i][j] = img[i][j]
                    else:
                        Z[i][j] = 0
                except IndexError as e:
                        pass        
        return Z 
    def threshold(self,img):
        max_value = max(map(max, img))
        highThreshold = max_value * self.highthreshold
        lowThreshold = highThreshold * self.lowthreshold

        res = [[0 for k in range(self.N)] for i in range(self.M)]


        for i in range(len(res)):
            for k in range(len(res[0])):
                if img[i][k] >= highThreshold:
                    res[i][k] = self.strong
                elif (img[i][k] <= highThreshold)  and  (img[i][k] >= lowThreshold):   
                    res[i][k] = self.weak    

        return res
    def hysteresis(self,img):
        for i in range(1, self.M-1):
            for j in range(1, self.N-1):
                if (img[i][j] == self.weak):
                    try:
                        if ((img[i+1][j-1] == self.strong) or (img[i+1][j] == self.strong) or (img[i+1][j+1] == self.strong)
                            or (img[i][j-1] == self.strong) or (img[i][j+1] == self.strong)
                            or (img[i-1][j-1] == self.strong) or (img[i-1][j] == self.strong) or (img[i-1][j+1] == self.strong)):
                            img[i][j] = self.strong
                        else:
                            img[i][j] = 0
                    except IndexError as e:
                        pass

        return img
    def detect(self, img):
        self.M, self.N = len(img), len(img[0])
        smooth = self.convolve(img,self.kernel)
        g,theta =  self.sobel_filters(smooth)
        non_max = self.non_max_suppression(g,theta)
        thresholdImg = self.threshold(non_max)
        img_final = np.array(self.hysteresis(thresholdImg))
        return img_final

if __name__ == "__main__":
    ext = ['jpg', 'jpeg','png']
    files = []
    [files.extend(glob.glob('*.' + e)) for e in ext]
    images = [cv2.imread(file,0) for file in files]
    filenames = [file.split(".")[0] for file in files]
    ced = CannyEdgeDetection()
    for i in range(len(images)):
        img = images[i]
        filename = filenames[i]
        img_final = ced.detect(img)
        cv2.imwrite(filename + "_canny_edge.jpg",img_final)



