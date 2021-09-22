import numpy as np
import math
import cv2
import glob


def convolve(image,kernel):
    size = len(kernel)
    smallWidth = len(image) - (size-1)
    smallHeight = len(image[0]) - (size-1)
    output = [([0]*smallHeight) for i in range(smallWidth)]
    for i in range(smallWidth):
        for j in range (smallHeight):
            value = 0
            for n in range(size):
                for m in range(size):
                    value = value + image[i+n][j+m]*kernel[n][m]
            output[i][j] = int(value)      
    return output 
def sobel_filters(img):
    Kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    Ky = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

    Ix = convolve(img, Kx)
    Iy = convolve(img, Ky)
    
    G = [[0 for k in range(len(Ix[0]))] for i in range(len(Ix))]
    theta = [[0 for k in range(len(Ix[0]))] for i in range(len(Ix))]
    for i in range(len(Ix)):
        for k in range(len(Ix[0])):
            G[i][k] = math.hypot(Ix[i][k],Iy[i][k])
            theta[i][k] = math.atan2(Ix[i][k],Iy[i][k])
    g_max = max(map(max, G))
    G = [[ G[i][k]/g_max * 255 for k in range(len(Ix[0]))] for i in range(len(Ix))]
    return (G, theta)          
def gaussian_kernel(size, sigma=1):
    x = [[-1+k for i in range(size)] for k in range(size)]
    y = [[-1,0,1] for k in range(size)]
    g = [[0,0,0] for k in range(size)]
    normal = 1 / (2.0 * math.pi * sigma**2)
    for i in range(size):
        for k in range(size):
            g[i][k] = math.exp(-((x[i][k]**2 + y[i][k]**2) / (2.0*sigma**2))) * normal
    return g
def non_max_suppression(img, D):
    M, N = len(img), len(img[0])
    Z = [[0 for k in range(N)] for i in range(M)]
    for i in range(len(D)):
        for k in range(len(D[0])):
            D[i][k] = D[i][k]*180/math.pi
            if(D[i][k]<0):
                D[i][k] += 180

    for i in range(1,M-1):
        for j in range(1,N-1):
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
def threshold(img,lowthreshold=0.05, highthreshold=0.15,weak=75, strong=255):
    max_value = max(map(max, img))
    highThreshold = max_value * highthreshold
    lowThreshold = highThreshold * lowthreshold

    M, N = len(img), len(img[0])
    res = [[0 for k in range(N)] for i in range(M)]


    for i in range(len(res)):
        for k in range(len(res[0])):
            if img[i][k] >= highThreshold:
                res[i][k] = strong
            elif (img[i][k] <= highThreshold)  and  (img[i][k] >= lowThreshold):   
                res[i][k] = weak    

    return res
def hysteresis(img,weak=75, strong=255, ):

    M, N = len(img), len(img[0])
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i][j] == weak):
                try:
                    if ((img[i+1][j-1] == strong) or (img[i+1][j] == strong) or (img[i+1][j+1] == strong)
                        or (img[i][j-1] == strong) or (img[i][j+1] == strong)
                        or (img[i-1][j-1] == strong) or (img[i-1][j] == strong) or (img[i-1][j+1] == strong)):
                        img[i][j] = strong
                    else:
                        img[i][j] = 0
                except IndexError as e:
                    pass

    return img
if __name__ == "__main__":
    ext = ['jpg', 'jpeg','png']    # Add image formats here
    files = []
    [files.extend(glob.glob('*.' + e)) for e in ext]
    images = [cv2.imread(file,0) for file in files]
    filenames = [file.split(".")[0] for file in files]
    gk = gaussian_kernel(3,1)
    for i in range(len(images)):
        img = images[i]
        filename = filenames[i]
        smooth = convolve(img,gk)
        g,theta =  sobel_filters(smooth)
        non_max = non_max_suppression(g,theta)
        thresholdImg = threshold(non_max)
        img_final = np.array(hysteresis(thresholdImg))
        cv2.imwrite(filename + "_canny_edge.jpg",img_final)



