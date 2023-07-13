import matplotlib.pyplot as plt
import numpy as np
CEILING = 256

def imread(nome_arquivo):
    im = plt.imread(nome_arquivo)
    if (im.dtype == "float32"):
        im = np.uint8(255*im)
    if (len(im.shape) >= 3 and im.shape[2] > 3):
        im = im[:, :, 0:3]
    return im


def imshow(im):
    plot = plt.imshow(im, cmap=plt.gray(), origin="upper")
    plot.set_interpolation('nearest')
    plt.show()

def nchannels(im):
    return im.shape[2] if len(im.shape) >= 3 else 1

def size(im):
    return (im.shape[1],im.shape[0])

def rgb2gray(im):
    if nchannels(im) == 1:
        return im

    dimensions = size(im)
    newim = np.zeros((dimensions[1],dimensions[0]))

    for x in range(dimensions[0]):
        for y in range(dimensions[1]):
           newim[y][x] = 0.299*im[y][x][0] + 0.587*im[y][x][1] + 0.114*im[y][x][2]
    return np.uint8(newim)

def imreadgray(nome_arquivo):
    im = imread(nome_arquivo)
    return rgb2gray(im)

def thresh(im,lim):
    newim = rgb2gray(im)
    dimensions = size(im)
    for x in range(dimensions[0]):
        for y in range(dimensions[1]):
            newim[y][x] = 255 if newim[y][x] >= lim else 0
    return newim

def negative(im):
    newim = rgb2gray(im)
    dimensions = size(im)
    for x in range(dimensions[0]):
        for y in range(dimensions[1]):
            newim[y][x] = 255 - newim[y][x]
    return newim

#def constrast(im,f,m):

def hist(im):
    histogram = np.zeros(256,dtype=int) if nchannels(im) == 1 else np.zeros((3,256),dtype=int)
    dimensions = size(im)
    if(nchannels(im) == 1):  
        for x in range(dimensions[0]):
            for y in range(dimensions[1]):
                histogram[im[y][x]] += 1
    else:
        for x in range(dimensions[0]):
            for y in range(dimensions[1]):
                for z in range(3):
                    histogram[z][im[y][x][z]] += + 1
    return histogram

def showhist(histogram, binsize=1):
    #fazer para rgb
    bins = np.arange(0, 256, binsize)
    groupedhistogram = [sum(histogram[i:i+binsize]) for i in bins]
    plt.bar(bins, groupedhistogram,width=binsize)
    plt.show()

def histeq(im):
    newim = rgb2gray(im)
    histogram = hist(im)
    dimensions = size(im)
    npixels = dimensions[0]*dimensions[1]
    eqhistogram = np.cumsum(histogram)/npixels
    return eqhistogram

def extrapolate(im,x,y,s,t,radius,dimensions,operation):
    if(operation == 'convolution'):
        newx = x + radius - s 
        newy = y + radius - t
    if(operation == 'correlation'):
        newx = x + radius + s 
        newy = y + radius + t

    newx = min(dimensions[0]-1,newx)
    newx = max(0,newx)

    newy = min(dimensions[1]-1,newy)
    newy = max(0,newy)

    return (newx,newy)

#dimensions, channels, empty image
def iminfo(im):
    dimensions = size(im)
    channels = nchannels(im)
    newim = np.zeros((dimensions[1],dimensions[0])) if channels == 1 else np.zeros((dimensions[1],dimensions[0],channels))
    return dimensions,channels,newim

#modificar para filtros não-quadrados
def convolve(im,mask):
    dimensions,channels,convolutedim = iminfo(im)
    maskdimensions = size(mask)
    radius = int(maskdimensions[0]/2)
    for channel in range(channels):
        for x in range(dimensions[0]):
                for y in range(dimensions[1]):
                    convsum = 0
                    for s in range(maskdimensions[0]):
                        for t in range(maskdimensions[1]):
                            convx,convy = extrapolate(im,x,y,s,t,radius,dimensions,operation='convolution')
                            if(channels == 1):
                                convsum +=  im[convy][convx] * mask[t][s]
                            else:
                                convsum +=  im[convy][convx][channel] * mask[t][s]
                    if channels == 1:
                        convolutedim[y][x] = convsum
                    else:
                        convolutedim[y][x][channel] = convsum
    
    return np.uint8(convolutedim)

def maskblur():
    return np.array([[1,2,1],[2,4,2],[1,2,1]])/16

def masksharpen():
    return np.array([[0,1,0],[1,-4,1],[0,1,0]])

def seSquare3():
    return np.array([[1,1,1],[1,1,1],[1,1,1]])

def seCross3():
    return np.array([[0,1,0],[1,1,1],[0,1,0]])

def erode(im,mask):
    dimensions,channels,erodedim = iminfo(im)
    maskdimensions = size(mask)
    radius = int(maskdimensions[0]/2)

    for channel in range(channels):
        for x in range(dimensions[0]):
                for y in range(dimensions[1]):
                    minvalue = CEILING
                    for s in range(maskdimensions[0]):
                        for t in range(maskdimensions[1]):
                            corrx,corry = extrapolate(im,x,y,s,t,radius,dimensions,operation='convolution')
                            if(channels == 1):
                                minvalue = min(minvalue,im[corry][corrx]) if mask[t][s] == 1 else minvalue
                            else:
                                minvalue = min(minvalue,im[corry][corrx][channel]) if mask[t][s] == 1 else minvalue
                    if channels == 1:
                        erodedim[y][x] = minvalue
                    else:
                        erodedim[y][x][channel] = minvalue
            
    return np.uint8(erodedim)

def dilate(im,mask):
    dimensions,channels,erodedim = iminfo(im)
    maskdimensions = size(mask)
    radius = int(maskdimensions[0]/2)

    for channel in range(channels):
        for x in range(dimensions[0]):
                for y in range(dimensions[1]):
                    maxvalue = 0
                    for s in range(maskdimensions[0]):
                        for t in range(maskdimensions[1]):
                            corrx,corry = extrapolate(im,x,y,s,t,radius,dimensions,operation='convolution')
                            if(channels == 1):
                                maxvalue = max(maxvalue,im[corry][corrx]) if mask[t][s] == 1 else maxvalue
                            else:
                                maxvalue = max(maxvalue,im[corry][corrx][channel]) if mask[t][s] == 1 else maxvalue
                    if channels == 1:
                        erodedim[y][x] = maxvalue
                    else:
                        erodedim[y][x][channel] = maxvalue
            
    return np.uint8(erodedim)
