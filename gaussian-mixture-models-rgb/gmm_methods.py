import math
import numpy
import warnings

def gaussianMixtureModel(mean, sigma, w, inputImage, data):
    """
    Gaussian mixture model use EM
    """
    width, height = inputImage.size
    numClusters = len(mean)
    pixels = data

    """
    #1 Expectation: compute Responsivity
    """
    logRes = numpy.zeros(shape=(width, height, numClusters), dtype=float )
    for x in range(width):
        for y in range(height) :
            for k in range(numClusters):
                Xn=pixels[x][y]
                logRes[x][y][k] = w[k] * logPdf(Xn, mean[k], sigma[k])
            sum=0
            for k in range(numClusters):
                if (k == 0):
                    sum = logRes[x][y][k]
                else:
                    sum = sum +logRes[x][y][k]
            for k in range(numClusters):
                logRes[x][y][k]= logRes[x][y][k]/sum

    """
    #2 Maximization: compute new parameters
    """

    """Compute New Mean"""
    for k in range(numClusters):
        numerator=0
        denominator=0
        for x in range(width):
            for y in range(height):
                numerator=numerator+pixels[x][y]*float(logRes[x][y][k])
                denominator=denominator+float(logRes[x][y][k])
        if denominator == 0:
            mean[k]=mean[k]
        else:
            mean[k] = numerator/float(denominator)

    """
    Compute new variance
    """
    arry = numpy.zeros(shape=(3, 3), dtype=float)
    for k in range(numClusters):
        numk = 0
        denk = 0
        for x in range(width):
            for y in range(height):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    arr = numpy.array(pixels[x][y] - mean[k])
                    for i in range(3):
                        for j in range(3):
                            arry[i][j] = arr[i]*arr[j]
                    numk =numk + (float(logRes[x][y][k])) * arry
                denk += float(logRes[x][y][k])

        if denk==0:
            sigma[k] = sigma[k]
        else:
            sigma[k]=(numk)/float(denk)
    """
    Compute new mixing coefficients
    """
    w = []
    for k in range(numClusters):
        numerator=0
        for x in range(width):
            for y in range(height):
                numerator=numerator+float(logRes[x][y][k])
        w.append(float(numerator)/(float(pixels.size/3)))
    
    """
    show new parameter
    """
    print "\nNew mean  : " , mean
    print "\nNew sigma  : " , sigma
    print "\nNew w  : " ,w

    """
    Update image and show
    """
    renderImage = inputImage
    for x in range(width):
            for y in range(height):
                logRes[x][y] = numpy.exp(logRes[x][y])
                highestKIndex = numpy.argmax(logRes[x][y])
                clor = mean[highestKIndex]*255
                r,g,b = int(clor[0]),int(clor[1]),int(clor[2])
                newclor = []
                newclor.append(r)
                newclor.append(g)
                newclor.append(b)
                newclor = tuple(newclor)
                renderImage.putpixel((x, y),(newclor))
    renderImage.show()
    renderImage.save("result_.jpg", "JPEG")
    return [mean, sigma, w, inputImage]

def logPdf(x, mean, sigma):
    """
    PDF function
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result1 = ((-0.5)*(math.log(((2*math.pi)**3)*((numpy.linalg.det(sigma))))))
            result2 = numpy.dot((x-mean),(numpy.linalg.inv(sigma)))
            result3 = (-0.5)*numpy.dot(result2,(x-mean).T)
            result = math.exp(result1 +result3)
            return result
        except ValueError:
            return 0

        

