#2018/11/26  lijinzhao 
from PIL import Image
from kmeans_methods import *

numClusters = 5
inputImage = Image.open("5.jpeg")
inputImage = inputImage.resize((100,100))
width, height = inputImage.size

data = numpy.zeros(shape=(width, height,3), dtype=float)
for i in xrange(width):
    for j in xrange(height):
        tem = []
        r,g,b =inputImage.getpixel((i,j))
        tem.append(r/255.0)
        tem.append(g/255.0)
        tem.append(b/255.0)
        data[i][j]=tem

# k-mean compute mean
mean = runKMeans(inputImage, numClusters, data)

#compute signa
sigma = numpy.zeros(shape=(numClusters,3,3), dtype=float)
for i in xrange(numClusters):
    sigma[i] = numpy.identity(3)
sigma= sigma/10

#compute mixing coefficients
w = [1/float(numClusters) for x in range(numClusters)] 


stats = [mean, sigma, w, inputImage]
print("\n\n======Running Expectation Maximization Algorithm=======")
print "\nInitial mean values using K-means: " + str(mean)
print "\nInitial variance values : " + str(sigma) 
print "\nInitial mixing coefficients : " + str(w)

iterations = 100
i = 0
while(True):
    stats = gaussianMixtureModel(stats[0], stats[1], stats[2], stats[3], data)
    i += 1
    if i == 100:
        break