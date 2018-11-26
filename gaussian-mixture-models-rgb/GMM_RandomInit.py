from gmm_methods import *
from PIL import Image

numClusters = 5
inputImage = Image.open("5.jpeg")
inputImage = inputImage.resize((70,70))
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

"""
Pick random values from image as initial means 

"""
mean = numpy.zeros(shape=(numClusters, 3), dtype=float)
x = numpy.random.randint(0, width, size=numClusters)
y = numpy.random.randint(0, height, size=numClusters)
for i in range(numClusters):
    mean[i] = (data[x[i]][y[i]])

"""
Pick initial variance values of 0.1 for each cluster
"""
sigma = numpy.zeros(shape=(numClusters,3,3), dtype=float)
for i in xrange(numClusters):
    sigma[i] = numpy.identity(3)
sigma= sigma/10

"""
Pick equal mixing coefficients of 1/numcluster to start with
"""
w = [1/float(numClusters) for x in range(numClusters)]

print("\n======Running Expectation Maximization Algorithm=======")
print "\nRandom initial mean values from image: " + str(mean)
print "\nInitial variance values : " + str(sigma) 
print "\nInitial mixing coefficients : " + str(w)

stats = [mean, sigma, w, inputImage]

while(True):
    stats = gaussianMixtureModel(stats[0], stats[1], stats[2], stats[3], data)
