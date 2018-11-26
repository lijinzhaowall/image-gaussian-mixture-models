from gmm_methods import *


def getKMeans(inputImage, mean, data):
    width, height = inputImage.size
    numclusters = len(mean)
    pixels = data
    pixCoord = numpy.empty(shape=(width,height,2,3))
    """compute ditance and sort"""
    for x in range(width):
        for y in range(height):
            closest_indexs = numpy.zeros(shape=(numclusters), dtype=float)
            pixCoord[x][y][0]=pixels[x][y]
            Xn=pixCoord[x][y][0]
            for i in xrange(numclusters):
                closest_indexs[i] = numpy.linalg.norm(Xn-mean[i])#euclidean distance
            closest_index = closest_indexs.argmin()
            pixCoord[x][y][1]=closest_index

    """update new mean"""
    for k in range(numclusters):
        current_mean_array = []
        for x in range(width):
            for y in range(height):
                if (pixCoord[x][y][1][0]==k):
                    current_mean_array.append(pixCoord[x][y][0])
        current_mean_array=numpy.asarray(current_mean_array)
        mean[k] = numpy.mean(current_mean_array)
    return mean   
                
def runKMeans(inputImage, numClusters, data):
    width, height = inputImage.size
    data = data
    mean = numpy.zeros(shape=(numClusters, 3), dtype=float)
    x = numpy.random.randint(0, width, size=numClusters)
    y = numpy.random.randint(0, height, size=numClusters)
    for i in range(numClusters):
        mean[i] = (data[x[i]][y[i]])

    print("\n======Running K-Means Algorithm=======")
    print "Randomly selecting initial means = " + str(mean)
    iterations = 5
    i = 0
    while (True):
        mean = getKMeans(inputImage, mean, data)
        print "\nIteration " + str(i + 1) + " : means = " + str(mean)
        i += 1
        if i == iterations:
            break
    return mean
