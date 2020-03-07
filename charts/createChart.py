import matplotlib.pyplot as plt
import numpy as np

def makeChartFromFile(fileList):

    for file in fileList:
        [time, numberOfSteps, steps] = np.load(file, allow_pickle=True)

        plt.plot(steps)
    plt.show()


if __name__ == "__main__":
    makeChartFromFile(["../results.npy", "../results2.npy"])