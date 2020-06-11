import matplotlib.pyplot as plt
import numpy as np

RESULT_PATH = '../results/RL/data/'

def makeChartFromParameters(mazeType, size, strategy, rStrategy, e, learningParameters, name = None):
    fig, ax = plt.subplots()

    if strategy != 2:
        e = 0.1
        t = 0.1

    for pathName in learningParameters:
            type = str(size) + 'x' + str(size) + '_' + mazeType
            folderName = '_'.join(str(x) for x in [strategy, rStrategy, e, e])

            file = RESULT_PATH + type + '/' + folderName + '/' + pathName + '.npy'

            [reason, time, epocheNumber, stepsNumber, _, _] = np.load(file, allow_pickle=True)

            [beta, gamma]= pathName.split("_")
            # ax.plot(stepsNumber, label="\u03B2=" + beta + " \u03B3=" + gamma)
            ax.plot(stepsNumber[-300:], label="\u03B2=" + beta + " \u03B3=" + gamma)
            ax.set_ylabel('Długość ścieżki')
            ax.set_xlabel('Liczba epok')



    ax.legend(loc='upper right', shadow=True)
    if name:
        plt.savefig('./results/' + name + '.png',)
    plt.show()

def makeChart(name, xValues,yValues, xlabel, ylabel, xTicks, labels):
    fig, ax = plt.subplots()

    for i,y in enumerate(yValues):
        ax.plot(xValues, y, label=labels[i])

    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0-0.04,
                     box.width, box.height*0.15])

    # Put a legend below current axis
    ax.legend(bbox_to_anchor=(0.5,0), loc="lower center",
                bbox_transform=fig.transFigure, ncol=5)

    plt.xticks(xValues, xTicks)

    fig.tight_layout()
    plt.savefig('./results/' + name + '.png', bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    # makeChart(
    #     'rank_10_wall',
    #     [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    #     [
    #         [4958266.119,4383119.249,3846346.473,4433273.391,5450542.177,3784928.421,3799912.354,3814409.109,3839597.278],
    #         [2695248.801,3293475.297,3767798.223,2684664.344,2147677.328,2729854.81,2700333.048,3263473.728,4915319.913],
    #         [2720003.202,2765586.324,3343090.553,3266878.309,1628221.953,2193959.56,2747532.699,2747146.25,3273166.161],
    #         [1612442.62,2174285.779,3224621.612,2718944.966,2708842.037,1611529.134,2700171.989,3266481.561,3240354.983],
    #         [4265159.713,4721158.818,4081127.856,4886760.725,4797212.327,4730908.975,4295729.544,4283572.044,4156431.165]
    #     ],
    #     'Wartość współczynnika uczenia', 'Średnia wartość wskaźnika',
    #     [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    #     ['Zachłanny','\u03B5 = 0.2', '\u03B5 = 0.4', '\u03B5 = 0.6', 'UCB-1']
    # )

    # makeChart(
    #     'rank_10_columns',
    #     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #     [
    #         [5442014.463, 5816070.371, 5977988.916, 5869562.905, 5747762.667, 6454134.946, 6442287.646, 5742614.824,
    #          5219862.164],
    #         [6678704.345, 6592983.027, 5480640.342, 4864268.816, 5412208.736, 5380318.371, 5366256.086, 4855511.927,
    #          5998131.366],
    #         [4856555.002, 4312087.923, 4328530.335, 4856105.183, 4917530.598, 3225351.096, 4870593.02, 3236729.976,
    #          3269342.405],
    #         [3217049.538, 3235828.863, 3837127.065, 3230880.956, 3265466.567, 3229877.032, 3218269.038, 3243040.021,
    #          3265198.838],
    #         [6032604.395, 6680957.714, 5442078.889, 6569221.748, 5997935.579, 5491733.656, 5970995.39, 6633276.913,
    #          5024437.327],
    #     ],
    #     'Wartość współczynnika uczenia', 'Średnia wartość wskaźnika',
    #     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #     ['Zachłanny', '\u03B5 = 0.2', '\u03B5 = 0.4', '\u03B5 = 0.6', 'UCB-1']
    # )

    # makeChart(
    #     'rank_10_board',
    #     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #     [
    #         [3853864.661, 5068171.815, 4872013.92, 6076520.657, 6540372.678, 6515210.194, 6053958.897, 6629247.002,
    #          6537534.099],
    #         [2179834.765, 4394117.134, 4411413.458, 5465626.082, 6565513.127, 6529513.648, 6552972.68, 5967930.411,
    #          6563377.66],
    #         [3243697.222, 4348546.376, 4957030.654, 6013174.258, 5498441.37, 6004288.235, 5529963.656, 5461451.716,
    #          6020848.898],
    #         [2766884.148, 1649561.064, 2779396.262, 2759109.959, 3313297.47, 2775112.453, 1646468.737, 3279233.739,
    #          2218145.441],
    #         [3925591.832, 3340742.7, 3324363.529, 6715976.976, 5692457.166, 6652944.806, 6652942.3, 6683038.192,
    #          6741233.744],
    #     ],
    #     'Wartość współczynnika uczenia', 'Średnia wartość wskaźnika',
    #     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #     ['Zachłanny', '\u03B5 = 0.2', '\u03B5 = 0.4', '\u03B5 = 0.6', 'UCB-1']
    # )

    # strategy = 4
    # rStrategy = 'bonus'
    # et = 0.2
    # size = 100
    # mazeType = 'wall'
    # learningParameters = ['0.9_0.9']
    makeChartFromParameters(
        name='egreedy_0.6_10_wall_0.9_0.5',
        mazeType='wall',
        size=10,
        strategy=2,
        rStrategy=1,
        e=0.6,
        learningParameters= ['0.9_0.5'])

    # makeChart(
    #     'time_1_4_10',
    #     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #     [
    #         [1.81873499, 0.961777449, 0.637000084, 0.480445014, 0.413776212, 0.330777433, 0.285221921, 0.251555098,
    #          0.24144419],
    #         [3.84588888, 1.76112122, 1.250607464, 1.112999598, 0.887443648, 0.714666473, 0.68198416, 0.577666468,
    #          0.566444238],
    #     ],
    #     'Wartość współczynnika uczenia', 'Czas (s)',
    #     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #     ['Zachłanny', 'UCB-1']
    # )

    # makeChartFromParameters(
    #     name='1_0.9_0.9_board_50',
    #     mazeType='board',
    #     size=50,
    #     strategy=1,
    #     rStrategy=1,
    #     e=0.6,
    #     learningParameters= ['0.9_0.9'])

    # makeChart(
    #     'time_1_1_2_10',
    #     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #     [
    #         [3.72655593, 2.071770853, 1.423111969, 1.106068293, 0.786887566, 0.709777276, 0.621443166, 0.518444618,
    #          0.445443471],
    #         [0.799613953, 0.454395453, 0.281086206, 0.226888604, 0.164778153, 0.167777485, 0.133999719, 0.133111159,
    #          0.126888222],
    #     ],
    #     'Wartość współczynnika uczenia', 'Czas (s)',
    #     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #     ['Jednolita nagroda', 'Sterowana nagroda']
    # )

    # makeChartFromParameters(
    #     name='1_0.9_0.9_board_100',
    #     mazeType='board',
    #     size=100,
    #     strategy=1,
    #     rStrategy='bonus',
    #     e=0.6,
    #     learningParameters= ['0.9_0.9'])