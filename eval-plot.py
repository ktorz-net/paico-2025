#!env python3
import json, matplotlib.pyplot as plt

# Parramétre de l'expérience
configs= [ ("medium-2", 0 ), ("large-4", 0), ("large-4", 1) ]

teams= ["red", "orange", "bleu", "vert" ]
colors= ["red", "orange", "blue", "green" ]

def getResults1to8(color, config, vip):
    results= {}
    fileContent= open(f"results-{color}.json")
    games= json.load(fileContent)
    fileContent.close()
    for bot in games :
        values= [[] for i in range(1, 9)]
        ok= True
        for i in range(1, 9) :
            configKey= f"{config}-{i}-{vip}"
            if configKey in games[bot] :
                values[i-1]= games[bot][configKey][0] 
                #averages[i-1]= sum(listValues)/len(listValues)
            else:
                ok= False
        if ok :
           results[bot]= values 
    return results

def plot1to8(data, colors):
    fig, axs = plt.subplots(1, 2)

    # global plot :
    maxValue= 100
    axs[0].grid(True)
    for botResults, color in zip(data, colors) :
        for bot in botResults :
            averages= [ sum(listValues)/len(listValues)
                for listValues in botResults[bot]
            ]
            maxValue= max( maxValue, max(averages) )
            axs[0].plot( 
                range(1,9),
                averages,
                color=color
            )
    maxValue= round( maxValue/100 ) * 100

    # Positive plot :
    axs[1].set_ylim([-100, maxValue])
    axs[1].grid(True)
    for botResults, color in zip(data, colors) :
        for bot in botResults :
            averages= [ sum(listValues)/len(listValues)
                for listValues in botResults[bot]
            ]
            axs[1].plot( 
                range(1,9),
                averages,
                color=color
            )
    # Show :    
    fig.suptitle( f"{configName} VIP({vip})" )
    fig.set_figwidth(15)
    fig.set_figheight(6)

def plotBox124(data, teamColors):
    fig, axs = plt.subplots(3, 2)

    line= 0
    for nbOfBots in [1, 2, 4] :
        # global plot :
        dataToPlot= []
        labels= []
        colors= []
        maxValue= 100
        for botResults, color in zip(data, teamColors) :
            for bot in botResults :
                listOfValues= botResults[bot][nbOfBots-1]
                maxValue= max( maxValue, max(listOfValues) )
                dataToPlot.append( listOfValues )
                labels.append( bot )
                colors.append( color )
        maxValue= round( 1 + maxValue/100 ) * 100

        bplot = axs[line, 0].boxplot(dataToPlot,
                        patch_artist=True,  # fill with color
                        tick_labels=labels)  # will be used to label x-ticks
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        # Positive plot :
        axs[line, 1].set_ylim([-100, maxValue])
        bplot = axs[line, 1].boxplot(dataToPlot,
                        patch_artist=True,  # fill with color
                        tick_labels=labels)  # will be used to label x-ticks
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        line+= 1

    # Show :    
    fig.suptitle( f"{configName} vip({vip}) nbOfRobots(1, 2 and 4)" )
    fig.set_figwidth(15)
    fig.set_figheight(12)

for configName, vip in configs :
    # Get data :
    data= []
    for t in teams :
        data.append( getResults1to8(t, configName, vip) )

    plot1to8(data, colors)
    plt.show()

    plotBox124(data, colors)
    plt.show()


