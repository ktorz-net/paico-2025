#!env python3
import json, matplotlib.pyplot as plt

# Parramétre de l'expérience
configs= [ ("medium-2", 0 ), ("large-4", 0) ]

teams= ["red", "orange", "bleu", "vert" ]
colors= ["red", "orange", "blue", "green" ]

def getresults(color, config, vip):
    results= {}
    fileContent= open(f"results-{color}.json")
    games= json.load(fileContent)
    fileContent.close()
    for bot in games :
        averages= [-10 for i in range(1, 9)]
        ok= True
        for i in range(1, 9) :
            configKey= f"{config}-{i}-{vip}"
            if configKey in games[bot] :
                listValues= games[bot][configKey][0]
                averages[i-1]= sum(listValues)/len(listValues)
            else:
                ok= False
        if ok :
           results[bot]= averages 
    return results

for configName, vip in configs :
    data= []
    for t in teams :
        data.append( getresults(t, configName, vip) )

    maxValue= 100
    for resultDico, color in zip(data, colors) :
        for bot in resultDico :
            maxValue= max( maxValue, max(resultDico[bot]) )
            plt.plot( 
                range(1,9),
                resultDico[bot],
                color=color )
    plt.show()

    maxValue= round( maxValue/100 ) * 100

    plt.ylim(-200, maxValue)
    plt.grid(True)
    
    for resultDico, color in zip(data, colors) :
        for bot in resultDico :
            plt.plot( 
                range(1,9),
                resultDico[bot],
                color=color )
    plt.show()
