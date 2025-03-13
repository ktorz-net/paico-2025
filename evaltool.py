#!env python3
import json, time, matplotlib.pyplot as plt
import hacka.games.moveit as moveit

from grporange.grporange.team import bots as obots
from grpbleu.src.team import bots as bbots
from grpvert.team import bots as vbots

class Eval():
    def __init__(self, name, nbOfXps):
        self._name= name
        self._filePath= f"log-eval-{self._name}.md"
        self._nbOfXps= nbOfXps
        self._vip= 0
        logDsc= open( self._filePath, "w" )
        logDsc.close()

    def setVip(self, anInt):
        self._vip= max( 0, min(anInt, 1))

    def initChallengers(self, config, teams, teamNames, okTeams, maxDuration= 1.0 ):
        self.write( "## Chalendgers\n\n" )
        challengers= {}
        for teamBots, teamName, okBots in zip( teams, teamNames, okTeams ) :
            bots= teamBots()
            self.write( f"\n{teamName}: ({ len(bots) } bots)\n" )
            for id in range( len(bots) ) :
                if okBots[id] :
                    results, duration= self.startGame( config, bots[id])
                    self.write( f" - duration ({config}): {duration}s ({results[0]})\n" )
                    if duration < 1.0 :
                        challengers[ f"{teamName}-{id}" ]= (teamBots, id)
        return challengers

    def startGame(self, configFile, bot1, nbOfGames= 1, nbOfRobots= None):
        with open( f"configs/{configFile}.json" ) as file:
            config= json.load(file)
        if nbOfRobots == None :
            nbOfRobots= config['numberOfRobots']
        # Configure the game:
        gameEngine= moveit.GameEngine(
            matrix= config['matrix'],
            tic= config['tic'],
            numberOfPlayers= 1,
            numberOfRobots= nbOfRobots,
            numberOfPVips= self._vip
        )
        # Then Go...
        gameMaster= moveit.GameMaster(
            gameEngine,
            randomMission= config['numberOfMissions'],
            vipZones= config['vipZones']
        )
        tStart= time.perf_counter()
        results= gameMaster.launch( [bot1], nbOfGames)
        tEnd= time.perf_counter()
        return results[0], round( (tEnd-tStart), 3 )

    def write(self, lines ):
        logDsc= open( self._filePath, "a" )
        logDsc.write( lines )
        logDsc.close()

    def log(self, config, results, duration ):
        logDsc= open( self._filePath, "a" )
        nbOfGames= len(results)
        rmin, rmax= min( results ), max( results)
        rave= round( sum(results)/nbOfGames, 3 )
        logDsc.write( f"{config} | {rmin} | {rave} | {rmax} | {round(duration/nbOfGames, 3)}s\n" )
        logDsc.close()

    def bench(self, challengers, configList, vip=0):
        for config in configList :
            logDsc= open( self._filePath, "a" )
            logDsc.write( f"\n{config} | min | average | max | t \n" )
            logDsc.write( f"-----------|-----|-----|-----|-----\n" )
            logDsc.close()
            for botName in challengers :
                team, index = challengers[botName]
                bot= team()[index]
                print( f">>> {botName}" )
                results, duration= self.startGame( config, bot, self._nbOfXps)
                self.log( botName, results, duration )

    def increasingTeam(self, challengers, config):
        configName= self._name + config

        logDsc= open( self._filePath, "a" )
        logDsc.write( f"\n![](./log-resources/{configName}.png)\n\n" )
        logDsc.close()
            
        botResults= {}
        for botName in challengers :
            team, index = challengers[botName]
            botResults[botName]= []
            for nbBots in range(1, 9) :
                results, duration= self.startGame( config, team()[index], self._nbOfXps, nbBots)
                botResults[botName].append( round( sum(results)/self._nbOfXps, 3 ) )

            logDsc= open( self._filePath, "a" )
            logDsc.write( f"- {botName}: {botResults[botName]}\n" )
            logDsc.close()
            self.drawBotResuls(botResults, configName)

    def drawBotResuls(self, botResults, configName ) :
        plt.ylim([-200,500])
        for botName in botResults :
            botColor= botName.split("-")[0]
            plt.plot( botResults[botName], color=botColor )
        plt.savefig( f"./log-resources/{configName}.png" )
        plt.clf()