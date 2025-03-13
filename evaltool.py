#!env python3
import json, time, matplotlib.pyplot as plt
import hacka.games.moveit as moveit

from grporange.grporange.team import bots as obots
from grpbleu.src.team import bots as bbots
from grpvert.team import bots as vbots

def loadLog(aFile, games=None):
    if games == None :
        games= {}
    logDsc= open( aFile, "r" )
    for line in logDsc:
        line= line[:-1]
        elts= line.split(", ")
        bot= elts[0]
        config= '-'.join( elts[1:4] )
        dur= float(elts[4])
        results= [ float(e) for e in elts[5:] ]
        games= mergeGames( games, bot, config, dur, results )
    logDsc.close()
    return games

def mergeGames( games, bot, config, dur, results ):
    if bot not in games :
        games[bot]= { config : [results, dur] }
        return games
    if config not in games[bot] :
        games[bot][config]= [results, dur]
        return games
    past= len(games[bot][config][0])
    new= len(results)
    games[bot][config][0]+= results
    games[bot][config][1]= (games[bot][config][1]*past + dur*new) / (past+new)
    return games

class Eval():
    def __init__(self, name, nbOfXps):
        self._name= name
        self._report= f"log-eval-{self._name}.md"
        self._solo= f"log-solo-{self._name}.log"
        self._nbOfXps= nbOfXps
        self._vip= 0
        reportDsc= open( self._report, "w" )
        reportDsc.close()
        logDsc= open( self._solo, "w" )
        logDsc.close()

    def setVip(self, anInt):
        self._vip= max( 0, min(anInt, 1))

    # report :
    def report(self, lines ):
        reportDsc= open( self._report, "a" )
        reportDsc.write( lines )
        reportDsc.close()
    
    def reportResults(self, botName, results, duration ):
        nbOfXps= len( results )
        rmin, rmax= min( results ), max( results)
        rave= round( sum(results)/nbOfXps, 3 )
        self.report( f"{botName} | {nbOfXps} | {rmin} | {rave} | {rmax} | {duration}s |\n" )
    
    def logSoloResults(self, botName, config, results, duration ):
        line= f"{botName}, {config['name']}, {config['numberOfRobots']}, {self._vip}, {duration}, "
        line+= ', '.join( [str(r) for r in results] )
        logDsc= open( self._solo, "a" )
        logDsc.write( line+"\n" )
        logDsc.close()
    
    # Challengers :
    def initChallengers(self, teams, teamNames, okTeams ):
        self.report( "## Chalendgers\n\n" )
        challengers= {}
        for teamBots, teamName, okBots in zip( teams, teamNames, okTeams ) :
            bots= teamBots()
            self.report( f"\n{teamName}: ({ len(bots) } bots)\n" )
            for id in range( len(bots) ) :
                challengers[ f"{teamName}-{id}" ]= (teamBots, id)
        return challengers


    def testChallengers(self, challengers, config, nbOfXps, maxAveDuration= 2.0):
        okChallengers= {}
        self.report( f"\n{config} | nb | min | average | max | t \n-----------|-----|-----|-----|-----|--\n" )

        for botName in challengers :
            team, index = challengers[botName]
            bot= team()[index]
            print( f">>> {botName}" )
            results, duration= self.launchSoloGame( botName, bot, config, nbOfXps)
            self.reportResults( botName, results, duration )
            if duration <= maxAveDuration :
                okChallengers[botName]= (team, index)
        return okChallengers


    def launchSoloGame(self, botName, bot1, configFile, nbOfGames= 1, nbOfRobots= None):
        with open( f"configs/{configFile}.json" ) as file:
            config= json.load(file)
        config['name']= configFile
        if nbOfRobots != None :
            config['numberOfRobots']= nbOfRobots
        # Configure the game:
        gameEngine= moveit.GameEngine(
            matrix= config['matrix'],
            tic= config['tic'],
            numberOfPlayers= 1,
            numberOfRobots= config['numberOfRobots'],
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
        self.logSoloResults( botName, config, results[0], round( (tEnd-tStart), 3 ) )
        return results[0], round( (tEnd-tStart), 3 )

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