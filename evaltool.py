#!env python3
import os, json, time, matplotlib.pyplot as plt
import hacka.games.moveit as moveit

from grporange.grporange.team import bots as obots
from grpbleu.src.team import bots as bbots
from grpvert.team import bots as vbots

def drawBotResuls(self, botResults, configName ) :
    plt.ylim([-200,500])
    for botName in botResults :
        botColor= botName.split("-")[0]
        plt.plot( botResults[botName], color=botColor )
    plt.savefig( f"./log-resources/{configName}.png" )
    plt.clf()

def loadLog(aFile, color, games=None):
    if games == None :
        games= {}
    logDsc= open( aFile, "r" )
    for line in logDsc:
        line= line[:-1]
        elts= line.split(", ")
        bot= elts[0]
        if bot.split('-')[0] == color :
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
    def __init__(self, teamName, nbOfXps, evalTag='test'):
        self._name= teamName
        self._report= f"auto-{evalTag}-{self._name}.md"
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
    
    def logDuoResults(self, botName1, botName2, config, results, duration ):
        line1= f"{botName1}, {botName2}, {config['name']}, {config['numberOfRobots']}, {self._vip}, {duration}, "
        line1+= ', '.join( [str(r) for r in results[0]] )
        line2= f"{botName2}, {botName1}, {config['name']}, {config['numberOfRobots']}, {self._vip}, {duration}, "
        line2+= ', '.join( [str(r) for r in results[1]] )
        logDsc= open( "log-duo-games.log", "a" )
        logDsc.write( line1+"\n" )
        logDsc.write( line2+"\n" )
        logDsc.close()
    
    def mergeLogs(self, color):
        games= {}
        if os.path.isfile(f"results-{color}.json" ) :
            fileContent= open(f"results-{color}.json")
            games= json.load(fileContent)
            fileContent.close()

        games= loadLog( self._solo, color, games )
        fileContent= open( f"results-{color}.json", "w" )
        json.dump( games, fileContent, indent=1 )
        fileContent.close()

    # Challengers :
    def initChallengers(self, teams, teamNames, okTeams ):
        self.report( "## Chalendgers\n\n" )
        challengers= {}
        for teamBots, teamName, okBots in zip( teams, teamNames, okTeams ) :
            bots= teamBots()
            self.report( f"\n{teamName}: ({ len(bots) } bots)\n" )
            for id in range( len(bots) ) :
                if okBots[id] :
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

    def increasingTeam(self, challengers, configFile):
        botResults= {}
        for botName in challengers :
            team, index = challengers[botName]
            botResults[botName]= []
            for nbBots in range(1, 9) :
                print( f">>>> {botName} : {nbBots}" )
                results, duration= self.launchSoloGame(
                    botName,
                    team()[index],
                    configFile,
                    self._nbOfXps,
                    nbBots
                )
                botResults[botName].append( round( sum(results)/self._nbOfXps, 3 ) )

    # Multiplayer: 
    def testConfront( self, challengers, botName1, botName2, configName, nbOfGames, maxDuration ):
        self.report( f"\n\n{configName} | nb | min | average | max | t \n-----------|-----|-----|-----|-----|--\n" )
        
        team1, index1= challengers[botName1]
        team2, index2= challengers[botName2]
        results, duration= self.launchDuoGame(
            botName1, team1()[index1],
            botName2, team2()[index2],
            configName, nbOfGames
        )

        self.reportResults( botName1, results[0], duration )
        self.reportResults( botName2, results[1], duration )
        assert duration < (maxDuration*2)
        return sum(results[0])
    
    def launchDuoGame(self, botName1, bot1, botName2, bot2, configFile, nbOfGames= 1, nbOfRobots= None):
        with open( f"configs/{configFile}.json" ) as file:
            config= json.load(file)
        config['name']= configFile
        if nbOfRobots != None :
            config['numberOfRobots']= nbOfRobots
        
        # Configure the game:
        gameEngine= moveit.GameEngine(
            matrix= config['matrix'],
            tic= config['tic'],
            numberOfPlayers= 2,
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
        results= gameMaster.launch( [bot1, bot2], nbOfGames)
        tEnd= time.perf_counter()
        self.logDuoResults( botName1, botName2, config, results, round( (tEnd-tStart), 3 ) )
        return results, round( (tEnd-tStart), 3 )
