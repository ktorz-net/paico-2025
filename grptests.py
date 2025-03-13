import evaltool

def testLoadBots( name, evaltool, bots):
    botInstances= bots()
    botOks= [True for b in botInstances ]
    nbOfBots= len(botInstances)
    challengers= evaltool.initChallengers( [bots], [name], [botOks] )
    assert len( challengers ) == nbOfBots
    challengers= evaltool.testChallengers( challengers, "small-1", 3, 3.0 )
    assert len( challengers ) == nbOfBots

    evaltool.report( f"\n challengers: { list(challengers.keys()) }\n\n" )
    
    return challengers

def testBot( name, evaltool, challengers, numBot ):
    print( f">> testBot({ name, evaltool, challengers, numBot })" )
    botname= f"{name}-{numBot}"

    evaltool.report( f"- test: {botname}\n" )

    team, index = challengers[botname]
    challengers.pop(botname)
    bot= team()[index]
    results, duration= evaltool.launchSoloGame( botname, bot, "small-1", 10 )
    evaltool.report( f"  * small-1: ({duration})\n" )
    assert len( results ) == 10
    assert duration < 3.0

    results, duration= evaltool.launchSoloGame( botname, bot, "medium-2", 10 )
    evaltool.report( f"  * medium-2: ({duration})\n" )
    assert len( results ) == 10
    assert duration < 6.0
    results, duration= evaltool.launchSoloGame( botname, bot,"medium-3", 10 )
    evaltool.report( f"  * medium-3: ({duration})\n" )
    assert len( results ) == 10
    assert duration < 6.0
    results, duration= evaltool.launchSoloGame( botname, bot,"large-1", 10 )
    evaltool.report( f"  * large-1: ({duration})\n" )
    assert len( results ) == 10
    assert duration < 12.0
    results, duration= evaltool.launchSoloGame( botname, bot,"large-2", 10 )
    evaltool.report( f"  * large-2: ({duration})\n" )
    assert len( results ) == 10
    assert duration < 12.0
    challengers[botname]= (team, index)
    
    return challengers


def testSmall( name, evaltool, challengers ):
    nbOfBots= len( challengers )
    challengers= evaltool.testChallengers( challengers, "small-1", 3 )
    assert len( challengers ) == nbOfBots
    challengers= evaltool.testChallengers( challengers, "small-2", 3 )
    assert len( challengers ) == nbOfBots
    challengers= evaltool.testChallengers( challengers, "small-3", 3 )
    assert len( challengers ) == nbOfBots

    challengers= evaltool.testChallengers( challengers, "small-1", 10 )
    assert len( challengers ) == nbOfBots
    challengers= evaltool.testChallengers( challengers, "small-2", 10 )
    assert len( challengers ) == nbOfBots
    challengers= evaltool.testChallengers( challengers, "small-3", 10 )
    assert len( challengers ) == nbOfBots
    
    return challengers

def testMedium( name, evaltool, challengers ):
    nbOfBots= len( challengers )
    challengers= evaltool.testChallengers( challengers, "medium-1", 3, 3.0 )
    assert len( challengers ) == nbOfBots
    challengers= evaltool.testChallengers( challengers, "medium-2", 3, 3.0 )
    assert len( challengers ) == nbOfBots
    challengers= evaltool.testChallengers( challengers, "medium-3", 3, 3.0 )
    assert len( challengers ) == nbOfBots
    challengers= evaltool.testChallengers( challengers, "medium-4", 3, 3.0 )
    assert len( challengers ) == nbOfBots

    challengers= evaltool.testChallengers( challengers, "medium-1", 10, 3.0 )
    assert len( challengers ) == nbOfBots
    challengers= evaltool.testChallengers( challengers, "medium-2", 10, 3.0 )
    assert len( challengers ) == nbOfBots
    challengers= evaltool.testChallengers( challengers, "medium-3", 10, 3.0 )
    assert len( challengers ) == nbOfBots
    challengers= evaltool.testChallengers( challengers, "medium-4", 10, 3.0 )
    assert len( challengers ) == nbOfBots

    return challengers

def testLarge( name, evaltool, challengers ):
    nbOfBots= len( challengers )
    challengers= evaltool.testChallengers( challengers, "large-1", 3, 6.0 )
    assert len( challengers ) == nbOfBots
    challengers= evaltool.testChallengers( challengers, "large-2", 3, 6.0 )
    assert len( challengers ) == nbOfBots
    challengers= evaltool.testChallengers( challengers, "large-3", 3, 6.0 )
    assert len( challengers ) == nbOfBots
    challengers= evaltool.testChallengers( challengers, "large-4", 3, 6.0 )
    assert len( challengers ) == nbOfBots

    challengers= evaltool.testChallengers( challengers, "large-1", 10, 6.0 )
    assert len( challengers ) == nbOfBots
    challengers= evaltool.testChallengers( challengers, "large-2", 10, 6.0 )
    assert len( challengers ) == nbOfBots
    challengers= evaltool.testChallengers( challengers, "large-3", 10, 6.0 )
    assert len( challengers ) == nbOfBots
    challengers= evaltool.testChallengers( challengers, "large-4", 10, 6.0 )
    assert len( challengers ) == nbOfBots

    return challengers
