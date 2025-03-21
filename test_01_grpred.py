import evaltool, localtests

from grpred.team import bots

name= "red"
evaltool= evaltool.Eval( f"{name}-test", 10 )
challengers= {}

def test_solo_loadBots():
    global name, evaltool, challengers
    challengers= localtests.testLoadBots(name, evaltool, bots)
    assert list(challengers.keys()) == ['red-0', 'red-1']

def test_solo_bot_0():
    global name, evaltool, challengers
    print( challengers )
    challengers= localtests.testBot(name, evaltool, challengers, 0)

def test_solo_bot_1():
    global name, evaltool, challengers
    challengers= localtests.testBot(name, evaltool, challengers, 1)

def test_solo_small():
    global name, evaltool, challengers
    evaltool.report( f"\n## Solo Small :\n" )
    challengers= localtests.testSmall(name, evaltool, challengers)

def test_solo_smallBis():
    global name, evaltool, challengers
    challengers= localtests.testSmall(name, evaltool, challengers)

def test_solo_medium():
    global name, evaltool, challengers
    evaltool.report( f"\n## Solo Medium :\n" )
    challengers= localtests.testMedium(name, evaltool, challengers)

def test_solo_large():
    global name, evaltool, challengers
    evaltool.report( f"\n## Solo Large :\n" )
    challengers= localtests.testLarge(name, evaltool, challengers)

def test_vip_small():
    global name, evaltool, challengers
    evaltool.setVip(1)
    evaltool.report( f"\n## Vip Small :\n" )
    challengers= localtests.testSmall(name, evaltool, challengers)

def test_vip_medium():
    global name, evaltool, challengers
    evaltool.setVip(1)
    evaltool.report( f"\n## Vip Medium :\n" )
    challengers= localtests.testMedium(name, evaltool, challengers)

def test_vip_large():
    global name, evaltool, challengers
    evaltool.setVip(1)
    evaltool.report( f"\n## Vip Large :\n" )
    challengers= localtests.testLarge(name, evaltool, challengers)

def test_final_bots():
    global name, evaltool, challengers
    evaltool.report( f"\n## Conclusion :\n\nChallengers: { ', '.join(challengers.keys()) }" )
    evaltool.mergeLogs(name)
    assert len(challengers) >= 1
