#!env python3
import evaltool, json

color= "red"

games= {}
games= evaltool.loadLog( f"log-solo-{color}-test.log", games )
fileContent= open( f"results-{color}.json", "w" )
json.dump( games, fileContent, indent=1 )
fileContent.close()
