from hacka.games.moveit import GameEngine

import random#, time

class BlancBot():
    def __init__(self, name= "0"):
        self._name= name
    
    # Player interface :
    def wakeUp(self, playerId, numberOfPlayers, gameConfiguration ):
        pass

    def perceive(self, state ):
        pass

    def decide(self):
        return "pass"

    def sleep(self, result):
        print( f"end on : {result}" )

class GhostBot():
    def __init__(self, botGen= "0"):
        self._gen= botGen
    
    # Player interface :
    def wakeUp(self, playerId, numberOfPlayers, gameConfiguration ):
        self._instance= self._gen()
        self._instance.wakeUp(playerId, numberOfPlayers, gameConfiguration)

    def perceive(self, state ):
        self._instance.perceive(state)

    def decide(self):
        return self._instance.decide()

    def sleep(self, result):
        self._instance.sleep(result)

class VoidBot(BlancBot):
    # Player interface :
    def wakeUp(self, playerId, numberOfPlayers, gameConfiguration ):
        self._model= GameEngine()
        self._model.fromPod(gameConfiguration)
        self._id= playerId
        self._model.render()
        # Compute distances
        s= self._model.map().size()
        self._distances= [ [ i for i in range(s+1) ] ]
        for i in range( 1, s+1 ) :
            dist= self.computeDistances(i)
            self._distances.append( dist )

    def perceive(self, state ):
        self._model.setOnState(state)

    def decide(self):
        #print( f"--- Player{self._id} rendering..." )
        #self._model.render()
        #time.sleep(0.33)
        action= self.randomDecide()
        #print( f"--- Player{self._id} do: {action}" )
        return action

    def randomDecide(self):
        # Get information:
        r1Position= self._model.mobilePosition(self._id, 1)
        r1Mission= self._model.mobile(self._id, 1).mission()
        miss= self.missionOn(r1Position)
        dirs= self._model.map().clockBearing(r1Position)
        dirs.remove(0)

        # Action construction:
        if r1Mission != 0 and r1Position == self._model.mission( r1Mission ).final :
            return f"mission 1 {r1Mission}"

        if r1Mission == 0 and len(miss) > 0 :
            return f"mission 1 {miss[0]}"
        
        return f"move 1 {random.choice(dirs)}"

    def sleep(self, result):
        pass
        #print( f"end on : {result}" )

    # Tools: 
    def missionOn(self, iTile):
        i= 1
        l= []
        for m in self._model.missions() :
            if m.start == iTile :
                l.append(i)
            i+= 1
        return l

    def computeDistances(self, iTile):
        # Initialize distances to 0:
        dists= [iTile] +  [0 for i in range( self._model.map().size() )]
        # Initialize step from iTile:
        ringNodes= self._model.map().neighbours(iTile)
        ringDistance= 1
        # while theire is nodes to visit
        while len(ringNodes) > 0 :
            nextNodes= []
            # Visit all step nodes:
            for node in ringNodes :
                # Update distance information
                dists[node]= ringDistance
            for node in ringNodes :
                # Search for new tile to visit:
                neighbours= self._model.map().neighbours(node)
                for candidate in neighbours :
                    if dists[candidate] == 0 :
                         nextNodes.append(candidate)
            # swith to the next step.
            ringNodes= nextNodes
            ringDistance+= 1
        # Correct 0 distance:
        dists[iTile]= 0
        return dists

    def moveToward(self, iTile, iTarget):
        # If no need to move:
        if iTile == iTarget :
            return 0, iTile
        # Get candidates:
        clockdirs= self._model.map().clockBearing(iTile)
        nextTiles= self._model.map().neighbours(iTile)
        selectedDir= clockdirs[0]
        selectedNext= nextTiles[0]
        # Test all candidates:
        for clock, tile in zip( clockdirs, nextTiles ) :
            if self._distances[tile][iTarget] < self._distances[selectedNext][iTarget] :
                selectedDir= clock
                selectedNext= tile
        # Return the selected candidates:
        return selectedDir, selectedNext

    def path(self, iTile, iTarget):
        clock, tile= self.moveToward(iTile, iTarget)
        move= [clock]
        path= [tile]
        while tile != iTarget :
            clock, tile= self.moveToward(tile, iTarget)
            move.append( clock )
            path.append( tile )
        return move, path

class FirstBot():
    # Player interface :
    def wakeUp(self, playerId, numberOfPlayers, gameConfiguration ):
        self._id= playerId
        self._model= GameEngine()
        self._model.fromPod(gameConfiguration)  # Load the model from gameConfiguration
        self._model.render()                    # Draw the game image in shot-moveIt.png
        #time.sleep(0.5)
        # Compute distances:
        self._distances= [[i for i in range( self._model.map().size()+1 )]]
        for i in range( 1, len(self._distances[0]) ) :
            self._distances.append( self.computeDistances(i) )

    def perceive(self, gameState ):
        self._model.setOnState(gameState)        # Update the model sate        self._model= GameEngine()
        self._model.render()                    # Draw the game image in shot-moveIt.png
        #time.sleep(0.2)

    def decide(self):
        r1Position= self._model.mobilePosition(self._id, 1)
        # Do mission actions:
        missionId= self._model.mobile(1, 1).mission()
        missions= self.missionOn( r1Position )
        if missionId != 0 :
            if r1Position == self._model.mission( missionId ).final :
                return f"mission 1 {missionId}"
        elif len(missions) > 0 :
            return f"mission 1 {missions[0]}"
        # Else, move:
        dirs= self._model.map().clockBearing(r1Position)
        dirs.remove(0)
        self._move= random.choice(dirs)
        return f"move 1 {self._move}"

    def sleep(self, result):
        pass#print( f"end on : {result}" )

    ## Tools: 
    def missionOn(self, iTile):
        i= 1
        l= []
        for m in self._model.missions() :
            if m.start == iTile :
                l.append(i)
            i+= 1
        return l
    
    def computeDistances(self, iTile):
        # Initialize distances to 0:
        dists= [iTile] +  [0 for i in range( self._model.map().size() )]
        # Initialize step from iTile:
        ringNodes= self._model.map().neighbours(iTile)
        ringDistance= 1
        # while theire is nodes to visit
        while len(ringNodes) > 0 :
            nextNodes= []
            # Visit all step nodes:
            for node in ringNodes :
                # Update distance information
                dists[node]= ringDistance
            for node in ringNodes :
                # Search for new tile to visit:
                neighbours= self._model.map().neighbours(node)
                for candidate in neighbours :
                    if dists[candidate] == 0 :
                         nextNodes.append(candidate)
            # swith to the next step.
            ringNodes= nextNodes
            ringDistance+= 1
        # Correct 0 distance:
        dists[iTile]= 0
        return dists

    def moveToward(self, iTile, iTarget):
        # If no need to move:
        if iTile == iTarget :
            return 0, iTile
        # Get candidates:
        directions= self._model.map().clockBearing(iTile)
        nextTiles= self._model.map().neighbours(iTile)
        selectedDir= directions[0]
        selectedNext= nextTiles[0]
        # Test all candidates:
        for direction, tile in zip( directions, nextTiles ) :
            if self._distances[tile][iTarget] < self._distances[selectedNext][iTarget] :
                selectedDir= direction
                selectedNext= tile
        return selectedDir, selectedNext

    def path(self, iTile, iTarget):
        dir, tile= self.moveToward(iTile, iTarget)
        move= [dir]
        path= [tile]
        while tile != iTarget :
            dir, tile= self.moveToward(tile, iTarget)
            move.append( dir )
            path.append( tile )
        return move, path