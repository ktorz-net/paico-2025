from hacka.games.moveit import GameEngine
import random
import time

class FirstBot():
    # Player interface :
    def wakeUp(self, playerId, numberOfPlayers, gameConfiguration ):
        self._id= playerId
        self._model= GameEngine()
        self._model.fromPod(gameConfiguration)  # Load the model from gameConfiguration
        self._model.render()

        s = self._model.map().size()
        self._distances = [[i for i in range(s + 1)]]
        print(self._distances)

        for i in range(1, s+1):
            dist = self.computeDistances(i)
            print(dist)
            self._distances.append(dist)

    def perceive(self, gameState ):
        self._model.setOnState(gameState)
        self._model.render()
        time.sleep(0.2)

    def decide(self):
        robot_id = 1
        msg= f'tic-{ self._model.tic() } | score { self._model.score(self._id) }'
        r1Position= self._model.mobilePosition(self._id, robot_id)
        r1Mission = self._model.mobileMission(self._id, robot_id)
        dirs= self._model.map().clockBearing(r1Position)
        dirs.remove(0)
        msg+= f' | postion {r1Position} and actions {dirs}'

        activable_missions = self.missionOn(r1Position)
        msg += f" and missions {activable_missions}"
        
        # Logging
        print( msg )

        # Decide
        if r1Mission != 0:
            mission = self._model.mission(r1Mission)
            if r1Position == mission.final:
                return f"mission {robot_id} {r1Mission}"
            else:
                (next_move, next_tile) = self.moveToward(r1Position, mission.final)
                return f"move {robot_id} {next_move}"
        
        if len(activable_missions) > 0:
            return f"mission {robot_id} {activable_missions[0]}"

        closest_mission = self.get_closest_mission(r1Position)
        if closest_mission is not None:
            (next_move, next_tile) = self.moveToward(r1Position, closest_mission.start)
            return f"move {robot_id} {next_move}"
        else:
            (next_move, next_tile) = self.moveToward(r1Position, int(self._model.map().size() / 2))
            return f"move {robot_id} {next_move}"

        self._move= random.choice(dirs)
        msg+= f' > move {self._move}'
        return f"move 1 {self._move}"

    def sleep(self, result):
        print( f"end on : {result}" )

    # Mon implementation
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
    
    def get_closest_mission(self, i_from):
        i= 1
        result=None
        all_free_missions = self._model.missions()
        if len(all_free_missions) == 0:
            return result
        
        result=all_free_missions[0]
        min_distance = self._distances[i_from][all_free_missions[0].start]
        for m in all_free_missions :
            if m.owner != 0:
                continue

            currentDistance = self._distances[i_from][m.start]
            if currentDistance < min_distance:
                result=m
            i+= 1
        
        return result

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