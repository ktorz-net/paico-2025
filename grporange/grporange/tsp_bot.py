from hacka.games.moveit import GameEngine
import random
import time
import copy

class TspBot():
    # Player interface :
    def wakeUp(self, playerId, numberOfPlayers, gameConfiguration ):
        self._id= playerId
        self._model= GameEngine()
        self._model.fromPod(gameConfiguration)  # Load the model from gameConfiguration
        #self._model.render()

        s = self._model.map().size()
        self._distances = [[i for i in range(s + 1)]]
        # # print(self._distances)

        for i in range(1, s+1):
            dist = self.computeDistances(i)
            # print(dist)
            self._distances.append(dist)

        self._path = self.tsp(self._id, 1)
        self._path = self._path[1:]
        # # print(self._path)
        # input()

    def perceive(self, gameState ):
        self._model.setOnState(gameState)
        # self._model.render()
        # time.sleep(0.2)

    def decide(self):
        robot_id = 1
        r1Position= self._model.mobilePosition(self._id, robot_id)
        r1Mission = self._model.mobileMission(self._id, robot_id)
        dirs= self._model.map().clockBearing(r1Position)
        dirs.remove(0)
        
        # Logging
        ## print( msg )

        # Decide
        if r1Mission != 0:
            mission = self._model.mission(r1Mission)
            if r1Position == mission.final:
                return f"mission {robot_id} {r1Mission}"
            else:
                (next_move, _) = self.moveToward(r1Position, mission.final)
                return f"move {robot_id} {next_move}"
        
        next_mission, _ = self.get_next_mission()
        if r1Position == next_mission:
            activable_missions = self.missionOn(r1Position)
            self._path.pop(0)
            return f"mission {robot_id} {activable_missions[0]}"

        if next_mission is not None:
            (next_move, _) = self.moveToward(r1Position, next_mission)
            return f"move {robot_id} {next_move}"
        else:
            (next_move, _) = self.moveToward(r1Position, int(self._model.map().size() / 2))
            return f"move {robot_id} {next_move}"

    def sleep(self, result):
        pass
        # # print( f"end on : {result}" )

    # Mon implementation
    def missionOn(self, iTile):
        i= 1
        l= []
        for m in self._model.missions() :
            if m.start == iTile :
                l.append(i)
                break
            i+= 1
        return l
    
    def get_next_mission(self):
        if len(self._path) > 0:
            return self._path[0]
        return None, None

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
    
    def get_closest_mission(self, i_from, all_missions_ids=None):
        result=None
        if all_missions_ids is None:
            all_missions_ids = self._model.missionsList()
        if len(all_missions_ids) == 0:
            return result
        
        current_mission = self._model.mission(all_missions_ids[0])
        result=(all_missions_ids[0], current_mission)
        min_distance = self._distances[i_from][current_mission.start]
        for m_id in all_missions_ids:
            current_mission = self._model.mission(m_id)
            if current_mission.owner != 0:
                continue

            currentDistance = self._distances[i_from][current_mission.start]
            if currentDistance < min_distance:
                result=(m_id, current_mission)
        
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
    
    def tsp(self, player_id, robot_id, duration=3):
        # # print("start tsp")
        robot_position = self._model.mobilePosition(player_id, robot_id)
        result_path=[(robot_position, robot_position)]
        
        mission_ids = [mission_id for mission_id in self._model.missionsList()]
        current_position = robot_position
        while len(mission_ids) > 0:
            closest_mission_id, closest_mission = self.get_closest_mission(current_position, mission_ids)
            result_path.append((closest_mission.start, closest_mission.final))
            current_position = closest_mission.final
            mission_ids.remove(closest_mission_id)
        

        current_score = self.calc_total_distance(result_path)

        # # print(f"first path: {result_path}")
        # # print(f"first score: {current_score}")

        start_time = time.time()
        current_time = start_time
        best_result = copy.deepcopy(result_path)
        best_score = current_score
        previous_best_score = best_score
        n = 0
        while current_time - start_time < duration:
            # Si on a trois fois le même score == la solution converge donc on sort de la boucle
            if n >= 10000:
                break
            result_path, current_score = self.swap(best_score, copy.deepcopy(best_result))
            # Si on a un meilleur score on reset le compteur de convergence
            if current_score < best_score:
                best_score = current_score
                previous_best_score = best_score
                n = 0
                best_result = copy.deepcopy(result_path)
            # Sinon si le meilleur score est le même alors on augment le compteur de convergence
            elif best_score == previous_best_score:
                n += 1
            
            current_time = time.time()
        # # print(f"best result: {best_result}")
        # # print(f"best score: {best_score}")
        # # print("fin tsp")
        # input()
        return result_path

    def calc_total_distance(self, path=[]):
        if len(path) <= 1:
            return 0
        
        total = 0
        for i in range(1, len(path)):
            _, previous_end = path[i-1]
            current_start, current_end = path[i]
            dist_prev_current = self._distances[previous_end][current_start]
            dist_current = self._distances[current_start][current_end]
            total += dist_prev_current + dist_current
        
        return total

    def swap(self, score, path=[]):
        _pos1, _pos2 = sorted(random.sample(range(1, len(path)), 2))

        are_neighbour = _pos2 - _pos1 == 1

        _, before_pos1 = path[_pos1 - 1]
        after_pos1, _ = path[_pos1 + 1]

        _, before_pos2 = path[_pos2 - 1]
        after_pos2 = path[_pos2 + 1][0] if _pos2 + 1 < len(path) else None
        
        start_pos1, end_pos1 = path[_pos1]
        start_pos2, end_pos2 = path[_pos2]
        score -= (self._distances[before_pos1][start_pos1] + self._distances[end_pos1][after_pos1])
        if after_pos2 is not None:
            score -= self._distances[end_pos2][after_pos2]

        if not are_neighbour:
            score -= self._distances[before_pos2][start_pos2]
        
        # Echange des positions
        path[_pos1], path[_pos2] = path[_pos2], path[_pos1]

        start_pos1, end_pos1 = path[_pos1]
        start_pos2, end_pos2 = path[_pos2]
        _, before_pos1 = path[_pos1 - 1]
        after_pos1, _ = path[_pos1 + 1]

        _, before_pos2 = path[_pos2 - 1]
        after_pos2 = path[_pos2 + 1][0] if _pos2 + 1 < len(path) else None

        score += self._distances[before_pos1][start_pos1] + self._distances[end_pos1][after_pos1]
        if after_pos2 is not None:
            score += self._distances[end_pos2][after_pos2]

        if not are_neighbour:
            score += self._distances[before_pos2][start_pos2] 

        return path, score
    
    # def shift(self, score, path):
    #     _pos1, _pos2 = sorted(random.sample(range(1, len(path)), 2))
    #     # print(path)
    #     # print(f"{_pos1}, {_pos2}")
    #     # print(score)
    #     # Récupération des éléments avant et après _pos1
    #     _, before_pos1 = path[_pos1 - 1]
    #     after_pos1, _ = path[_pos1 + 1] if _pos1 + 1 < len(path) else (None, None)

    #     # Récupération de la mission à déplacer
    #     start_pos1, end_pos1 = path[_pos1]

    #     # Suppression des distances liées à _pos1
    #     score -= self._distances[before_pos1][start_pos1]
    #     score -= self._distances[end_pos1][after_pos1] if after_pos1 is not None else 0
    #     score += self._distances[before_pos1][after_pos1] if after_pos1 is not None else 0

    #     # Suppression effective de _pos1
    #     removed_mission = path.pop(_pos1)

    #     # Correction de _pos2 si _pos1 était avant
    #     if _pos1 < _pos2:
    #         _pos2 -= 1  # La liste a perdu un élément avant cet index

    #     # Récupération des nouvelles références après suppression
    #     _, before_pos2 = path[_pos2 - 1]
    #     after_pos2 = path[_pos2 + 1][0] if _pos2 + 1 < len(path) else None

    #     # Suppression des anciennes distances à _pos2
    #     score -= self._distances[before_pos2][after_pos2] if after_pos2 is not None else 0

    #     # Ajout des nouvelles distances pour insérer _pos1 à _pos2
    #     score += self._distances[before_pos2][start_pos1]
    #     score += self._distances[end_pos1][after_pos2] if after_pos2 is not None else 0

    #     # Insertion du wagon déplacé à _pos2
    #     path.insert(_pos2, removed_mission)

    #     # Vérification du score recalculé globalement
    #     recalculated_score = self.calc_total_distance(path)
    #     assert abs(score - recalculated_score) < 1e-6, f"Mismatch! score={score}, recalculated={recalculated_score}"

    #     return path, score
            
    def path(self, iTile, iTarget):
        clock, tile= self.moveToward(iTile, iTarget)
        move= [clock]
        path= [tile]
        while tile != iTarget :
            clock, tile= self.moveToward(tile, iTarget)
            move.append( clock )
            path.append( tile )
        return move, path