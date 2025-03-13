from hacka.games.moveit import GameEngine
import time
import random
from hacka.games.moveit import BasicBot
    
def detect_cycle(numbers):
    """
    Détecte un cycle d'une longueur donnée dans une liste.
    Retourne le cycle si trouvé, sinon une liste vide.
    """
    n = len(numbers)
    for cycle_length in range(2, n // 2 + 1):
        candidate_cycle = numbers[-cycle_length:]
        if all(numbers[i - cycle_length:i] == candidate_cycle for i in range(n - cycle_length, n - 2 * cycle_length, -cycle_length)):
            if len(set(candidate_cycle)) == 2:
                return candidate_cycle
    return []

class MultiBot():

    # Player interface :
    def wakeUp(self, playerId, numberOfPlayers, gameConfiguration ):
        self._model= GameEngine()
        self._model.fromPod(gameConfiguration)
        self._id= playerId
        self._model.render()
        print( f"Output image : ./shot-moveIt.png" ) 
        self.initGame()
        
    def initGame(self):
        self._distances = {}
        self._numberOfMobiles = self._model.numberOfMobiles(iPlayer=self._id)
        self._numberOfPlayers = self._model.numberOfPlayers()
        map_size = self._model.map().size()
        for tile in range(1, map_size + 1):
            self._distances[tile] = self.computeDistances(tile)
        self._sumResult = 0
        self._countResult = 0
        self.paths = {}

        self._free_missions = self._model.freeMissions()
        self._missions = {robot_id: 0 for robot_id in range(1, self._numberOfMobiles + 1)}
        self.assignOptimalMissions()
        self._enemyPositions = {}


    def assignOptimalMissions(self):
        bot_positions = {bot_id: self._model.mobilePosition(self._id, bot_id) for bot_id in range(1, self._numberOfMobiles + 1)}
        remaining_missions = list(self._free_missions)
        bot_mission_distances = {bot_id: [] for bot_id in bot_positions}
        
        # Calcul des distances entre chaque bot et chaque mission
        for bot_id, bot_pos in bot_positions.items():
            for mission_id in remaining_missions:
                mission_start = self._model.mission(mission_id).start
                distance = self._distances[bot_pos][mission_start]
                bot_mission_distances[bot_id].append((distance, mission_id))
        
        # Trie les missions par distance croissante pour chaque bot
        for bot_id in bot_mission_distances:
            bot_mission_distances[bot_id].sort()

        #print("dict trié des distances : ", bot_mission_distances)

        assigned_missions = set()
        
        # Assigner les missions en fonction des distances
        for bot_id in bot_positions:
            for distance, mission_id in bot_mission_distances[bot_id]:
                if mission_id not in assigned_missions:
                    self._missions[bot_id] = mission_id
                    assigned_missions.add(mission_id)
                    break

    def minDistanceToMission(self, id_player, id_bot):
        current_tile = self._model.mobilePosition(id_player, id_bot)

        if not self._free_missions:
            return None
        
        min_mission_id = None
        best_score = float('-inf')
        
        for mission_id in self._free_missions:
            mission_obj = self._model.mission(mission_id)
            start, final, reward = mission_obj.start, mission_obj.final, mission_obj.reward

            distance_to_start = self._distances[current_tile][start]

            if distance_to_start == 0:
                return mission_id
            
            distance_start_to_end = self._distances[start][final]
            total_distance = distance_to_start + distance_start_to_end
            
            if total_distance == 0:
                continue
            
            score = reward / total_distance
            
            # Pénalité si un robot ennemi est trop proche
            for enemy_id, (enemy_pos, enemy_mission) in self._enemyPositions.items():
                
                if enemy_mission != 0 and enemy_mission != None:
                    enemy_distance = self._distances[enemy_pos][start]
                    if enemy_distance < distance_to_start:
                        score -= 10
            
            if score > best_score:
                best_score = score
                min_mission_id = mission_id

        return min_mission_id


    def perceive(self, state ):
        """
        Mise à jour de l'état du jeu à chaque tour.
        """
        self._model.setOnState(state)
        self._model.render()
        self._free_missions = self._model.freeMissions()
        self._enemyPositions = self.getEnemyPosition()

    def setMissions(self, robot_id):
        if not self._free_missions:
            return
        min_mission_id = self.minDistanceToMission(self._id, robot_id)
        if min_mission_id in self._free_missions:
            self._missions[robot_id]= min_mission_id

    def getEnemyPosition(self):
        """ Récupère la position des robots adverses et leurs missions. """
        enemy_players = [player_id for player_id in range(1, self._numberOfPlayers + 1) if self._id != player_id]
        enemy_positions = {}
        
        for player_id in enemy_players:
            for bot_id in range(1, self._model.numberOfMobiles(iPlayer=player_id) + 1):
                position = self._model.mobilePosition(player_id, bot_id)
                mission = self._model.mobileMission(player_id, bot_id)
                if mission == 0:
                    mission = self.minDistanceToMission(player_id, bot_id)
                enemy_positions[bot_id] = (position, mission)
        return enemy_positions

    def decideBot(self, idBot):
        current_tile = self._model.mobilePosition(self._id, idBot)
        current_mission = self._model.mobile(self._id, idBot).mission()
        dirs= self._model.map().clockBearing(current_tile)
        dirs.remove(0)

        msg= f'tic-{ self._model.tic() } | score { self._model.score(self._id) }'
        msg+= f' | position {current_tile} and move {dirs} mission {current_mission}'

        adjacent_tiles = self._model.map().neighbours(current_tile)
        close_enemies = [enemy_id for enemy_id, (pos, _) in self._enemyPositions.items() if pos in adjacent_tiles]
        try :
            vip_position = self._model.mobilePosition(0, 0)
        except Exception as e:
            vip_position = None
        if vip_position in adjacent_tiles:
            
            # Chercher un autre mouvement disponible qui évite le VIP
            for direction, possible_tile in zip(self._model.map().clockBearing(current_tile), self._model.map().neighbours(current_tile)):
                if possible_tile != vip_position:
                    bot_next_move, bot_next_pos = direction, possible_tile
                    return f"move {idBot} {bot_next_move}"  


        if close_enemies:
            return "pass"

        missions_bot = self._missions[idBot]
        bot_next_move, bot_next_pos = self.moveToward(current_tile, self._model.mission(missions_bot).final)
        for enemy_id, (enemy_pos, enemy_mission) in self._enemyPositions.items():
            if enemy_mission :
                enemy_mission_obj = self._model.mission(enemy_mission)
                enemy_target = enemy_mission_obj.final  # Case finale de la mission de l'ennemi
                enemy_next_move, enemy_next_pos = self.moveToward(enemy_pos, enemy_target)  # Prochaine case estimée

                # Si mon bot veut aller là où l'ennemi va se déplacer, il ne bouge pas
                if enemy_next_pos == bot_next_pos:
                    return "pass"
        
        if current_mission != 0:
            mission_obj = self._model.mission(current_mission)
            target_tile = mission_obj.final
            if current_tile == target_tile:
                self._missions[idBot] = 0
                return f"mission {idBot} {current_mission}"       

        else:
            missions_bot = self._missions[idBot]
            if missions_bot != 0 and missions_bot in self._free_missions:
                mission_obj = self._model.mission(missions_bot)
                target_tile = mission_obj.start
                if current_tile == target_tile:
                    return f"mission {idBot} {missions_bot}"
            else:
                self.setMissions(idBot)
                return "pass"
            
        if target_tile:
        # Utilise computeSafePath pour trouver un chemin sans collision
            new_paths = self.computeSafePath(idBot, current_tile, target_tile, self.paths)
        
            # Si un chemin a été trouvé pour ce robot
            if idBot in new_paths:
                self.paths = new_paths  # Met à jour les chemins
                moves, path = new_paths[idBot]  # Décompresse le tuple
                if moves:  # S'il y a des mouvements possibles
                    return f"move {idBot} {moves[0]}"  # Prend le premier mouvement
        
        return "pass"
    
    def decide(self):
        nbr_mobile = self._model.numberOfMobiles(self._id)
        mission = ""
        move = ""
        self.robot_positions = {str(i): self._model.mobilePosition(self._id, i) for i in range(1, nbr_mobile + 1)}
        for i in range(1, nbr_mobile+1):
            decision = self.decideBot(i)
            if "mission" in decision:
                decision_result = decision.split("mission")[1]
                mission += decision_result
            if "move" in decision:
                decision_result = decision.split("move")[1]
                move += decision_result
        return f"mission{mission} move{move}"


    def sleep(self, result):
        print(f"end on : {result}")
        self._sumResult += result
        self._countResult += 1

    def computeDistances(self, iTile):
        size = self._model.map().size()
        dists = [0] * (size + 1)
        dists[iTile] = 0
        ringNodes = self._model.map().neighbours(iTile)
        ringDistance = 1
        while ringNodes:
            nextNodes = []
            for node in ringNodes:
                if dists[node] == 0:
                    dists[node] = ringDistance
            for node in ringNodes:
                neighbours = self._model.map().neighbours(node)
                for candidate in neighbours:
                    if candidate != iTile and dists[candidate] == 0:
                        nextNodes.append(candidate)
            ringNodes = list(set(nextNodes))
            ringDistance += 1
        dists[iTile] = 0
        return dists
    
    def moveToward(self, iTile, iTarget):
        if iTile == iTarget:
            return 0, iTile
        clockdirs = self._model.map().clockBearing(iTile)
        nextTiles = self._model.map().neighbours(iTile)
        selectedDir = clockdirs[0]
        selectedNext = nextTiles[0]
        for clock, tile in zip(clockdirs, nextTiles):
            if self._distances[tile][iTarget] < self._distances[selectedNext][iTarget]:
                selectedDir = clock
                selectedNext = tile
        return selectedDir, selectedNext
    
    def path(self, iTile, iTarget, max_paths=3):
        all_paths = []  # Liste des chemins trouvés
        queue = [(iTile, [], [iTile])]  # (position actuelle, moves, path)
        
        while queue and len(all_paths) < max_paths:
            current, moves, path = queue.pop(0)
            
            # Si on atteint la destination
            if current == iTarget:
                all_paths.append((moves, path[1:]))  # path[1:] pour exclure la position initiale
                continue
            
            # Récupère les directions possibles depuis la position actuelle
            clockdirs = self._model.map().clockBearing(current)
            nextTiles = self._model.map().neighbours(current)
            
            # Trie les voisins par distance à la cible
            moves_and_tiles = sorted(zip(clockdirs, nextTiles),
                                key=lambda x: self._distances[x[1]][iTarget])
            
            # Explore chaque direction possible
            for clock, next_tile in moves_and_tiles:
                if next_tile not in path:  # Évite les cycles
                    new_moves = moves + [clock]
                    new_path = path + [next_tile]
                    queue.append((next_tile, new_moves, new_path))
        
        return all_paths
    
    def detectCollision(self, path1, path2):
        min_len = min(len(path1), len(path2))
        
        for t in range(min_len):
            # Cas 1 : Même position au même moment
            if path1[t] == path2[t]:
                return True
                
            # Cas 2 : Échange de positions entre deux pas de temps
            if t > 0 and path1[t] == path2[t-1] and path1[t-1] == path2[t]:
                return True
                
        return False
    
    def computeSafePath(self, idBot, iTile, iTarget, paths):
        candidate_paths = self.path(iTile, iTarget, max_paths=50)
        if len(candidate_paths) == 0:
            return []
       # Premier robot
        if not paths:
            # Essaie tous les chemins candidats jusqu'à en trouver un valide
            for moves, pathbot in candidate_paths:
                next_position = pathbot[0]
                position_occupied = False
                
                # Vérifie si la première position est occupée
                for robot_id, pos in self.robot_positions.items():
                    if robot_id != str(idBot) and next_position == pos:
                        position_occupied = True
                        break
                        
                if not position_occupied:  # Si on trouve un chemin dont la première position est libre
                    paths[idBot] = (moves, pathbot)
                    return paths
                    
            return paths  # Si aucun chemin valide n'est trouvé
        
        valid_paths = []  # Va stocker des tuples (moves, path)
        
        # Trouve chemins sans collision 
        for moves, pathbot in candidate_paths:
            # Vérifie d'abord que la première position n'est pas occupée par un autre robot
            next_position = pathbot[0]
            position_occupied = False
            for robot_id, pos in self.robot_positions.items():
                if robot_id != str(idBot) and next_position == pos:
                    position_occupied = True
                    break
                    
            if position_occupied:
                continue
                
            has_collision = False
            for bot_id, existing_data in paths.items():
                if bot_id != idBot:
                    _, existing_path = existing_data
                    if self.detectCollision(pathbot, existing_path):
                        has_collision = True
                        break
            
            if not has_collision:
                valid_paths.append((moves, pathbot))
        
        # S'il y a des chemins valides, prend le plus court
        if valid_paths:
            shortest = min(valid_paths, key=lambda x: len(x[1]))  # Compare longueur des chemins
            paths[idBot] = shortest  # Stocke le tuple (moves, path)
            return paths
        
        # Si aucun chemin valide, modifie le chemin du robot 1
        robot1_current = self._model.mobilePosition(self._id, 1)
        robot1_target = paths[1][1][-1]  # Accède à la dernière position du chemin
        
        robot1_paths = self.path(robot1_current, robot1_target, max_paths=50)
        
        # Test toutes les combinaisons
        if len(robot1_paths) != 0 :
            for r1_moves, r1_path in robot1_paths:
                if len(r1_path) == 0:
                    return []
                if r1_path[0] == self.robot_positions["2"]:
                    continue
                for r2_moves, r2_path in candidate_paths:
                    if len(r2_path) == 0:
                        return []
                    if r2_path[0] == self.robot_positions["1"]:
                        continue
                    if not self.detectCollision(r1_path, r2_path):
                        paths[1] = (r1_moves, r1_path)  # Stocke les tuples
                        paths[idBot] = (r2_moves, r2_path)
                        return paths
                        
            return paths
        else :
            return []

class UltimateBot():

    def __init__(self, name):
        self._name = name

    def wakeUp(self, playerId, numberOfPlayers, gameConfiguration):
        """Initialise le bot avec les configurations de jeu."""
        self._model = GameEngine()
        self._model.fromPod(gameConfiguration)
        self._id = playerId
        self._numberOfMobiles = self._model.numberOfMobiles(iPlayer=self._id)
        self._numberOfPlayers = self._model.numberOfPlayers()
        self._distances = {}
        self._free_missions = []
        self._old_action = {}
        self._robot_next_tile = []
        self._neighbours_tiles = []
        self._other_robot_positions = []
        self._dirs = []
        self._model.render()

        # Calcul des distances pour toutes les tuiles
        map_size = self._model.map().size()
        for tile in range(1, map_size + 1):
            self._distances[tile] = self.computeDistances(tile)

        self._free_missions = self._model.freeMissions()
        self._missions = {robot_id: 0 for robot_id in range(1, self._numberOfMobiles + 1)}
        self.assignOptimalMissions()
        self._enemyPositions = {}
        self._sumResult = 0
        self._countResult = 0

    def perceive(self, state):
        """Mise à jour de l'état du jeu."""
        self._model.setOnState(state)
        self._model.render()
        self._free_missions = self._model.freeMissions()
        self._enemyPositions = self.getEnemyPosition()

    def assignOptimalMissions(self):
        """Attribue des missions optimales en fonction des distances."""
        bot_positions = {bot_id: self._model.mobilePosition(self._id, bot_id) for bot_id in range(1, self._numberOfMobiles + 1)}
        remaining_missions = list(self._free_missions)
        bot_mission_distances = {bot_id: [] for bot_id in bot_positions}

        # Calcul des distances bot -> mission
        for bot_id, bot_pos in bot_positions.items():
            for mission_id in remaining_missions:
                mission_start = self._model.mission(mission_id).start
                distance = self._distances[bot_pos][mission_start]
                bot_mission_distances[bot_id].append((distance, mission_id))

        # Trie des missions par distance croissante
        for bot_id in bot_mission_distances:
            bot_mission_distances[bot_id].sort()

        assigned_missions = set()
        for bot_id in bot_positions:
            for distance, mission_id in bot_mission_distances[bot_id]:
                if mission_id not in assigned_missions:
                    self._missions[bot_id] = mission_id
                    assigned_missions.add(mission_id)
                    break

    def minDistanceToMission(self, id_player, id_bot):
        current_tile = self._model.mobilePosition(id_player, id_bot)

        if not self._free_missions:
            return None
        
        min_mission_id = None
        best_score = float('-inf')
        
        for mission_id in self._free_missions:
            mission_obj = self._model.mission(mission_id)
            start, final, reward = mission_obj.start, mission_obj.final, mission_obj.reward

            distance_to_start = self._distances[current_tile][start]

            if distance_to_start == 0:
                return mission_id
            
            distance_start_to_end = self._distances[start][final]
            total_distance = distance_to_start + distance_start_to_end
            
            if total_distance == 0:
                continue
            
            score = reward / total_distance
            
            # Pénalité si un robot ennemi est trop proche
            if self._enemyPositions:
                for enemy_id, (enemy_pos, enemy_mission) in self._enemyPositions.items():
                    
                    if enemy_mission != 0 and enemy_mission != None:
                        enemy_distance = self._distances[enemy_pos][start]
                        if enemy_distance < distance_to_start:
                            score -= 10
            
            if score > best_score:
                best_score = score
                min_mission_id = mission_id

        return min_mission_id

    def setMissions(self, robot_id):
        """Attribue une mission à un robot spécifique."""
        if self._free_missions:
            mission_id = self.minDistanceToMission(self._id, robot_id)
            if mission_id in self._free_missions:
                self._missions[robot_id] = mission_id

    def getEnemyPosition(self):
        """ Récupère la position des robots adverses et leurs missions. """
        enemy_players = [player_id for player_id in range(1, self._numberOfPlayers + 1) if self._id != player_id]
        enemy_positions = {}
        
        for player_id in enemy_players:
            for bot_id in range(1, self._model.numberOfMobiles(iPlayer=player_id) + 1):
                position = self._model.mobilePosition(player_id, bot_id)
                mission = self._model.mobileMission(player_id, bot_id)
                if mission == 0:
                    mission = self.minDistanceToMission(player_id, bot_id)
                enemy_positions[bot_id] = (position, mission)
        return enemy_positions

    def takeDecision(self, idBot):
        """Prend une décision pour un robot spécifique."""
        current_tile = self._model.mobilePosition(self._id, idBot)
        current_mission = self._model.mobile(self._id, idBot).mission()
        self._dirs = self._model.map().clockBearing(current_tile)
        all_neighbours = self._model.map().neighbours(current_tile)
        self._neighbours_tiles = [e for e in all_neighbours]
        next_tile = ""

        self.remove_avoid_tiles(current_tile)

        if current_mission != 0:
            mission_obj = self._model.mission(current_mission)
            target_tile = mission_obj.final
            if current_tile == target_tile:
                if current_tile not in self._vip_neighbours:
                    self._missions[idBot] = 0
                    return f"mission {idBot} {current_mission}"
                else:
                    self.remove_not_moving_action()
                    return self.randomMove(idBot, current_tile)
        else:
            mission_id = self._missions.get(idBot, 0)
            if mission_id != 0 and mission_id in self._free_missions:
                target_tile = self._model.mission(mission_id).start
                if current_tile == target_tile:
                    if current_tile not in self._vip_neighbours:
                        return f"mission {idBot} {mission_id}"
                    else:
                        self.remove_not_moving_action()
                        return self.randomMove(idBot, current_tile)
            else:
                if (current_tile in self._robot_next_tile) or (current_tile in self._vip_neighbours) or any(element in all_neighbours for element in self._other_robot_positions):
                    self.remove_not_moving_action()
                    return self.randomMove(idBot, current_tile)
                else:
                    self.setMissions(idBot)
                    return "pass"
                
        current_old_actions = self._old_action.get(idBot)
        if current_old_actions:
            if detect_cycle(current_old_actions):
                t = current_old_actions[-2]
                if t in self._neighbours_tiles:
                    index_t = self._neighbours_tiles.index(t)
                    self._neighbours_tiles.remove(t)
                    del self._dirs[index_t]
                self._break = True

        clock, next_tile = self.moveToward(current_tile, target_tile)

        msg= f'tic-{ self._model.tic() } | score { self._model.score(self._id) }'
        msg+= f' | postion {current_tile} and move {self._dirs} mission {current_mission}'

        self.updateActions(idBot, next_tile)
        return f"move {idBot} {clock}"

    def randomMove(self, idBot, current_tile):
        """Effectue un mouvement aléatoire."""
        try:
            choice = random.choice(self._dirs)
            next_tile = self._neighbours_tiles[self._dirs.index(choice)]
        except (IndexError, ValueError):
            choice = 0
            next_tile = current_tile

        self.updateActions(idBot, next_tile)
        return f"move {idBot} {choice}"

    def remove_avoid_tiles(self, current_tile):
        """Enlève les directions menant à des tuiles à éviter."""
        avoid_tiles = set(self._other_robot_positions) | set(self._robot_next_tile) | set(self._vip_neighbours)
        self._dirs = [dir for dir in self._dirs if self._model.map().clockposition(current_tile, dir) not in avoid_tiles]
        self._neighbours_tiles = [tile for tile in self._neighbours_tiles if tile not in avoid_tiles]

    def updateActions(self, idBot, next_tile):
        """Mise à jour des actions pour un bot."""
        if idBot not in self._old_action:
            self._old_action[idBot] = []
        self._old_action[idBot].append(next_tile)
        self._robot_next_tile.append(next_tile)

    def remove_not_moving_action(self):
        """Retirer la décision de ne pas bouger."""
        if len(self._dirs) >= 2 and 0 in self._dirs:
            index_0 = self._dirs.index(0)
            del self._neighbours_tiles[index_0]
            self._dirs.remove(0)

    def decide(self):
        """Prend les décisions pour tous les robots."""
        self._robot_next_tile = []
        self._break = False
        nbr_mobile = self._model.numberOfMobiles(self._id)
        mission = ""
        move = ""

        robot_positions = {str(i): self._model.mobilePosition(self._id, i) for i in range(1, nbr_mobile + 1)}
        try:
            self._vip_neighbours = self._model.map().neighbours(self._model.mobilePosition(0, 0))
        except Exception as e:
            self._vip_neighbours = []

        sorted_robots = self.sort_robots_by_remain_dirs(nbr_mobile, robot_positions)

        for robot_number in sorted_robots:
            self._other_robot_positions = [valeur for cle, valeur in robot_positions.items() if cle != robot_number]
            enemies_positions = [ enemy_pos for _, (enemy_pos, _) in self._enemyPositions.items()]
            self._other_robot_positions += enemies_positions
            for position in enemies_positions:
                self._other_robot_positions += self._model.map().neighbours(position)
            decision = self.takeDecision(int(robot_number))
            if "mission" in decision:
                decision_result = decision.split("mission")[1]
                mission += decision_result
            if "move" in decision:
                decision_result = decision.split("move")[1]
                move += decision_result

        return f"mission {mission.lstrip()} move {move.lstrip()}"

    def computeDistances(self, iTile):
        """Calculer les meilleures distances."""
        size = self._model.map().size()
        dists = [0] * (size + 1)
        dists[iTile] = 0  # Distance de départ = 0
        ringNodes = self._model.map().neighbours(iTile)
        ringDistance = 1
        while ringNodes:
            nextNodes = []
            for node in ringNodes:
                if dists[node] == 0:
                    dists[node] = ringDistance
            for node in ringNodes:
                neighbours = self._model.map().neighbours(node)
                for candidate in neighbours:
                    if candidate != iTile and dists[candidate] == 0:
                        nextNodes.append(candidate)
            ringNodes = list(set(nextNodes))
            ringDistance += 1
        dists[iTile] = 0
        return dists

    def moveToward(self, iTile, iTarget):
        """Prendre la meilleure direction."""    
        if iTile == iTarget:
            return 0, iTile  # Pas de mouvement nécessaire.

        self.remove_not_moving_action()

        try:
            selectedDir = self._dirs[0]
            selectedNext = self._neighbours_tiles[0]
        except IndexError:
            return 0, iTile
        
        for clock, tile in zip(self._dirs, self._neighbours_tiles):
            if self._distances[tile][iTarget] < self._distances[selectedNext][iTarget]:
                selectedDir = clock
                selectedNext = tile

        return selectedDir, selectedNext

    def sort_robots_by_remain_dirs(self, nbr_mobile, robot_positions):
        """Trier les robots en fonctions du nombre de directions possibles."""
        dict_remain_dirs = {}

        for i in range(1, nbr_mobile+1):
            current_tile = self._model.mobilePosition(self._id, i)
            self._dirs= self._model.map().clockBearing(current_tile)
            all_neighbours = self._model.map().neighbours(current_tile)
            self._neighbours_tiles = [e for e in all_neighbours]
            self._other_robot_positions = [valeur for cle, valeur in robot_positions.items() if cle != str(i)]
            enemies_positions = [ enemy_pos for _, (enemy_pos, _) in self._enemyPositions.items()]
            self._other_robot_positions += enemies_positions
            for position in enemies_positions:
                self._other_robot_positions += self._model.map().neighbours(position)
            current_tile = self._model.mobilePosition(self._id, i)
            self.remove_avoid_tiles(current_tile)
            dict_remain_dirs[str(i)] = len(self._dirs)

        sorted_robots = sorted(dict_remain_dirs.keys(), key=lambda k: dict_remain_dirs[k])

        return sorted_robots

    def sleep(self, result):
        """Met fin au tour."""
        print()
        print(f"{self._name} end on : {result}")
        self._sumResult += result
        self._countResult += 1

class SoloBot():
    def wakeUp(self, playerId, numberOfPlayers, gameConfiguration ):
        self._model= GameEngine()
        self._model.fromPod(gameConfiguration)
        self._id= playerId
        self._model.render()
        #print( f"Output image : ./shot-moveIt.png" ) 
        self.initGame()

    def initGame(self):
        self._distances = {}
        self._numberOfMobiles = self._model.numberOfMobiles()
        map_size = self._model.map().size()
        for tile in range(1, map_size + 1):
            self._distances[tile] = self.computeDistances(tile)
        self._sumResult = 0
        self._countResult = 0
        self.paths = {}
        self._free_missions = self._model.freeMissions()
        self._missions = {robot_id: 0 for robot_id in range(1, self._numberOfMobiles + 1)}
        self._start_positions = {i: self._model.mobilePosition(self._id, i) for i in range(1, self._numberOfMobiles + 1)}
        self.assignOptimalMissions()


    def assignOptimalMissions(self):
        bot_positions = {bot_id: self._model.mobilePosition(self._id, bot_id) for bot_id in range(1, self._numberOfMobiles + 1)}
        remaining_missions = list(self._free_missions)
        bot_mission_distances = {bot_id: [] for bot_id in bot_positions}
        
        # Calcul des distances entre chaque bot et chaque mission
        for bot_id, bot_pos in bot_positions.items():
            for mission_id in remaining_missions:
                mission_start = self._model.mission(mission_id).start
                distance = self._distances[bot_pos][mission_start]
                bot_mission_distances[bot_id].append((distance, mission_id))
        
        # Trie les missions par distance croissante pour chaque bot
        for bot_id in bot_mission_distances:
            bot_mission_distances[bot_id].sort()


        assigned_missions = set()
        # Assigner les missions en fonction des distances
        for bot_id in bot_positions:
            for distance, mission_id in bot_mission_distances[bot_id]:
                if mission_id not in assigned_missions:
                    self._missions[bot_id] = mission_id
                    assigned_missions.add(mission_id)
                    break 
    
    def minDistanceToMission(self, id_player, id_bot):
        """Trouve la mission la plus proche d'un bot."""
        current_tile = self._model.mobilePosition(id_player, id_bot)
        if not self._free_missions:
            return None

        min_mission_id = None
        best_score = float('-inf')
        for mission_id in self._free_missions:
            mission_obj = self._model.mission(mission_id)
            start, final, reward = mission_obj.start, mission_obj.final, mission_obj.reward
            total_distance = self._distances[current_tile][start] + self._distances[start][final]

            if total_distance > 0:
                score = reward / total_distance
                if score > best_score:
                    best_score = score
                    min_mission_id = mission_id

        return min_mission_id
    
    def setMissions(self, robot_id):
        if not self._free_missions:
            return
        mission_id = self.minDistanceToMission(self._id, robot_id)
        if mission_id in self._free_missions:
            self._missions[robot_id]= mission_id

    def perceive(self, state ):
        self._model.setOnState(state)
        self._model.render()
        self.assignOptimalMissions()
        self._free_missions = self._model.freeMissions()
        """print(self.paths)
        print(self._missions)"""

    def decideBot(self, idBot):
        current_tile = self._model.mobilePosition(self._id, idBot)
        current_mission = self._model.mobile(self._id, idBot).mission()
        msg= f'tic-{ self._model.tic() } | score { self._model.score(self._id) }'
        msg+= f' | postion {current_tile}'
        #print( msg )
        if idBot in self.paths:
            self.paths.pop(idBot)

        target_tile = None
        if current_mission != 0:
            mission_obj = self._model.mission(current_mission)
            target_tile = mission_obj.final
            if current_tile == target_tile:
                self._missions[idBot] = 0
                return f"mission {idBot} {current_mission}"       
        else:
            missions_bot = self._missions[idBot]
            if missions_bot != 0:
                mission_obj = self._model.mission(missions_bot)
                target_tile = mission_obj.start
                if current_tile == target_tile:
                    return f"mission {idBot} {missions_bot}"
            else:
                self.setMissions(idBot)
                if not self._missions[idBot]:
                    neighbour = self._model.map().neighbours(current_tile)
                    if any(element in neighbour for element in self.other_robot_position):
                        for n in neighbour:
                            new_paths = self.computeSafePath(idBot, current_tile, n, self.paths)
                            if idBot in new_paths:
                                self.paths[idBot] = new_paths[idBot]
                                moves, path = new_paths[idBot]
                                if moves:
                                    return f"move {idBot} {moves[0]}"
                            continue
                        return "pass"
                    return "pass"
        if target_tile is None:
            return "pass"
        new_paths = self.computeSafePath(idBot, current_tile, target_tile, self.paths)
        if idBot in new_paths:
            self.paths[idBot] = new_paths[idBot]
            moves, path = new_paths[idBot]
            if moves:
                return f"move {idBot} {moves[0]}"
        return "pass"


    def decide(self):
        nbr_mobile = self._model.numberOfMobiles(self._id)
        mission = ""
        move = ""
        self.robot_positions = {str(i): self._model.mobilePosition(self._id, i) for i in range(1, nbr_mobile + 1)}
        bots_sorted = self.sortedBotsByPriority()
        for robot_id in bots_sorted:
            self.other_robot_position = [valeur for cle, valeur in self.robot_positions.items() if cle !=robot_id]
            decision = self.decideBot(robot_id)
            if "mission" in decision:
                decision_result = decision.split("mission")[1]
                mission += decision_result
            if "move" in decision:
                decision_result = decision.split("move")[1]
                move += decision_result
        return f"mission{mission} move{move}"


    def sleep(self, result):
        #print(f"end on : {result}")
        self.final_score = result
        self._sumResult += result
        self._countResult += 1


    def computeDistances(self, iTile):
        size = self._model.map().size()
        dists = [0] * (size + 1)
        dists[iTile] = 0
        ringNodes = self._model.map().neighbours(iTile)
        ringDistance = 1
        while ringNodes:
            nextNodes = []
            for node in ringNodes:
                if dists[node] == 0:
                    dists[node] = ringDistance
            for node in ringNodes:
                neighbours = self._model.map().neighbours(node)
                for candidate in neighbours:
                    if candidate != iTile and dists[candidate] == 0:
                        nextNodes.append(candidate)
            ringNodes = list(set(nextNodes))
            ringDistance += 1
        dists[iTile] = 0
        return dists
    

    def path(self, iTile, iTarget, max_paths=3):
        all_paths = []
        queue = [(iTile, [], [iTile])]
        while queue and len(all_paths) < max_paths:
            current, moves, path = queue.pop(0)
            if current == iTarget:
                all_paths.append((moves, path[1:]))
                continue
            clockdirs = self._model.map().clockBearing(current)
            nextTiles = self._model.map().neighbours(current)
            moves_and_tiles = sorted(zip(clockdirs, nextTiles),
                                key=lambda x: self._distances[x[1]][iTarget])
            for clock, next_tile in moves_and_tiles:
                if next_tile not in path:
                    new_moves = moves + [clock]
                    new_path = path + [next_tile]
                    queue.append((next_tile, new_moves, new_path))
        return all_paths
    

    def detectCollision(self, path1, path2):
        if len(path1) == 1 and len(path2)<1 and path1[0]==path2[1]:
            return True
        
        if len(path1) > 1 and len(path2)==1 and path1[1]==path2[0]:
            return True

        min_len = min(len(path1), len(path2))
        for t in range(min_len):
            if path1[t] == path2[t]:
                return True
            if t > 0 and path1[t] == path2[t-1] and path1[t-1] == path2[t]:
                return True
            if t > 0 and path1[t]==path2[t-1]:
                return True
            if t > 0 and path1[t-1] == path2[t]:
                return True

        return False


    def computeSafePath(self, idBot, iTile, iTarget, paths):

        candidate_paths = self.path(iTile, iTarget, max_paths=50)
        valid_paths = []

        for moves, pathbot in candidate_paths:
            if not pathbot:
                continue
            next_position = pathbot[0]
            position_occupied = False
            for robot_id, pos in self.robot_positions.items():
                if robot_id != str(idBot) and next_position == pos:
                    position_occupied = True
                    break
            if position_occupied:
                continue
            has_collision = False
            for bot_id, existing_data in paths.items():
                if bot_id != idBot:
                    _, existing_path = existing_data
                    if self.detectCollision(pathbot, existing_path):
                        has_collision = True
                        break
            if not has_collision:
                valid_paths.append((moves, pathbot))
        if valid_paths:
            shortest = min(valid_paths, key=lambda x: len(x[1]))
            paths[idBot] = shortest
            return paths
        return paths

    def availableMovesCount(self, robot_id):
        current_tile = self._model.mobilePosition(self._id, robot_id)
        neighbors = self._model.map().neighbours(current_tile)
        available = []
        for n in neighbors:
            if any(pos == n for rid, pos in self.robot_positions.items() if str(rid) != str(robot_id)):
                continue
            reserved = False
            for other_id, data in self.paths.items():
                if str(other_id) != str(robot_id):
                    moves, path = data
                    if path and path[0] == n:
                        reserved = True
                        break
            if reserved:
                continue
            available.append(n)
        return len(available)
    
    def sortedBotsByPriority(self):
        nbr_mobile = self._model.numberOfMobiles(self._id)
        self.robot_positions = {str(i): self._model.mobilePosition(self._id, i) 
                                for i in range(1, nbr_mobile + 1)}
        bots = []
        for i in range(1, nbr_mobile + 1):
            mobility = self.availableMovesCount(i)
            bots.append((i, mobility))
        bots.sort(key=lambda x: x[1])
        return [bot_id for bot_id, _ in bots]