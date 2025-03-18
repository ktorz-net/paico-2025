# MultiPlayerBot.py
from hacka.games.moveit import GameEngine
import random
import copy

class MultiPlayerBot():
    # Player interface :
    def wakeUp(self, playerId, numberOfPlayers, gameConfiguration):
        self._id = playerId
        self._model = GameEngine()
        self._model.fromPod(gameConfiguration)
        self._model.render()
        self._nb_robots = self._model.numberOfMobiles(self._id)
        self._nb_players = numberOfPlayers
        self._debug = False
        self.log(f'number of robot for player {self._id}: {self._nb_robots}')
        self.vip_prediction_total = 0
        self.vip_prediction_success = 0
        s = self._model.map().size()
        self._distances = [[i for i in range(s + 1)]]
        self.log(self._distances)
        for i in range(1, s + 1):
            dist = self.computeDistances(i)
            self.log(dist)
            self._distances.append(dist)

        # Initialisation du tracker pour le VIP (playerID 0)
        self._vip_tracker = VIPMoveTracker()
        self._vip_prev = self._model.mobilePosition(0, 1)
        # Initialisation des positions des autres robots (players 1 et 2)
        self._other_robots_prev = self.get_other_robots_positions()

    def perceive(self, gameState):
        self._model.setOnState(gameState)
        #self._model.render()
        #time.sleep(0.3)
        curr_vip = self._model.mobilePosition(0, 1)
        other_robots_curr = self.get_other_robots_positions()
        actual_move = self._vip_tracker.update(self._vip_prev, curr_vip, self._model.map(), self._other_robots_prev,
                                               other_robots_curr)
        self._vip_prev = curr_vip
        self._other_robots_prev = other_robots_curr
        predicted_moves = self._vip_tracker.predict_next_move(curr_vip, self._model.map())
        if len(predicted_moves) > 1:
            best_predicted = predicted_moves[1]
        else:
            best_predicted = predicted_moves[0]
        if actual_move is not None and actual_move != -1:
            self.vip_prediction_total += 1
            if actual_move == best_predicted:
                self.vip_prediction_success += 1
        #print("VIP possibility", self._vip_tracker.predict_next_move_distribution(curr_vip, self._model.map()))
        #print("VIP moves:", self._vip_tracker.moves)

    # Méthode utilitaire pour récupérer les positions des robots des players 1 et 2
    def get_other_robots_positions(self):
        positions = {}
        # On suppose qu'il y a toujours 2 joueurs (IDs 1 et 2)
        for player in [1, 2]:
            num_mobiles = self._model.numberOfMobiles(player)
            for robot_id in range(1, num_mobiles + 1):
                # On utilise une clé tuple (player, robot_id) pour identifier chaque robot
                positions[(player, robot_id)] = self._model.mobilePosition(player, robot_id)
        return positions

    def decide(self):
        actions = self.get_actions()
        #print(f"{self._model._tic} multiplayerbot player-{self._id}({self._model.score(self._id)}): {actions}")
        return actions

    def sleep(self, result):
        return
        #self.log(f"end on : {result}")
        #print(f"VIP prediction accuracy: {self.vip_prediction_success} / {self.vip_prediction_total}")

    def log(self, message):
        if self._debug:
            print(message)

    def get_actions(self, next_steps=None):
        if next_steps is None:
            next_steps = self.get_next_steps_override()

        move_actions = []
        mission_actions = []
        actions = []
        for robot_id in range(1, self._nb_robots + 1):
            self.log(f"deciding for {robot_id}")
            robot_position = self._model.mobilePosition(self._id, robot_id)
            robot_mission = self._model.mobileMission(self._id, robot_id)
            self.log(f"\tposition: {robot_position}")
            self.log(f"\tmission: {robot_mission}")
            current_action = ""

            next_moves = next_steps[robot_id]
            (action_keyword, next_move, _) = random.choice(next_moves)
            current_action = f"{action_keyword} {robot_id} {next_move}"
            self.log(f"\taction: {current_action}")
            if "move" in current_action:
                current_action = current_action.removeprefix("move ")
                move_actions.append(current_action)
            if "mission" in current_action:
                current_action = current_action.removeprefix("mission ")
                mission_actions.append(current_action)

        if len(mission_actions) > 0:
            actions.append("mission " + " ".join(mission_actions))
        if len(move_actions) > 0:
            actions.append("move " + " ".join(move_actions))

        actions = " ".join(actions)
        self.log(f"{self._model._tic} - {actions}")
        return actions

    def get_missions_distances(self, player_id=1, all_mission_ids=None):
        result = [[]]
        if all_mission_ids is None:
            all_mission_ids = self._model.missionsList()
        self.log(all_mission_ids)
        for robot_id in range(1, self._nb_robots + 1):
            result.append([])
            robot_position = self._model.mobilePosition(player_id, robot_id)
            result[robot_id] = [-1 for _ in range(max(all_mission_ids) + 1)]
            for mission_id in all_mission_ids:
                mission = self._model.mission(mission_id)
                if mission.owner != 0:
                    mission_distance = -1
                else:
                    mission_distance = self._distances[robot_position][mission.start]
                result[robot_id][mission_id] = mission_distance
        self.log("Distances between robot and missions calculated :")
        self.log(result)
        return result

    def get_all_players_missions_distances(self, all_mission_ids=None):
        result = [[[]]]
        if all_mission_ids is None:
            all_mission_ids = self._model.missionsList()
        for player_id in range(1, self._nb_players + 1):
            player_mission_distances = self.get_missions_distances(player_id=player_id, all_mission_ids=all_mission_ids)
            result.append(player_mission_distances)
        return result
    
    

    def assign_missions(self, robot_to_missions_distances=None, all_mission_ids=None):
        if all_mission_ids is None:
            all_mission_ids = self._model.missionsList()
        if robot_to_missions_distances is None:
            robot_to_missions_distances = self.get_all_players_missions_distances(all_mission_ids=all_mission_ids)
        prev_reservation = []
        reservation = [[]]
        for _ in range(1, self._nb_players + 1):
            reservation.append([-1 for _ in range(0, self._nb_robots + 1)])
        max_iter = 1000
        counter = 0
        while prev_reservation != reservation:
            prev_reservation = copy.deepcopy(reservation)
            counter += 1
            if counter > max_iter:
                break
            for player_id in range(1, self._nb_players + 1):
                for robot_id in range(1, self._nb_robots + 1):
                    robot_mission = self._model.mobileMission(player_id, robot_id)
                    if robot_mission != 0:
                        reservation[player_id][robot_id] = robot_mission
                    if reservation[player_id][robot_id] > 0:
                        continue
                    distances = robot_to_missions_distances[player_id][robot_id]
                    min_distance = max(distances) + 1
                    selected_mission_id = None
                    for i, distance in enumerate(distances):
                        if distance < 0:
                            continue
                        if i in reservation[player_id]:
                            continue
                        if distance < min_distance:
                            min_distance = distance
                            selected_mission_id = i
                    if selected_mission_id != None:
                        is_closest = True
                        for other_player_id in range(1, self._nb_players + 1):
                            for other_robot_id in range(1, self._nb_robots + 1):
                                other_robot_mission = self._model.mobileMission(other_player_id, other_robot_id)
                                if other_robot_mission != 0:
                                    reservation[other_player_id][other_robot_id] = other_robot_mission
                                if reservation[other_player_id][other_robot_id] > 0:
                                    continue
                                if other_player_id == self._id and other_robot_id == robot_id:
                                    continue
                                other_robot_distances = robot_to_missions_distances[other_player_id][other_robot_id]
                                if other_robot_distances[selected_mission_id] < min_distance:
                                    is_closest = False
                                    break
                        if is_closest:
                            reservation[self._id][robot_id] = selected_mission_id
                    self.log(f"{counter} - reservations: {reservation}")
        self.log(f"reservations:\n{reservation}")
        return reservation

    def get_closest_mission(self, i_from):
        i = 1
        result = None
        all_free_missions = self._model.missions()
        if len(all_free_missions) == 0:
            return result
        result = all_free_missions[0]
        min_distance = self._distances[i_from][all_free_missions[0].start]
        for m in all_free_missions:
            if m.owner != 0:
                continue
            currentDistance = self._distances[i_from][m.start]
            if currentDistance < min_distance:
                result = m
            i += 1
        return result

    def missionOn(self, iTile):
        i = 1
        l = []
        for m in self._model.missions():
            if m.start == iTile:
                l.append(i)
            i += 1
        return l

    def computeDistances(self, iTile):
        dists = [iTile] + [0 for _ in range(self._model.map().size())]
        ringNodes = self._model.map().neighbours(iTile)
        ringDistance = 1
        while len(ringNodes) > 0:
            nextNodes = []
            for node in ringNodes:
                dists[node] = ringDistance
            for node in ringNodes:
                neighbours = self._model.map().neighbours(node)
                for candidate in neighbours:
                    if dists[candidate] == 0:
                        nextNodes.append(candidate)
            ringNodes = nextNodes
            ringDistance += 1
        dists[iTile] = 0
        return dists

    def get_next_steps_override(self, reservations=None):
        if reservations is None:
            reservations = self.assign_missions()
        all_next_steps = [[]]
        for player_id in range(1, self._nb_players + 1):
            player_next_steps = self.get_next_steps(player_id=player_id, reservations=reservations)
            all_next_steps.append(player_next_steps)
        next_step = all_next_steps[self._id]
        for robot_id in range(1, self._nb_robots + 1):
            current_next_moves = next_step[robot_id]
            for other_player_id in range(1, self._nb_players + 1):
                if other_player_id == self._id:
                    continue
                for other_robot_id in range(1, self._nb_robots + 1):
                    moves_to_remove = []
                    for i, (_, _, current_next_tile) in enumerate(current_next_moves):
                        for (_, _, other_robot_tile) in all_next_steps[other_player_id][other_robot_id]:
                            if current_next_tile == other_robot_tile:
                                if i not in moves_to_remove:
                                    moves_to_remove.insert(0, i)
                    for i in moves_to_remove:
                        self.log(f"remove {i} in {current_next_moves}")
                        del current_next_moves[i]
            if len(current_next_moves) <= 0:
                current_next_moves.append(("move", 0, self._model.mobilePosition(self._id, robot_id)))
        return next_step

    def get_next_steps(self, player_id=1, reservations=None):
        if reservations is None:
            reservations = self.assign_missions()
        player_reservations = reservations[player_id]
        robot_positions = [self._model.mobilePosition(player_id, i) for i in range(1, self._nb_robots + 1)]
        robot_positions.insert(0, -1)
        map_centers = []
        next_moves = [[] for _ in range(self._nb_robots + 1)]
        for robot_id in range(1, self._nb_robots + 1):
            self.log(f"{robot_id} robot:")
            robot_position = robot_positions[robot_id]
            robot_mission_id = self._model.mobileMission(player_id, robot_id)
            current_next_moves = []
            if player_reservations[robot_id] < 0:
                self.log("\tno reserved missions")
                map_center = int((self._model._map.size() / 2)) + 1
                c = 1
                while map_center in map_centers:
                    map_center = map_center + c
                    c += 1
                    c *= -1
                map_centers.append(map_center)
                self.log(f"\tgo to map center at cell {map_center}")
                current_next_moves = self.move_toward(robot_position, map_center)
                current_next_moves = [('move', *move) for move in current_next_moves]
            elif robot_mission_id == 0 and player_reservations[robot_id] > 0:
                self.log(f"\thas reserved {player_reservations[robot_id]} mission")
                robot_mission = self._model.mission(player_reservations[robot_id])
                self.log(f"\tgo to {robot_mission.start} cell")
                if robot_position == robot_mission.start:
                    current_next_moves = [('mission', player_reservations[robot_id], robot_position)]
                else:
                    current_next_moves = self.move_toward(robot_position, robot_mission.start)
                    current_next_moves = [('move', *move) for move in current_next_moves]
            elif robot_mission_id > 0:
                self.log(f"\thas activated {robot_mission_id} mission")
                robot_mission = self._model.mission(robot_mission_id)
                self.log(f"\tgo to {robot_mission.final} cell")
                if robot_position == robot_mission.final:
                    current_next_moves = [('mission', robot_mission_id, robot_position)]
                else:
                    current_next_moves = self.move_toward(robot_position, robot_mission.final)
                    current_next_moves = [('move', *move) for move in current_next_moves]
            for other_robot_id in range(1, robot_id + 1):
                if other_robot_id == robot_id:
                    continue
                other_robot_moves_to_remove = []
                moves_to_remove = []
                for i, (_, _, current_next_tile) in enumerate(current_next_moves):
                    for j, (_, _, other_robot_tile) in enumerate(next_moves[other_robot_id]):
                        if current_next_tile == other_robot_tile:
                            if len(current_next_moves) > 1:
                                if i not in moves_to_remove:
                                    moves_to_remove.insert(0, i)
                            elif len(next_moves[other_robot_id]) > 1:
                                if j not in other_robot_moves_to_remove:
                                    other_robot_moves_to_remove.insert(0, j)
                            else:
                                if i not in moves_to_remove:
                                    moves_to_remove.insert(0, i)
                for i in moves_to_remove:
                    self.log(f"remove {i} in {current_next_moves}")
                    del current_next_moves[i]
                for i in other_robot_moves_to_remove:
                    self.log(f"remove {i} in {other_robot_moves_to_remove}")
                    del next_moves[other_robot_id][i]
            if len(current_next_moves) <= 0:
                current_next_moves.append(("move", 0, robot_position))
            self.log(f"\t- next_moves: {current_next_moves}")
            next_moves[robot_id] = current_next_moves
        return next_moves

    def move_toward(self, iTile, iTarget):
        if iTile == iTarget:
            return [(0, iTile)]
        result = []
        clockdirs = self._model.map().clockBearing(iTile)
        nextTiles = self._model.map().neighbours(iTile)
        selectedDir = clockdirs[0]
        selectedNext = nextTiles[0]
        robot_positions = [[]]
        for player_id in range(1, self._nb_players + 1):
            player_robot_positions = [self._model.mobilePosition(player_id, i) for i in range(1, self._nb_robots + 1)]
            robot_positions.append(player_robot_positions)
        for clock, tile in zip(clockdirs, nextTiles):
            if any(tile in sub_positions for sub_positions in robot_positions):
                continue
            if self._distances[tile][iTarget] < self._distances[selectedNext][iTarget]:
                result = []
                selectedDir = clock
                selectedNext = tile
                result.append((selectedDir, selectedNext))
            elif self._distances[tile][iTarget] == self._distances[selectedNext][iTarget]:
                selectedDir = clock
                selectedNext = tile
                result.append((selectedDir, selectedNext))
        if len(result) == 0:
            result.append((0, iTile))
        return result

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

    def path(self, iTile, iTarget):
        clock, tile = self.moveToward(iTile, iTarget)
        move = [clock]
        path = [tile]
        while tile != iTarget:
            clock, tile = self.moveToward(tile, iTarget)
            move.append(clock)
            path.append(tile)
        return move, path