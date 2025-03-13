import copy
import random
import time
import numpy as np
from hacka.games.moveit import GameEngine
from dl_trainer import DeadlockSolver

class MultiPlayerBotEnnemy():
    # Interface joueur :
    def wakeUp(self, playerId, numberOfPlayers, gameConfiguration):
        self._id = playerId
        self._model = GameEngine()
        self._model.fromPod(gameConfiguration)  # Charge la configuration du jeu
        self._model.render()
        self._nb_robots = self._model.numberOfMobiles(self._id)
        self._nb_players = numberOfPlayers
        self._debug = True
        self.log(f"[wakeUp] Nombre de robots pour le joueur {self._id}: {self._nb_robots}")
        
        # Initialisation des dernières positions pour le suivi des répétitions
        self.last_positions = {robot_id: None for robot_id in range(1, self._nb_robots + 1)}
        # Nouvel attribut pour stocker l'historique des positions (pour les 4 derniers tours)
        self.position_history = {robot_id: [] for robot_id in range(1, self._nb_robots + 1)}
        
        # Analyse préalable de la map pour identifier les tunnels
        self.identify_tunnels()
        
        # Initialisation et chargement (ou entraînement rapide) du modèle DL pour le deadlock
        # (Vous pouvez ajuster extra_features selon vos besoins)
        self.deadlock_solver = DeadlockSolver(map_size=9, extra_features=5)
        try:
            self.deadlock_solver.load_model()  # Charger un modèle pré-entraîné
            self.log("[wakeUp] Modèle DL chargé avec succès")
        except Exception as e:
            self.log("[wakeUp] Entraînement d'un nouveau modèle DL...")
            self.deadlock_solver.train_on_deadlocks(num_episodes=500)
            self.deadlock_solver.save_model()
        
        # Initialisation de la table des distances
        s = self._model.map().size()
        self._distances = [[i for i in range(s + 1)]]
        self.log(f"[wakeUp] Distances initiales: {self._distances}")
        for i in range(1, s+1):
            dist = self.computeDistances(i)
            self.log(f"[wakeUp] Distances depuis la tuile {i}: {dist}")
            self._distances.append(dist)

    def perceive(self, gameState):
        self._model.setOnState(gameState)
        self._model.render()
        time.sleep(0.3)

    def decide(self):
        actions = self.get_actions()
        print(f"{self._model._tic} multiplayerbot player-{self._id}({self._model.score(self._id)}): {actions}")
        return actions

    def sleep(self, result):
        self.log(f"[sleep] end on : {result}")

    def log(self, message):
        if self._debug:
            print(message)

    # --- Méthode d'action du bot ---
    def get_actions(self, next_steps=None):
        if next_steps is None:
            next_steps = self.get_next_steps_override()

        move_actions = []
        mission_actions = []
        for robot_id in range(1, self._nb_robots + 1):
            self.log(f"[get_actions] Deciding for robot {robot_id}")
            robot_position = self._model.mobilePosition(self._id, robot_id)
            robot_mission = self._model.mobileMission(self._id, robot_id)
            self.log(f"\tPosition: {robot_position}, Mission: {robot_mission}")
            next_moves = next_steps[robot_id]
            # Choisir aléatoirement une action parmi celles proposées
            (action_keyword, next_move, _) = random.choice(next_moves)
            current_action = f"{action_keyword} {robot_id} {next_move}"
            self.log(f"\tAction choisie: {current_action}")
            if current_action.startswith("move"):
                move_actions.append(current_action.removeprefix("move ").strip())
            elif current_action.startswith("mission"):
                mission_actions.append(current_action.removeprefix("mission ").strip())

        actions = ""
        if mission_actions:
            actions += "mission " + " ".join(mission_actions) + " "
        if move_actions:
            actions += "move " + " ".join(move_actions)
        self.log(f"[get_actions] {self._model._tic} - {actions}")
        return actions

    # --- Calcul des distances sur la map (par parcours en largeur) ---
    def computeDistances(self, iTile):
        size = self._model.map().size()
        dists = [iTile] + [0 for _ in range(size)]
        ringNodes = self._model.map().neighbours(iTile)
        ringDistance = 1
        while ringNodes:
            nextNodes = []
            for node in ringNodes:
                dists[node] = ringDistance
            for node in ringNodes:
                for candidate in self._model.map().neighbours(node):
                    if dists[candidate] == 0:
                        nextNodes.append(candidate)
            ringNodes = nextNodes
            ringDistance += 1
        dists[iTile] = 0
        return dists

    # --- Planification des mouvements global ---
    def get_next_steps_override(self, reservations=None):
        if reservations is None:
            reservations = self.assign_missions()
        all_next_steps = [[]]
        for player_id in range(1, self._nb_players + 1):
            player_next_steps = self.get_next_steps(player_id=player_id, reservations=reservations)
            all_next_steps.append(player_next_steps)
        
        # Élimination simple de collisions entre joueurs d'équipes différentes
        next_step = all_next_steps[self._id]
        for robot_id in range(1, self._nb_robots + 1):
            current_moves = next_step[robot_id]
            for other_player_id in range(1, self._nb_players + 1):
                if other_player_id == self._id:
                    continue
                for other_robot_id in range(1, self._nb_robots + 1):
                    remove_indices = []
                    for i, (_, _, move_tile) in enumerate(current_moves):
                        for (_, _, other_tile) in all_next_steps[other_player_id][other_robot_id]:
                            if move_tile == other_tile:
                                if i not in remove_indices:
                                    remove_indices.insert(0, i)
                    for idx in remove_indices:
                        self.log(f"[get_next_steps_override] Suppression de mouvement {idx} pour robot {robot_id}")
                        del current_moves[idx]
            if not current_moves:
                current_moves.append(("move", 0, self._model.mobilePosition(self._id, robot_id)))
        
        # Résolution des collisions au sein de la même équipe
        self.resolve_team_collisions(next_step)
        
        # Gestion générale des blocages : on vérifie l'historique pour détecter un arrêt répété > 3 tours ou une alternance répétitive
        for robot_id in range(1, self._nb_robots + 1):
            current_position = self._model.mobilePosition(self._id, robot_id)
            if self.is_blocked(robot_id, current_position):
                self.log(f"[get_next_steps_override] Blocage confirmé pour robot {robot_id} (arrêt prolongé ou alternance répétitive).")
                new_move = self.handle_blockage(robot_id, current_position)
                next_step[robot_id] = [("move", 0, new_move)]
        return next_step

    def resolve_team_collisions(self, team_moves):
        planned = {}
        for robot_id in range(1, len(team_moves)):
            if team_moves[robot_id]:
                action, clock, move_tile = team_moves[robot_id][0]
                if action == "move":
                    if move_tile not in planned:
                        planned[move_tile] = []
                    planned[move_tile].append(robot_id)
        for tile, robots in planned.items():
            if len(robots) > 1:
                best_robot = None
                best_priority = None
                for r in robots:
                    mission = self._model.mobileMission(self._id, r)
                    if mission != 0:
                        reward_val = self._model.mission(mission).reward
                        current_pos = self._model.mobilePosition(self._id, r)
                        mission_start = self._model.mission(mission).start
                        x1, y1 = current_pos % 3, current_pos // 3
                        x2, y2 = mission_start % 3, mission_start // 3
                        dist = abs(x1 - x2) + abs(y1 - y2)
                    else:
                        reward_val = 0
                        current_pos = self._model.mobilePosition(self._id, r)
                        dist = 100  
                    priority = (reward_val, -dist)
                    if best_priority is None or priority > best_priority:
                        best_priority = priority
                        best_robot = r
                for r in robots:
                    if r != best_robot:
                        current_position = self._model.mobilePosition(self._id, r)
                        team_moves[r] = [("move", 0, current_position)]
                        self.log(f"[resolve_team_collisions] Robot {r} perd le conflit sur la case {tile} et reste sur place.")

    def is_blocked(self, robot_id, current_position):
        """
        Détecte un blocage si, dans l'historique, le robot reste sur la même case pendant plus de 4 tours
        ou s'il alterne entre deux positions de manière répétitive (ex: A, B, A, B).
        """
        self.position_history[robot_id].append(current_position)
        # Ne conserver que les 4 derniers mouvements
        if len(self.position_history[robot_id]) > 4:
            self.position_history[robot_id] = self.position_history[robot_id][-4:]
        hist = self.position_history[robot_id]
        # Condition 1 : arrêt prolongé sur plus de 3 tours
        if len(hist) >= 3 and all(pos == hist[0] for pos in hist[-3:]):
            return True
        # Condition 2 : alternance répétitive entre deux positions (ex: A, B, A, B)
        if len(hist) == 4 and hist[0] == hist[2] and hist[1] == hist[3] and hist[0] != hist[1]:
            return True
        return False


    # --- Attribution des missions ---
    def assign_missions(self, robot_to_missions_distances=None, all_mission_ids=None):
        if all_mission_ids is None:
            all_mission_ids = self._model.missionsList()
        if robot_to_missions_distances is None:
            robot_to_missions_distances = self.get_all_players_missions_distances(all_mission_ids=all_mission_ids)
        prev_reservation = []
        reservation = [[]]
        for _ in range(1, self._nb_players + 1):
            reservation.append([-1 for _ in range(self._nb_robots + 1)])
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
                    if selected_mission_id is not None:
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
                            if not is_closest:
                                break
                        if is_closest:
                            reservation[self._id][robot_id] = selected_mission_id
            self.log(f"[assign_missions] Iteration {counter} - reservations: {reservation}")
        self.log(f"[assign_missions] Reservations finales:\n{reservation}")
        return reservation

    def get_all_players_missions_distances(self, all_mission_ids=None):
        result = [[[]]]
        if all_mission_ids is None:
            all_mission_ids = self._model.missionsList()
        for player_id in range(1, self._nb_players + 1):
            player_mission_distances = self.get_missions_distances(player_id=player_id, all_mission_ids=all_mission_ids)
            result.append(player_mission_distances)
        return result

    def get_missions_distances(self, player_id=1, all_mission_ids=None):
        result = [[]]
        if all_mission_ids is None:
            all_mission_ids = self._model.missionsList()
        self.log(f"[get_missions_distances] All mission ids: {all_mission_ids}")
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
        self.log(f"[get_missions_distances] Calcul des distances: {result}")
        return result

    # --- Planification des mouvements par robot ---
    def get_next_steps(self, player_id=1, reservations=None):
        next_moves = [[] for _ in range(self._nb_robots + 1)]
        if reservations is None:
            reservations = self.assign_missions()
        player_reservations = reservations[player_id]
        robot_positions = [self._model.mobilePosition(player_id, i) for i in range(1, self._nb_robots + 1)]
        robot_positions.insert(0, -1)
        for robot_id in range(1, self._nb_robots + 1):
            self.log(f"[get_next_steps] Robot {robot_id}:")
            current_position = robot_positions[robot_id]
            # Vérifier en premier le tunnel
            tunnel = self.get_tunnel_for_tile(current_position)
            if tunnel:
                retreat_move = self.handle_tunnel_collision(robot_id, current_position, current_position)
                if retreat_move is not None and retreat_move != current_position:
                    self.log(f"[get_next_steps] Le mouvement pour robot {robot_id} est modifié: {current_position} -> {retreat_move}")
                    next_moves[robot_id] = [("move", 0, retreat_move)]
                    continue  # On passe immédiatement au robot suivant

            # Sinon, planification habituelle
            current_next_moves = []
            robot_mission_id = self._model.mobileMission(player_id, robot_id)
            if player_reservations[robot_id] < 0:
                self.log("\tAucune mission réservée : aller vers le centre")
                map_center = int((self._model._map.size() / 2)) + 1
                c = 1
                centers = []
                while map_center in centers:
                    map_center = map_center + c
                    c += 1
                    c *= -1
                centers.append(map_center)
                self.log(f"\tAller vers le centre: case {map_center}")
                current_next_moves = self.move_toward(current_position, map_center)
                current_next_moves = [('move', *move) for move in current_next_moves]
            elif robot_mission_id == 0 and player_reservations[robot_id] > 0:
                self.log(f"\tMission réservée: {player_reservations[robot_id]}")
                robot_mission = self._model.mission(player_reservations[robot_id])
                self.log(f"\tAller vers la case de départ: {robot_mission.start}")
                if current_position == robot_mission.start:
                    current_next_moves = [('mission', player_reservations[robot_id], current_position)]
                else:
                    current_next_moves = self.move_toward(current_position, robot_mission.start)
                    current_next_moves = [('move', *move) for move in current_next_moves]
            elif robot_mission_id > 0:
                self.log(f"\tMission activée: {robot_mission_id}")
                robot_mission = self._model.mission(robot_mission_id)
                self.log(f"\tAller vers la case d'arrivée: {robot_mission.final}")
                if current_position == robot_mission.final:
                    current_next_moves = [('mission', robot_mission_id, current_position)]
                else:
                    current_next_moves = self.move_toward(current_position, robot_mission.final)
                    current_next_moves = [('move', *move) for move in current_next_moves]
            next_moves[robot_id] = current_next_moves
            self.log(f"\t- Mouvements possibles: {current_next_moves}")
        return next_moves

    def move_toward(self, iTile, iTarget):
        if iTile == iTarget:
            return [(0, iTile)]
        result = []
        clockdirs = self._model.map().clockBearing(iTile)
        nextTiles = self._model.map().neighbours(iTile)
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
                selectedNext = tile
                result.append((clock, tile))
            elif self._distances[tile][iTarget] == self._distances[selectedNext][iTarget]:
                result.append((clock, tile))
        if not result:
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

    # --- Identification des tunnels ---
    def identify_tunnels(self):
        map_obj = self._model.map()
        candidate = set()
        for tile_id in range(1, map_obj.size() + 1):
            deg = len(map_obj.neighbours(tile_id))
            if deg <= 3:
                candidate.add(tile_id)
        tunnels = []
        visited = set()
        for tile in candidate:
            if tile in visited:
                continue
            comp = set()
            stack = [tile]
            while stack:
                t = stack.pop()
                if t in comp:
                    continue
                comp.add(t)
                visited.add(t)
                for nb in map_obj.neighbours(t):
                    if nb in candidate and nb not in comp:
                        stack.append(nb)
            exits = set()
            for t in comp:
                for nb in map_obj.neighbours(t):
                    if nb not in comp:
                        exits.add(nb)
            if len(exits) == 2:
                tunnels.append({'tiles': comp, 'entrances': exits})
        self._tunnels = tunnels
        if self._debug:
            self.log("Tunnels identifiés :")
            for idx, tunnel in enumerate(self._tunnels, start=1):
                self.log(f"Tunnel {idx} : Tuiles = {sorted(tunnel['tiles'])}, Entrées = {sorted(tunnel['entrances'])}")

    def get_tunnel_for_tile(self, tile_id):
        if not hasattr(self, '_tunnels'):
            return None
        for tunnel in self._tunnels:
            if tile_id in tunnel['tiles']:
                return tunnel
        return None

    def get_nearest_entrance(self, tile_id, tunnel):
        best = None
        best_dist = float('inf')
        for entrance in tunnel['entrances']:
            d = self._distances[tile_id][entrance]
            if d < best_dist:
                best_dist = d
                best = entrance
        return best

    def has_diagonal_neighbors(self, tile_id):
        map_obj = self._model.map()
        size = map_obj.size()
        x = (tile_id - 1) % size
        y = (tile_id - 1) // size
        direct_neighbors = set(map_obj.neighbours(tile_id))
        neighbors_of_neighbors = set()
        for n in direct_neighbors:
            neighbors_of_neighbors.update(map_obj.neighbours(n))
        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size:
                diag_tile = nx + ny * size + 1
                if diag_tile in neighbors_of_neighbors:
                    return True
        return False

    # --- Gestion de collision dans un tunnel avec Deep Learning ---
    def handle_tunnel_collision(self, robot_id, current_position, next_position):
        tunnel = self.get_tunnel_for_tile(current_position)
        if not tunnel:
            self.log(f"[handle_tunnel_collision] Robot {robot_id} n'est pas dans un tunnel")
            return next_position

        if current_position in tunnel['entrances']:
            self.log(f"[handle_tunnel_collision] Robot {robot_id} est déjà à une entrée du tunnel")
            return current_position

        robots_in_tunnel = [r for r in range(1, self._nb_robots + 1)
                              if self._model.mobilePosition(self._id, r) in tunnel['tiles']]
        if len(robots_in_tunnel) == 2:
            other_robot_id = next(r for r in robots_in_tunnel if r != robot_id)
            current_mission = self._model.mobileMission(self._id, robot_id)
            other_mission = self._model.mobileMission(self._id, other_robot_id)
            current_points = self._model.mission(current_mission).reward if current_mission != 0 else 0
            other_points = self._model.mission(other_mission).reward if other_mission != 0 else 0

            if (current_points < other_points) or (current_points == other_points and
               (current_mission != 0 and other_mission != 0 and
                self._distances[current_position][self._model.mission(current_mission).start] >
                self._distances[self._model.mobilePosition(self._id, other_robot_id)][self._model.mission(other_mission).start])):
                nearest_entrance = self.get_nearest_entrance(current_position, tunnel)
                self.log(f"[handle_tunnel_collision] Robot {robot_id} n'est pas prioritaire et recule à l'entrée du tunnel + 1")
                return self.retreat_to_entrance_plus_one(current_position, nearest_entrance, tunnel)
            else:
                return next_position

        if self.last_positions.get(robot_id, None) == current_position:
            state = self.get_state_representation(robot_id, current_position, tunnel)
            action = self.deadlock_solver.get_action(state)
            self.log(f"[handle_tunnel_collision] Deadlock détecté pour robot {robot_id} en situation de stagnation: action DL = {action}")
            return self.apply_learned_action(action, current_position, tunnel)

        return next_position

    def retreat_to_entrance_plus_one(self, current_position, entrance_tile, tunnel):
        map_obj = self._model.map()
        neighbors = map_obj.neighbours(entrance_tile)
        outside_neighbors = [tile for tile in neighbors if tile not in tunnel['tiles']]
        if outside_neighbors:
            return outside_neighbors[0]
        else:
            return entrance_tile

    def move_toward_without_tunnel_check(self, start, end):
        map_obj = self._model.map()
        best_neighbor = None
        min_dist = float('inf')
        for neighbor in map_obj.neighbours(start):
            dist = self._distances[neighbor][end]
            if dist < min_dist:
                min_dist = dist
                best_neighbor = neighbor
        return best_neighbor

    def is_deadlock(self, position, tunnel):
        robots_in_tunnel = []
        for robot_id in range(1, self._nb_robots + 1):
            robot_pos = self._model.mobilePosition(self._id, robot_id)
            if robot_pos in tunnel['tiles']:
                robots_in_tunnel.append(robot_id)
        return len(robots_in_tunnel) > 1

    def get_state_representation(self, robot_id, position, tunnel):
        state = np.zeros(self.deadlock_solver.state_size)
        robots_in_tunnel = []
        for rid in range(1, self._nb_robots + 1):
            rpos = self._model.mobilePosition(self._id, rid)
            if rpos in tunnel['tiles']:
                robots_in_tunnel.append((rid, rpos))
        if len(robots_in_tunnel) >= 2:
            state[0] = position
            other_robot = next(r for r in robots_in_tunnel if r[0] != robot_id)
            state[1] = other_robot[1]
            current_mission = self._model.mobileMission(self._id, robot_id)
            other_mission = self._model.mobileMission(self._id, other_robot[0])
            if current_mission != 0:
                state[2] = self._model.mission(current_mission).start
            if other_mission != 0:
                state[3] = self._model.mission(other_mission).start
        for i in range(9):
            state[i + 4] = 1 if i in tunnel['tiles'] else 0
        repetition = 1 if self.last_positions.get(robot_id, None) == position else 0
        state[13] = repetition
        # Mise à jour de l'historique des positions (on garde les 4 dernières valeurs)
        self.position_history[robot_id].append(position)
        if len(self.position_history[robot_id]) > 4:
            self.position_history[robot_id] = self.position_history[robot_id][-4:]
        # Ajout des distances aux bords pour une grille 3x3
        x = position % 3
        y = position // 3
        state[14] = x           # distance au bord gauche
        state[15] = 2 - x       # distance au bord droit
        state[16] = y           # distance au bord haut
        state[17] = 2 - y       # distance au bord bas
        self.last_positions[robot_id] = position
        return state

    def apply_learned_action(self, action, position, tunnel):
        mapping = {0: 12, 1: 6, 2: 3, 3: 9, 4: 0}
        move_code = mapping.get(action, 0)
        x = position % 3
        y = position // 3
        if move_code == 12:
            x_new, y_new = x, y - 1
        elif move_code == 6:
            x_new, y_new = x, y + 1
        elif move_code == 3:
            x_new, y_new = x + 1, y
        elif move_code == 9:
            x_new, y_new = x - 1, y
        else:
            x_new, y_new = x, y

        if 0 <= x_new < 3 and 0 <= y_new < 3:
            new_pos = y_new * 3 + x_new
        else:
            new_pos = position
        return new_pos

    #  Gestion générale des blocages en dehors des tunnels ---
    def handle_blockage(self, robot_id, current_position):
        """
        Gère une situation de blocage uniquement si le robot est arrêté sur la même case pendant plus de 3 tours
        consécutifs ou s'il présente une alternance répétitive entre deux positions (ex. A, B, A, B).
        Sinon, ne change pas son mouvement.
        """
        hist = self.position_history.get(robot_id, [])
        blocked = False
        # Condition 1: arrêt prolongé sur plus de 3 tours
        if len(hist) > 3 and all(pos == hist[0] for pos in hist[-4:]):
            blocked = True
        # Condition 2: alternance répétitive (ex: A, B, A, B)
        if len(hist) == 4 and hist[0] == hist[2] and hist[1] == hist[3] and hist[0] != hist[1]:
            blocked = True
        if not blocked:
            return current_position
        dummy_tunnel = {'tiles': set(range(9)), 'entrances': set()}
        state = self.get_state_representation(robot_id, current_position, dummy_tunnel)
        action = self.deadlock_solver.get_action(state)
        self.log(f"[handle_blockage] Blocage confirmé pour robot {robot_id} (arrêt >3 tours ou alternance répétitive): DL action = {action}")
        new_move = self.apply_learned_action(action, current_position, dummy_tunnel)
        return new_move
