import random
import time
import copy
import torch
import sys
import os

from collections import deque
from hacka.games.moveit import GameEngine

# classifier = os.path.abspath('./classifier')
# if classifier not in sys.path:
#     sys.path.insert(0, classifier)

from .classifier import VipMovementPredictor

class Node:
    def __init__(self, value, move=None, next_nodes=None):
        self._value = value
        self.move = move if move is not None else 0
        self._next_nodes = next_nodes if next_nodes is not None else []

    def __eq__(self, other):
        if isinstance(other, Node):
            return self._value == other._value
        return False

    def __hash__(self):
        return hash(self._value)

    @property
    def value(self):
        return self._value

    def add_node(self, node):
        if node not in self._next_nodes:
            self._next_nodes.append(node)
    
    def get_node(self, value):
        for node in self._next_nodes:
            if node.value == value:
                return node
            
        return None

    def remove_by_node(self, node):
        # Suppression sécurisée en recréant la liste (évite les erreurs d'indexation)
        self._next_nodes = [n for n in self._next_nodes if n != node]

    def __repr__(self, depth=0):
        result = f"({self._value}-{self.move})"
        for node in self._next_nodes:
            indent= '\t' * (depth+1)
            result += f"\n{indent}- {node.__repr__(depth=depth+1)}"
        return result


class SemiCompleteBot():
    # Player interface :
    def wakeUp(self, playerId, numberOfPlayers, gameConfiguration):
        self._history_length = 5
        self._id = playerId
        self._model = GameEngine()
        self._model.fromPod(gameConfiguration)  # Load the model from gameConfiguration
        self._model.render()
        self._nb_robots = self._model.numberOfMobiles(self._id)
        self._nb_players = numberOfPlayers
        self._debug = True
        self._depth = 5
        #self.log(f'number of robot for player {self._id}: {self._nb_robots}')
        self._is_vip_activated = self._model.numberOfMobiles(0) > 0

        s = self._model.map().size()
        self._distances = [[i for i in range(s + 1)]]
        #self.log(self._distances)

        for i in range(1, s+1):
            dist = self.computeDistances(i)
            #self.log(dist)
            self._distances.append(dist)
        
        self.init_ennemies_state()

        self._previous_vip_position = -1
        self._move_history = deque([-1] * self._history_length, maxlen=self._history_length)


        if self._is_vip_activated:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._vip_predictor = VipMovementPredictor()

            pthFile= os.path.dirname(os.path.realpath(__file__))
            pthFile+= "/classifier/weights/vip_movement_predictor.pth"

            self._vip_predictor.load_state_dict(torch.load(pthFile, map_location=self._device))
            self._vip_predictor.to(self._device) 
            self._vip_predictor.eval() 

            self._move_mapping = {-1: 0,  0: 1,  3: 2,  6: 3,  9: 4,  12: 5}
            self._inv_mapping = {v: k for k, v in self._move_mapping.items()}

    def perceive(self, gameState ):
        self._model.setOnState(gameState)
        self._model.render()
        self.init_ennemies_state(update=True)
        time.sleep(0.3)

    def decide(self):
        actions = self.get_actions()
        print(f"{self._model._tic} semi_complete_bot player-{self._id}({self._model.score(self._id)}): {actions}")
        #input()
        return actions
            
    def sleep(self, result):
        pass
        #self.log( f"end on : {result}" )
   
    # Mon implementation
    def init_ennemies_state(self, update=False):
        ennemies_state = {}
        
        # VIP
        if self._is_vip_activated:
            ennemies_state[0] = {1:{
                "passage_count": {
                    "total": [0 for _ in range(self._depth)]
                }
            }}
        
        # Other players
        for player_id in range(1, self._nb_players + 1):
            ennemies_state[player_id] = {}
            for robot_id in range(1, self._nb_robots + 1):
                ennemies_state[player_id][robot_id] = {
                    "passage_count": {
                        "total": [0 for _ in range(self._depth)]
                        }
                }
        
        self._dict_state = ennemies_state
        if update:
            if self._is_vip_activated:
                self.refresh_vip_positions()
                vip_next_moves = self.predict_next_vip_movement(self._depth-1)
                print(vip_next_moves)
                for i, (_, dest) in enumerate(vip_next_moves):
                    self.add_count_tile(0, 1, dest, i+1)
            for player_id in range(1, self._nb_players + 1):
                #print(f"player_{player_id}:")
                if player_id == self._id:
                    continue

                for robot_id in range(1, self._nb_robots + 1):
                    # print(f"\t-robot_{robot_id}:")
                    robot_position = self._model.mobilePosition(player_id, robot_id)
                    robot_mission = self._model.mobileMission(player_id, robot_id)
                    if robot_mission == 0:
                        for mission_id in self.select_valuable_missions(player_id, robot_id):
                            mission = self._model.mission(mission_id)
                            self.calc_all_paths(player_id, robot_id, robot_position, mission.start)
                    else:
                        mission = self._model.mission(robot_mission)
                        self.calc_all_paths(player_id, robot_id, robot_position, mission.final)

        return self._dict_state

    def add_count_tile(self, i_player, i_robot, i_tile, step_t):
        if i_tile not in self._dict_state[i_player][i_robot]["passage_count"]:
            self._dict_state[i_player][i_robot]["passage_count"][i_tile] = [0 for _ in range(self._depth)]
        self._dict_state[i_player][i_robot]["passage_count"][i_tile][step_t] += 1
        self._dict_state[i_player][i_robot]["passage_count"]["total"][step_t] += 1
    
    def get_tile_probability(self, i_player, i_robot, i_tile, step_t):
        try:
            numerator = self._dict_state[i_player][i_robot]["passage_count"][i_tile][step_t]
            denominator = self._dict_state[i_player][i_robot]["passage_count"]["total"][step_t]
        except Exception:
            numerator = 0
            denominator = 1
        
        return numerator/(denominator if denominator != 0 else 1)
    
    def calc_all_paths(self, player_i, robot_i, from_i, to_i, step_t=0):
        # Stop condition : if we're at the end then we stop the algo
        if from_i == to_i:
            return
        if step_t == self._depth:
            return

        self.add_count_tile(player_i, robot_i, from_i, step_t)
        
        next_tiles = [x for x in self._model._map.neighbours(from_i)]
        selected_nexts = [next_tiles[0]]
        min = self._distances[next_tiles[0]][to_i]
        for next_tile in next_tiles:
            current_distance = self._distances[next_tile][to_i] 
            if current_distance < min:
                min = current_distance
                selected_nexts = []
                selected_nexts.append(next_tile)
            elif current_distance == min:
                selected_nexts.append(next_tile)
        
        for tile_i in selected_nexts:
            self.calc_all_paths(player_i, robot_i, tile_i, to_i, step_t+1)
    
    def get_all_paths(self, from_i, to_i, move=None, current_node=None, step_t=0):
        if from_i == to_i or step_t == self._depth:
            return Node(from_i, move)
        
        if current_node is None:
            current_node = Node(from_i, move)
        
        next_tiles = [x for x in self._model._map.neighbours(from_i)]
        next_moves = self._model._map.clockBearing(from_i)
        if len(next_tiles) > 0:
            selected_nexts = [(next_tiles[0], next_moves[0])]
            min = self._distances[next_tiles[0]][to_i]
            for next_tile, next_move in zip(next_tiles, next_moves):
                current_distance = self._distances[next_tile][to_i] 
                if current_distance < min:
                    min = current_distance
                    selected_nexts = []
                    selected_nexts.append((next_tile, next_move))
                elif current_distance == min:
                    selected_nexts.append((next_tile, next_move))
            
            for tile_i, move_i in selected_nexts:
                next_node = current_node.get_node(tile_i)
                current_node.add_node(self.get_all_paths(tile_i, to_i, move_i, next_node, step_t+1))
        
        return current_node

    def get_next_step_from_node(self, head):
        return [(next_node.move, next_node.value) for next_node in head._next_nodes]

    def select_valuable_missions(self, player_id, robot_id):
        set_missions = set()
        robot_position = self._model.mobilePosition(player_id, robot_id)
        closest_missions = self.get_closest_missions(robot_position)
        high_ratio_missions = self.get_high_ratio_missions(player_id, robot_id)
        max_rewarded_missions = self.get_max_rewarded_mission()

        listes = [closest_missions, high_ratio_missions, max_rewarded_missions]
        for liste in listes:
            for mission in liste:
                set_missions.add(mission)
        
        missions = list(set_missions)
        return missions

    def calc_mission_reward_distance_ratio(self, player_i, robot_i, mission_i):
        robot_position = self._model.mobilePosition(player_i, robot_i)
        mission = self._model.mission(mission_i)

        dist_robot_mission = self._distances[robot_position][mission.start]
        dist_mission = self._distances[mission.start][mission.final]
        denominator = (dist_robot_mission + dist_mission)
        return mission.reward / denominator if denominator > 0 else 1
    
    def get_closest_missions(self, from_i):
        closest_missions = []
        mission_ids = self._model.missionsList()

        if len(mission_ids) == 0:
            return []
        
        mission = self._model.mission(0)
        min_distance = self._distances[from_i][mission.start]

        for mission_id in mission_ids:
            mission = self._model.mission(mission_id)
            current_distance = self._distances[from_i][mission.start]
            if current_distance < min_distance:
                min_distance = current_distance
                closest_missions = []
                closest_missions.append(mission_id)
            elif current_distance == min_distance:
                closest_missions.append(mission_id)
        
        return closest_missions

    def get_high_ratio_missions(self, player_i, robot_i):
        mission_ids = self._model.missionsList()

        if len(mission_ids) == 0:
            return []
        
        selected_missions = []
        max_ratio = self.calc_mission_reward_distance_ratio(player_i, robot_i, 0)

        for mission_id in mission_ids:
            current_ratio = self.calc_mission_reward_distance_ratio(player_i, robot_i, mission_id)
            if current_ratio > max_ratio:
                max_ratio = current_ratio
                selected_missions = []
                selected_missions.append(mission_id)
            elif current_ratio == max_ratio:
                selected_missions.append(mission_id)
        
        return selected_missions
    
    def get_max_rewarded_mission(self):
        mission_ids = self._model.missionsList()

        if len(mission_ids) == 0:
            return []
        
        selected_missions = []
        mission = self._model.mission(0)
        max_reward = mission.reward

        for mission_id in mission_ids:
            mission = self._model.mission(mission_id)
            current_reward = mission.reward
            if current_reward > max_reward:
                max_reward = current_reward
                selected_missions = []
                selected_missions.append(mission_id)
            elif current_reward == max_reward:
                selected_missions.append(mission_id)
        
        return selected_missions

    def log(self, message):
        '''
        #print the message if debug is activated

        Parameters:
        message (str): The message to #print

        Returns:
        None: Nothing is returned
        '''
        if self._debug:
            print(message)

    def get_actions(self, next_steps=None):
        '''
        This function will compute the next move for each player robot

        Parameters:
        reservations (list[int] or None): The list containing the reserved mission for each robot.
        next_steps (list[(int, int)] or None): The list containing the next action possible. the tuple contain the direction and the next tile.

        Returns:
        str: The action to perform foreach robot
        '''
        if next_steps is None:
            next_steps = self.get_next_steps_better(self._id)

        move_actions = []
        mission_actions = []
        actions = []
        for robot_id in range(1, self._nb_robots + 1):
            #self.log(f"deciding for {robot_id}")
            # robot_position = self._model.mobilePosition(self._id, robot_id)
            # robot_mission = self._model.mobileMission(self._id, robot_id)
            #self.log(f"\tposition: {robot_position}")
            #self.log(f"\tmission: {robot_mission}")
            current_action = ""
            
            next_moves = next_steps[robot_id]
            (action_keyword, next_move, _) = random.choice(next_moves)
            current_action = f"{action_keyword} {robot_id} {next_move}"


            #self.log(f"\taction: {current_action}")
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
        #self.log(f"{self._model._tic} - {actions}")
        return actions
    
    def get_missions_distances(self, player_id=1, all_mission_ids=None):
        result = [[]]

        if all_mission_ids is None:
            all_mission_ids = self._model.missionsList()
        
        #self.log(all_mission_ids)
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
            
        #self.log(f"Distances between robot and missions calculated for player_{player_id}:")
        #self.log(result)

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
        '''
        1. Récupère les missions les plus proches
        2. Si un autre robot est plus proche on le lui laisse peu importe son camp
        '''
        if all_mission_ids is None:
            all_mission_ids = self._model.missionsList()
        if robot_to_missions_distances is None:
            robot_to_missions_distances = self.get_all_players_missions_distances(all_mission_ids=all_mission_ids)
        
        prev_reservation = []
        # reservation = [-1 for _ in range(0, self._nb_robots+1)]
        reservation = [[]]
        for _ in range(1, self._nb_players + 1):
            reservation.append([-1 for _ in range(0, self._nb_robots+1)])

        max_iter = 1000

        counter = 0
        # While the solution not converge
        while prev_reservation != reservation:
            prev_reservation = copy.deepcopy(reservation)
            #self.log(f"{counter} - reservations: {reservation}")
            counter += 1
            if counter > max_iter:
                break
            
            for player_id in range(1, self._nb_players+1):
                for robot_id in range(1, self._nb_robots+1):
                    robot_mission = self._model.mobileMission(player_id, robot_id)
                    # Si le robot a déjà activé une mission
                    if robot_mission != 0:
                        reservation[player_id][robot_id] = robot_mission
                    # Si le robot a déjà réservé une mission
                    if reservation[player_id][robot_id] > 0:
                        continue
                
                    # On récupère la mission la plus proche et libre
                    distances = robot_to_missions_distances[player_id][robot_id]
                    print(f"\n(player_{player_id}) reserved missions: {reservation[player_id]}")
                    print(f"(player_{player_id}, robot_{robot_id}) distances to missions: {distances}")
                    min_distance = max(distances) + 1
                    selected_mission_id = None
                    for i, distance in enumerate(distances):
                        # If the mission has already been activated
                        if distance < 0:
                            continue
                        # If the mission is already assigned
                        if i in reservation[player_id]:
                            continue

                        if distance < min_distance:
                            min_distance = distance
                            selected_mission_id = i
                    
                    print(f"(player_{player_id}, robot_{robot_id}) mission la plus proche: {selected_mission_id} with distance of {min_distance}")
                    
                    # If a mission has been selected
                    if selected_mission_id != None:
                        is_closest = True
                        # Check if another robot is closer than the current one
                        for other_player_id in range(1, self._nb_players + 1):
                            for other_robot_id in range(1, self._nb_robots + 1):
                                other_robot_mission = self._model.mobileMission(other_player_id, other_robot_id)
                                # Si le robot a une mission
                                if other_robot_mission > 0:
                                    reservation[other_player_id][other_robot_id] = other_robot_mission
                                # If the robot has a mission assigned
                                if reservation[other_player_id][other_robot_id] > 0:
                                    continue
                                if other_player_id == player_id and other_robot_id == robot_id:
                                    continue

                                other_robot_distances = robot_to_missions_distances[other_player_id][other_robot_id]
                                if other_robot_distances[selected_mission_id] < min_distance:
                                    print(f"(player_{player_id}, robot_{robot_id}) is not the closest of {selected_mission_id} it is (player_{other_player_id}, robot_{other_robot_id})")
                                    is_closest = False
                                    break
                            
                            if not is_closest:
                                break
                            
                        if is_closest:
                            reservation[player_id][robot_id] = selected_mission_id

        
        #self.log(f"reservations:\n{reservation}")
        return reservation[self._id]
    
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
            
            # Check if another robot is already going on the tile == avoid a collision between two robots
            for other_player_id in range(1, self._nb_players + 1):
                if other_player_id == self._id:
                    continue

                for other_robot_id in range(1, self._nb_robots + 1):
                    moves_to_remove = []

                    # We loop through the other robot moves to check if there is a collision
                    for i, (_, _, current_next_tile) in enumerate(current_next_moves):
                        for (_, _, other_robot_tile) in all_next_steps[other_player_id][other_robot_id]:
                            if current_next_tile == other_robot_tile:
                                if i not in moves_to_remove:
                                    moves_to_remove.insert(0, i)
                    
                    for i in moves_to_remove:
                        #self.log(f"remove {i} in {current_next_moves}")
                        del current_next_moves[i]
            
            # if no move is left, then it should stay at the same position
            if len(current_next_moves) <= 0:
                current_next_moves.append(("move", 0, self._model.mobilePosition(self._id, robot_id)))
        
        return next_step

    def is_someone_on(self, tile_i):
        if self._is_vip_activated:
            vip_position = self._model.mobilePosition(0, 1)
            if vip_position == tile_i:
                return True, True
        
        for player_id in range(1, self._model.numberOfPlayers() + 1):
            for robot_id in range(1, self._model.numberOfMobiles() + 1):
                robot_position = self._model.mobilePosition(player_id, robot_id)
                if robot_position == tile_i:
                    return True, False
        
        return False, False
    
    def get_next_steps_better(self, player_id=1, reservations=None):
        #self.log("get_next_steps_better")
        if reservations is None:
            reservations = self.assign_missions()
        
        next_moves=[[] for _ in range(self._nb_robots + 1)]
        for robot_id in range(1, self._nb_robots + 1):
            #self.log(f"{robot_id} robot:")
            robot_position = self._model.mobilePosition(player_id, robot_id)
            robot_mission_id = self._model.mobileMission(player_id, robot_id)
            if reservations[robot_id] < 0:
                #self.log(f"\tNo reserved mission")
                current_next_moves = [('move', 0, robot_position)]
            elif robot_mission_id == 0 and reservations[robot_id] > 0:
                #self.log(f"\treserved mission {reservations[robot_id]}")
                robot_mission = self._model.mission(reservations[robot_id])
                if robot_position == robot_mission.start:
                    current_next_moves = [('mission', reservations[robot_id], robot_position)]
                else:
                    head = self.get_all_paths(robot_position, robot_mission.start)
                    print(head)
                    head, _ = self.detect_potential_collisions(player_id, robot_id, head, 1)
                    print(head)
                    current_next_moves = self.get_next_step_from_node(head)
                    current_next_moves = [('move', *move) for move in current_next_moves]
            elif robot_mission_id > 0:
                robot_mission = self._model.mission(robot_mission_id)
                if robot_position == robot_mission.final:
                    current_next_moves = [('mission', robot_mission_id, robot_position)]
                else:
                    head = self.get_all_paths(robot_position, robot_mission.final)
                    print(head)
                    head, _ = self.detect_potential_collisions(player_id, robot_id, head, 1)
                    print(head)
                    current_next_moves = self.get_next_step_from_node(head)
                    current_next_moves = [('move', *move) for move in current_next_moves]
            
            if self._is_vip_activated:
                vip_position = self._model.mobilePosition(0, 1)
                current_dist = self._distances[robot_position][vip_position]
                if current_dist <= 3:
                    tile = self._model._map.tile(robot_position)
                    neighbours = [x for x in tile.adjacencies()]
                    print(f"neighbours: {neighbours}")
                    directions = self._model._map.clockBearing(robot_position)
                    if 0 in directions:
                        i_zero = directions.index(0)
                        del neighbours[i_zero]
                        del directions[i_zero]
                    
                    to_remove_indexes = []
                    for i, neighbour in enumerate(neighbours):
                        if self._distances[neighbour][vip_position] < current_dist:
                            if i not in to_remove_indexes:
                                to_remove_indexes.append(i)
                                break
                    
                    to_remove_indexes.sort(reverse=True)
                    to_remove_dir = []
                    for i in to_remove_indexes:
                        to_remove_dir.append(directions[i])
                        del directions[i]
                        del neighbours[i]
                    
                    print(f"directions to run: {directions}")
                    #input()

                    if len(to_remove_dir) > 0:
                        to_remove_indexes = []
                        for  i, (_, next_dir, _) in enumerate(current_next_moves):
                            if next_dir in to_remove_dir:
                                to_remove_indexes.append(i)
                        
                        to_remove_indexes.sort(reverse=True)
                        for i in to_remove_indexes:
                            del current_next_moves[i]
                
                    for direction, neighbour in zip(directions, neighbours):
                        exist = False
                        for _, next_dir, _ in current_next_moves:
                            if direction == next_dir:
                                exist = True
                                break
                        if not exist:
                            print("add: ", direction)
                            current_next_moves.append(("move", direction, neighbour))
                    
            print(f"\tcurrent_next_moves: {current_next_moves}")
            #self.log(f"\tcurrent_next_moves: {current_next_moves}")
            # Check if another robot is already going on the tile == avoid a collision between two robots
            for other_robot_id in range(1, self._nb_robots + 1):
                # If the other robot is the same then it is not a collision
                if other_robot_id == robot_id:
                    continue
                
                other_robot_moves_to_remove = []
                moves_to_remove = []
                other_robot_position = self._model.mobilePosition(player_id, other_robot_id)
                # We loop through the other robot moves to check if there is a collision
                for i, (_, _, current_next_tile) in enumerate(current_next_moves):
                    # print(f"\robot-{robot_id} current_next_tile: {current_next_tile}")
                    # print(f"\robot-{other_robot_id} position: {other_robot_position}")
                    if current_next_tile == other_robot_position:
                        if i not in moves_to_remove:
                            moves_to_remove.insert(0, i)
                            continue
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
                    #self.log(f"remove {i} in {current_next_moves}")
                    del current_next_moves[i]
                for i in other_robot_moves_to_remove:
                    #self.log(f"remove {i} in {other_robot_moves_to_remove}")
                    del next_moves[other_robot_id][i]
            
            moves_to_remove = []
            for other_player_id in range(0, self._nb_players + 1):
                if other_player_id == player_id:
                    continue
                
                other_robots_positions = self._model._map.mobilePositions(other_player_id)
                for i, (_, _, current_next_tile) in enumerate(current_next_moves):
                    if current_next_tile in other_robots_positions:
                        if i not in moves_to_remove:
                            moves_to_remove.append(i)
                            continue
            
            moves_to_remove.sort(reverse=True)
            for i in moves_to_remove:
                #self.log(f"remove {i} in {current_next_moves}")
                del current_next_moves[i]
            
            if (len(current_next_moves) == 1 and current_next_moves[0][1] == 0) or len(current_next_moves) <= 0:
                tile = self._model._map.tile(robot_position)
                neighbours = [x for x in tile.adjacencies()]
                directions = self._model._map.clockBearing(robot_position)
                if 0 in neighbours:
                    i_zero = neighbours.index(0)
                    del neighbours[i_zero]
                    del directions[i_zero]
                to_remove_indexes = []
                for i, neighbour  in enumerate(neighbours):
                    is_someone_on, _ = self.is_someone_on(neighbour)
                    if is_someone_on:
                        if i not in to_remove_indexes:
                            to_remove_indexes.append(i)
                
                to_remove_indexes.sort(reverse=True)
                for i in to_remove_indexes:
                    del neighbours[i]
                    del directions[i]
                
                if len(directions) > 0:
                    i_rand = random.randint(0, len(directions) - 1)
                    current_next_moves = [('move', directions[i_rand], neighbours[i_rand])]

            # if no move is left, then it should stay at the same position
            if len(current_next_moves) <= 0:
                current_next_moves.append(("move", 0, robot_position))
            
            #self.log(f"\t- next_moves: {current_next_moves}")
            next_moves[robot_id] = current_next_moves
    
        return next_moves
                
    def get_next_steps(self, player_id=1, reservations=None):
        if reservations is None:
            reservations = self.assign_missions()
        
        robot_positions = [self._model.mobilePosition(player_id, i) for i in range(1, self._nb_robots + 1)]
        robot_positions.insert(0, -1)

        map_centers=[]
        next_moves=[[] for _ in range(self._nb_robots + 1)]
        for robot_id in range(1, self._nb_robots + 1):
            #self.log(f"{robot_id} robot:")
            robot_position = robot_positions[robot_id] 
            robot_mission_id = self._model.mobileMission(player_id, robot_id)
            
            current_next_moves = []
            # if robot do not have a reserved mission, then it should go on the center of the map
            if reservations[robot_id] < 0:
                #self.log("\tno reserved missions")
                map_center = int((self._model._map.size() / 2)) + 1
                c = 1
                while map_center in map_centers:
                    map_center = map_center + c
                    c += 1
                    c *= -1
                map_centers.append(map_center)
                #self.log(f"\tgo to map center at cell {map_center}")
                current_next_moves = self.move_toward(robot_position, map_center)
                current_next_moves = [('move', *move) for move in current_next_moves]
            # If robot reserved a mission, then it should go on its start position
            elif robot_mission_id == 0 and reservations[robot_id] > 0:
                #self.log(f"\thas reserved {reservations[robot_id]} mission")
                robot_mission = self._model.mission(reservations[robot_id])
                #self.log(f"\tgo to {robot_mission.start} cell")
                if robot_position == robot_mission.start:
                    current_next_moves = [('mission', reservations[robot_id], robot_position)]
                else:
                    current_next_moves = self.move_toward(robot_position, robot_mission.start)
                    current_next_moves = [('move', *move) for move in current_next_moves]
            # If robot activated a mission, then it should go on its final position
            elif robot_mission_id > 0:
                #self.log(f"\thas activated {robot_mission_id} mission")
                robot_mission = self._model.mission(robot_mission_id)
                #self.log(f"\tgo to {robot_mission.final} cell")
                if robot_position == robot_mission.final:
                    current_next_moves = [('mission', robot_mission_id, robot_position)]
                else:
                    current_next_moves = self.move_toward(robot_position, robot_mission.final)
                    current_next_moves = [('move', *move) for move in current_next_moves]
            
            # Check if another robot is already going on the tile == avoid a collision between two robots
            for other_robot_id in range(1, robot_id + 1):
                # If the other robot is the same then it is not a collision
                if other_robot_id == robot_id:
                    continue
                
                other_robot_moves_to_remove = []
                moves_to_remove = []
                # We loop through the other robot moves to check if there is a collision
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
                    #self.log(f"remove {i} in {current_next_moves}")
                    del current_next_moves[i]
                for i in other_robot_moves_to_remove:
                    #self.log(f"remove {i} in {other_robot_moves_to_remove}")
                    del next_moves[other_robot_id][i]
            
            # if no move is left, then it should stay at the same position
            if len(current_next_moves) <= 0:
                current_next_moves.append(("move", 0, robot_position))

            #self.log(f"\t- next_moves: {current_next_moves}")
            next_moves[robot_id] = current_next_moves
        
        return next_moves

    def detect_potential_collisions(self, player_i, robot_i, path, step_t=0):
        current_node = path
        print(f"current_node: {current_node.value}")
        robot_position = self._model.mobilePosition(player_i, robot_i)
        to_remove_at = []
        remove_current = False
        for i, next_node in enumerate(current_node._next_nodes):
            to_remove = False
            for other_player_id in range(0, self._nb_players + 1):
                if other_player_id == player_i:
                    continue
                for other_robot_id in range(1, self._model.numberOfMobiles(other_player_id)):
                    other_robot_position = self._model.mobilePosition(other_player_id, other_robot_id)
                    distance_to_robot = self._distances[robot_position][other_robot_position]
                    # print(self._dict_state[other_player_id][other_robot_id])
                    if distance_to_robot % 2 == 0:
                        proba_other_robot = self.get_tile_probability(other_player_id, other_robot_id, current_node.value, step_t)
                    else:
                        proba_other_robot= self.get_tile_probability(other_player_id, other_robot_id, current_node.value, step_t+1)
                    # print(f"proba_first: {proba_other_robot_first}")
                    # print(f"proba_second: {proba_other_robot_second}")
                    if proba_other_robot >= 0.2:
                        to_remove = True
                        print(f"remove {next_node.value}")
                        to_remove_at.append(i)
                
                    if to_remove:
                        break
                if to_remove:
                    break
        
        to_remove_at.reverse()
        for i in to_remove_at:
            del current_node._next_nodes[i] 
        
        if len(current_node._next_nodes) <= 0:
            all_missions = self._model.missions()
            remove_current = True
            for mission in all_missions:
                if mission.start == current_node.value or mission.final == current_node.value:
                    remove_current = False
                if not remove_current:
                    break
        else:
            other_remove = []
            for i, next_node in enumerate(current_node._next_nodes):
                _, remove_node = self.detect_potential_collisions(player_i, robot_i, next_node, step_t+1)
                if remove_node:
                    other_remove.append(i)
            
            other_remove.reverse()
            for i in other_remove:
                del current_node._next_nodes[i]

        return current_node, remove_current

    def move_toward(self, iTile, iTarget):
        # If no need to move:
        if iTile == iTarget :
            return [(0, iTile)]

        result=[]
        # Get candidates:
        clockdirs= self._model.map().clockBearing(iTile)
        nextTiles= self._model.map().neighbours(iTile)
        selectedDir= clockdirs[0]
        selectedNext= nextTiles[0]
        robot_positions = [[]]
        for player_id in range(1, self._nb_players + 1):
            player_robot_positions = [self._model.mobilePosition(player_id, i) for i in range(1, self._nb_robots + 1)]
            robot_positions.append(player_robot_positions)
        # Test all candidates:
        for clock, tile in zip( clockdirs, nextTiles ):
            # If there already is a robot in the cell then we do not go on it
            if any(tile in sub_positions for sub_positions in robot_positions):
                continue

            if self._distances[tile][iTarget] < self._distances[selectedNext][iTarget] :
                result=[]
                selectedDir= clock
                selectedNext= tile
                result.append((selectedDir, selectedNext))
            elif self._distances[tile][iTarget] == self._distances[selectedNext][iTarget] :
                selectedDir= clock
                selectedNext= tile
                result.append((selectedDir, selectedNext))

        if len(result) == 0:
            result.append((0, iTile))
        
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
    
    def get_area_around(self, i_tile):
        tile = self._model._map.tile(i_tile)
        sub_matrice = [[0 for _ in range(3)] for _ in range(3)]
        adjencies = tile.adjacencies()
        directions = self._model._map.clockBearing(i_tile)
        if 12 not in directions or 9 not in directions or 3 not in directions or 6 not in directions:
            if 12 not in directions:
                sub_matrice[0][1] = -1
            else:
                top_i = adjencies[directions.index(12)]
                top_directions = self._model._map.clockBearing(top_i)
                if 9 not in top_directions:
                    sub_matrice[0][0] = -1
                if 3 not in top_directions:
                    sub_matrice[0][2] = 0
            
            if 9 not in directions:
                sub_matrice[1][0] = -1
            else:
                left_i = adjencies[directions.index(9)]
                left_directions = self._model._map.clockBearing(left_i)
                if 12 not in left_directions:
                    sub_matrice[0][0] = -1
                if 6 not in left_directions:
                    sub_matrice[2][0] = -1
            
            if 3 not in directions:
                sub_matrice[1][2] = -1
            else:
                right_i = adjencies[directions.index(3)]
                right_directions = self._model._map.clockBearing(right_i)
                if 12 not in right_directions:
                    sub_matrice[0][2] = -1
                if 6 not in right_directions:
                    sub_matrice[2][2] = -1
            
            if 6 not in directions:
                sub_matrice[2][1] = -1
            else:
                bot_i = adjencies[directions.index(6)]
                bot_directions = self._model._map.clockBearing(bot_i)
                if 9 not in bot_directions:
                    sub_matrice[2][0] = -1
                if 3 not in bot_directions:
                    sub_matrice[2][2] = -1
            
            if 12 not in directions and 3 not in directions:
                sub_matrice[0][2] = -1
            if 12 not in directions and 9 not in directions:
                sub_matrice[0][0] = -1
            if 6 not in directions and 3 not in directions:
                sub_matrice[2][2] = -1
            if 6 not in directions and 9 not in directions:
                sub_matrice[2][0] = -1

        return sub_matrice

    def refresh_vip_positions(self):
        if self._is_vip_activated:
            current_position = self._model.mobilePosition(0, 1)
            
            if self._previous_vip_position >= 0:
                possible_moves = self._model._map.clockBearing(self._previous_vip_position)
                adjacent_tiles = [x for x in self._model._map.neighbours(self._previous_vip_position)]

                for move, tile in zip(possible_moves, adjacent_tiles):
                    if tile == current_position:
                        movement_sequence = [str(move) for move in self._move_history]  # Convert history to strings
                        movement_sequence.append(str(move))  # Append current move

                        self._move_history.append(move)  # Update history
                        break
            
            self._previous_vip_position = current_position
    
    def predict_next_vip_movement(self, depth):
        if self._is_vip_activated:
            vip_position = self._model.mobilePosition(0, 1)
            sub_matrice = self.get_area_around(vip_position)
            result = []
            
            move_history = deque(maxlen=self._history_length)
            for move in self._move_history:
                move_history.append(move)
            
            for _ in range(depth):
                # Exemple d'utilisation sur une nouvelle situation (env_new, moves_new)
                env_new = torch.tensor([[element for row in sub_matrice for element in row]], dtype=torch.float32).to(self._device)  # environnement 3x3 (à adapter)
                moves_new = torch.tensor([[self._move_mapping[move] for move in move_history]], dtype=torch.long).to(self._device)  # derniers mouvements encodés (exemple)
                with torch.no_grad():
                    output = self._vip_predictor(env_new, moves_new)
                    predicted_class = torch.argmax(output, dim=1).item()
                # Conversion inverse de l'encodage pour obtenir le mouvement prédict en valeur originale
                predicted_move = self._inv_mapping[predicted_class]

                vip_current_tile = self._model._map.tile(vip_position)
                directions = self._model._map.clockBearing(vip_position)

                if predicted_move in directions: 
                    adjencies = vip_current_tile.adjacencies()
                    vip_position = adjencies[directions.index(predicted_move)]
                    result.append((predicted_move, vip_position))
                    move_history.append(predicted_move)
                    sub_matrice = self.get_area_around(vip_position)
            
            return result