import random
import time
import copy
import torch
import os
from collections import deque
from hacka.games.moveit import GameEngine

from .classifier import VipMovementPredictor


class Node:
    """
    Représente un nœud d'un arbre ou d'un graphe, utilisé pour le pathfinding et l'organisation
    des déplacements potentiels.

    Attributes
    ----------
    _value : int
        L'identifiant (numéro de tuile) du nœud courant.
    move : int
        L'action ou la direction associée au nœud (par exemple 0, 3, 6, 9, 12 pour les directions).
    _next_nodes : list of Node
        La liste des nœuds enfants, représentant les transitions possibles depuis ce nœud.
    """

    def __init__(self, value, move=None, next_nodes=None):
        """
        Initialise un objet Node.

        Parameters
        ----------
        value : int
            Identifiant ou valeur du nœud (numéro de la tuile).
        move : int, optional
            Mouvement (direction) ou action associée à ce nœud, par défaut 0.
        next_nodes : list of Node, optional
            Liste des nœuds enfants (pour les chemins possibles), par défaut [].
        """
        self._value = value
        self.move = move if move is not None else 0
        self._next_nodes = next_nodes if next_nodes is not None else []

    def __eq__(self, other):
        """
        Vérifie l'égalité entre deux nœuds sur la base de la même valeur.
        """
        if isinstance(other, Node):
            return self._value == other._value
        return False

    def __hash__(self):
        """
        Permet l'utilisation de Node comme clé dans un dictionnaire ou un ensemble (set).
        """
        return hash(self._value)

    @property
    def value(self):
        """
        Retourne la valeur (tuile) associée au nœud.
        """
        return self._value

    def add_node(self, node):
        """
        Ajoute un nœud enfant à la liste `_next_nodes` s'il n'existe pas déjà.

        Parameters
        ----------
        node : Node
            Le nœud enfant à ajouter.
        """
        if node not in self._next_nodes:
            self._next_nodes.append(node)
    
    def get_node(self, value):
        """
        Recherche un nœud enfant par sa valeur (tuile).

        Parameters
        ----------
        value : int
            La valeur (tuile) du nœud recherché.

        Returns
        -------
        Node or None
            Le nœud correspondant s'il est trouvé, sinon None.
        """
        for node in self._next_nodes:
            if node.value == value:
                return node
        return None

    def remove_by_node(self, node):
        """
        Supprime un nœud enfant spécifique en recréant la liste pour éviter les problèmes d'index.

        Parameters
        ----------
        node : Node
            Le nœud à supprimer de `_next_nodes`.
        """
        self._next_nodes = [n for n in self._next_nodes if n != node]

    def __repr__(self, depth=0):
        """
        Représentation textuelle récursive du nœud et de ses descendants.

        Parameters
        ----------
        depth : int, optional
            Profondeur actuelle, utilisée pour l'indentation visuelle, par défaut 0.

        Returns
        -------
        str
            Représentation textuelle du graphe de nœuds.
        """
        result = f"({self._value}-{self.move})"
        for node in self._next_nodes:
            indent= '\t' * (depth+1)
            result += f"\n{indent}- {node.__repr__(depth=depth+1)}"
        return result


class CompleteBot():
    """
    Bot complet pour la gestion des déplacements et missions dans le jeu.
    Utilise un moteur de jeu (GameEngine), divers algorithmes de pathfinding,
    et un prédicteur de mouvements VIP (VipMovementPredictor).

    Attributes
    ----------
    _debug : bool
        Indique si le bot est en mode debug (avec affichage détaillé).
    _depth : int
        Profondeur maximale pour l'anticipation des déplacements ennemis.
    _history_length : int
        Nombre de déplacements du VIP à mémoriser pour la prédiction.
    _id : int
        Identifiant du joueur (bot).
    _model : GameEngine
        Le moteur de jeu, chargé de l'état actuel et des informations de la carte.
    _nb_robots : int
        Nombre de robots contrôlés par le bot.
    _nb_players : int
        Nombre total de joueurs dans la partie (incluant ce bot et le VIP s'il y en a un).
    _is_vip_activated : bool
        Indique si la mécanique de VIP est activée dans la partie.
    _distances : list of list
        Matrice de distances précalculées entre toutes les tuiles.
    _dict_state : dict
        Stocke les informations de passage probable sur chaque tuile pour chaque robot/ennemi.
    _previous_vip_position : int
        Dernière position connue du VIP.
    _move_history : collections.deque
        Historique des déplacements du VIP (pour la prédiction).
    _device : torch.device
        Dispositif (GPU ou CPU) pour PyTorch.
    _vip_predictor : VipMovementPredictor
        Modèle de prédiction pour les déplacements VIP.
    _move_mapping : dict
        Conversion des mouvements bruts en indices de classe.
    _inv_mapping : dict
        Inverse de `_move_mapping`.
    _reservations : list
        Missions réservées/assignées par robot (pour ce bot).
    """

    def __init__(self, debug=False, depth=5):
        """
        Initialise le bot avec le niveau de débogage et la profondeur de calcul.

        Parameters
        ----------
        debug : bool, optional
            Active le mode debug (affichages de logs), par défaut False.
        depth : int, optional
            Profondeur d'anticipation des déplacements ennemis, par défaut 5.
        """
        self._debug = debug
        self._depth = depth

    # Méthodes Player interface :
    def wakeUp(self, playerId, numberOfPlayers, gameConfiguration):
        """
        Initialise les paramètres du bot à la création de la partie (ou au réveil du bot).

        Parameters
        ----------
        playerId : int
            Identifiant du bot/joueur.
        numberOfPlayers : int
            Nombre total de joueurs dans la partie.
        gameConfiguration : dict
            Configuration de la partie, transmise par le moteur de jeu.
        """
        self._history_length = 5
        self._id = playerId
        self._model = GameEngine()
        self._model.fromPod(gameConfiguration)  # Charge la configuration depuis le gameConfiguration

        if self._debug:
            self._model.render()
        
        self._nb_robots = self._model.numberOfMobiles(self._id)
        self._nb_players = numberOfPlayers
        self._is_vip_activated = self._model.numberOfMobiles(0) > 0

        # Calcul des distances
        s = self._model.map().size()
        self._distances = [[i for i in range(s + 1)]]
        self.log(self._distances)

        for i in range(1, s+1):
            dist = self.computeDistances(i)
            self.log(dist)
            self._distances.append(dist)
        
        # Initialise le dictionnaire de passage des ennemis
        self.init_ennemies_state()

        # Initialise le module de prédiction VIP si VIP activé
        if self._is_vip_activated:
            self._previous_vip_position = -1
            self._move_history = deque([-1] * self._history_length, maxlen=self._history_length)

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._vip_predictor = VipMovementPredictor()

            # Sélection du fichier de poids en fonction de la taille de la carte
            if s < 15:
                map_size = "small"
            elif s < 40:
                map_size = "medium"
            else:
                map_size = "large"

            pthFile= os.path.dirname(os.path.realpath(__file__))
            pthFile+= f"/classifier/weights/vip_movement_{map_size}_predictor.pth"

            self._vip_predictor.load_state_dict(torch.load(pthFile, map_location=self._device))
            self._vip_predictor.to(self._device) 
            self._vip_predictor.eval() 

            # Mappings pour la prédiction
            self._move_mapping = {-1: 0,  0: 1,  3: 2,  6: 3,  9: 4,  12: 5}
            self._inv_mapping = {v: k for k, v in self._move_mapping.items()}

    def perceive(self, gameState ):
        """
        Met à jour l'état du bot en fonction du nouvel état de la partie (gameState).

        Parameters
        ----------
        gameState : dict
            État du jeu (informations sur les joueurs, missions, positions, etc.).
        """
        self._model.setOnState(gameState)
        if self._debug:
            self._model.render()
            time.sleep(0.3)

        # Met à jour la logique d'état des ennemis
        self.init_ennemies_state(update=True)

    def decide(self):
        """
        Génère la décision (action) du bot pour ce tour.

        Returns
        -------
        str
            Chaîne de caractères représentant l'action (ou les actions) à réaliser
            (missions et moves pour chaque robot).
        """
        actions = self.get_actions()
        self.log(f"{self._model._tic} semi_complete_bot player-{self._id}({self._model.score(self._id)}): {actions}")
        return actions
            
    def sleep(self, result):
        """
        Méthode appelée à la fin de la partie ou quand le bot se rendort.

        Parameters
        ----------
        result : any
            Résultat final ou état de fin.
        """
        self.log(f"end on : {result}")
   
    # --- Méthodes internes ---

    # ------------------------------------------------------------------------
    # 2. Gestion du VIP
    # ------------------------------------------------------------------------
    def get_area_around(self, i_tile):
        """
        Extrait une zone (3x3) autour de la tuile i_tile pour la prédiction VIP. 
        Gère les cas de bords où certaines directions ne sont pas disponibles.

        Parameters
        ----------
        i_tile : int
            Tuile centrale.

        Returns
        -------
        list of list
            Matrice 3x3, avec -1 pour indiquer une absence de tuile.
        """
        sub_matrice = [[0 for _ in range(3)] for _ in range(3)]
        adjencies = [x for x in self._model._map.neighbours(i_tile)]
        directions = self._model._map.clockBearing(i_tile)

        # Pour chaque direction non disponible, on met -1 dans la matrice
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
        """
        Met à jour la position précédente du VIP et enregistre le dernier mouvement
        dans l'historique (_move_history).
        """
        if self._is_vip_activated:
            current_position = self._model.mobilePosition(0, 1)
            
            if self._previous_vip_position >= 0:
                possible_moves = self._model._map.clockBearing(self._previous_vip_position)
                adjacent_tiles = [x for x in self._model._map.neighbours(self._previous_vip_position)]

                # On trouve le move correspondant à la transition entre l'ancienne position et la nouvelle
                for move, tile in zip(possible_moves, adjacent_tiles):
                    if tile == current_position:
                        movement_sequence = [str(move) for move in self._move_history]
                        movement_sequence.append(str(move))

                        self._move_history.append(move)
                        break
            
            self._previous_vip_position = current_position
    
    def predict_next_vip_movement(self, depth):
        """
        Prédit les prochains déplacements du VIP sur un certain nombre d'étapes (depth),
        à l'aide du modèle _vip_predictor.

        Parameters
        ----------
        depth : int
            Nombre d'étapes (déplacements) à prédire.

        Returns
        -------
        list of tuple
            Liste de tuples (direction, position) pour chaque étape prédite.
        """
        if self._is_vip_activated:
            # Position initiale
            vip_position = self._model.mobilePosition(0, 1)

            # Zone locale autour du VIP
            sub_matrice = self.get_area_around(vip_position)
            result = []
            
            # Copie de l'historique local
            move_history = deque(maxlen=self._history_length)
            for move in self._move_history:
                move_history.append(move)
            
            # Prédiction pour plusieurs coups
            for _ in range(depth):
                env_new = torch.tensor([[element for row in sub_matrice for element in row]], 
                                       dtype=torch.float32).to(self._device)
                moves_new = torch.tensor([[self._move_mapping[move] for move in move_history]], 
                                         dtype=torch.long).to(self._device)

                with torch.no_grad():
                    output = self._vip_predictor(env_new, moves_new)
                    predicted_class = torch.argmax(output, dim=1).item()

                # Conversion de l'indice en direction (0, 3, 6, 9, 12)
                predicted_move = self._inv_mapping[predicted_class]

                # Mise à jour de la position si mouvement valide
                vip_current_tile = self._model._map.tile(vip_position)
                directions = self._model._map.clockBearing(vip_position)
                if predicted_move in directions:
                    adjencies = vip_current_tile.adjacencies()
                    vip_position = adjencies[directions.index(predicted_move)]
                    result.append((predicted_move, vip_position))
                    move_history.append(predicted_move)

                    # Mise à jour de la zone locale
                    sub_matrice = self.get_area_around(vip_position)

            return result
        
    # ------------------------------------------------------------------------
    # 3. Gestion des Missions
    # ------------------------------------------------------------------------
    def assign_missions(self, robot_to_missions_distances=None, all_mission_ids=None):
        """
        Assigne les missions aux robots du bot en se basant sur la proximité et la compétition
        entre robots (pour déterminer le plus proche).

        Parameters
        ----------
        robot_to_missions_distances : list, optional
            Matrice de distances [player_id][robot_id][mission_id].
        all_mission_ids : list of int, optional
            Liste des missions à assigner.

        Returns
        -------
        list
            Liste d'affectations de mission par robot (indice = robot_id).
        """
        if all_mission_ids is None:
            all_mission_ids = self._model.missionsList()
        if robot_to_missions_distances is None:
            robot_to_missions_distances = self.get_all_players_missions_distances(all_mission_ids=all_mission_ids)
        
        prev_reservation = []
        reservation = [[]]
        for _ in range(1, self._nb_players + 1):
            reservation.append([-1 for _ in range(0, self._nb_robots+1)])

        max_iter = 1000
        counter = 0

        # Boucle jusqu'à convergence ou max_iter
        while prev_reservation != reservation:
            prev_reservation = copy.deepcopy(reservation)
            counter += 1
            if counter > max_iter:
                break
            
            # Pour chaque robot de chaque joueur
            for player_id in range(1, self._nb_players+1):
                for robot_id in range(1, self._nb_robots+1):
                    robot_mission = self._model.mobileMission(player_id, robot_id)
                    # Si déjà une mission active, on fixe
                    if robot_mission != 0:
                        reservation[player_id][robot_id] = robot_mission
                    if reservation[player_id][robot_id] > 0:
                        continue

                    # Recherche de la mission la plus proche et libre
                    distances = robot_to_missions_distances[player_id][robot_id]
                    self.log(f"\n(player_{player_id}) reserved missions: {reservation[player_id]}")
                    self.log(f"(player_{player_id}, robot_{robot_id}) distances to missions: {distances}")
                    min_distance = max(distances) + 1
                    selected_mission_id = None
                    for i, distance in enumerate(distances):
                        if distance < 0:  # mission déjà prise ou invalide
                            continue
                        if i in reservation[player_id]:  # déjà réservée
                            continue
                        if distance < min_distance:
                            min_distance = distance
                            selected_mission_id = i
                    
                    self.log(f"(player_{player_id}, robot_{robot_id}) mission la plus proche: {selected_mission_id} with distance of {min_distance}")
                    
                    # Si on a sélectionné une mission
                    if selected_mission_id != None:
                        is_closest = True
                        # Vérifie qu'aucun autre robot n'est plus proche
                        for other_player_id in range(1, self._nb_players + 1):
                            for other_robot_id in range(1, self._nb_robots + 1):
                                other_robot_mission = self._model.mobileMission(other_player_id, other_robot_id)
                                if other_robot_mission > 0:
                                    reservation[other_player_id][other_robot_id] = other_robot_mission
                                if reservation[other_player_id][other_robot_id] > 0:
                                    continue
                                if other_player_id == player_id and other_robot_id == robot_id:
                                    continue

                                other_robot_distances = robot_to_missions_distances[other_player_id][other_robot_id]
                                if other_robot_distances[selected_mission_id] < min_distance:
                                    self.log(f"(player_{player_id}, robot_{robot_id}) is not the closest of {selected_mission_id} it is (player_{other_player_id}, robot_{other_robot_id})")
                                    is_closest = False
                                    break
                            
                            if not is_closest:
                                break
                            
                        if is_closest:
                            reservation[player_id][robot_id] = selected_mission_id

        self._reservations = reservation[self._id]
        return reservation[self._id]

    def select_valuable_missions(self, player_id, robot_id):
        """
        Sélectionne un ensemble de missions “intéressantes” pour un robot (missions proches,
        bon ratio récompense/distance, et missions à forte récompense).

        Parameters
        ----------
        player_id : int
            Identifiant du joueur.
        robot_id : int
            Identifiant du robot.

        Returns
        -------
        list of int
            Liste d'IDs de missions jugées intéressantes.
        """
        set_missions = set()
        # On combine 3 stratégies de sélection
        closest_missions = self.get_closest_missions(self._model.mobilePosition(player_id, robot_id))
        high_ratio_missions = self.get_high_ratio_missions(player_id, robot_id)
        max_rewarded_missions = self.get_max_rewarded_mission()

        listes = [closest_missions, high_ratio_missions, max_rewarded_missions]
        for liste in listes:
            for mission in liste:
                set_missions.add(mission)
        
        return list(set_missions)
    
    def calc_mission_reward_distance_ratio(self, player_i, robot_i, mission_i):
        """
        Calcule le ratio (récompense / distance totale) pour une mission donnée.

        Parameters
        ----------
        player_i : int
            Identifiant du joueur.
        robot_i : int
            Identifiant du robot.
        mission_i : int
            Identifiant de la mission.

        Returns
        -------
        float
            Ratio mission.reward / (distance du robot->start + start->final).
        """
        robot_position = self._model.mobilePosition(player_i, robot_i)
        mission = self._model.mission(mission_i)

        dist_robot_mission = self._distances[robot_position][mission.start]
        dist_mission = self._distances[mission.start][mission.final]
        denominator = (dist_robot_mission + dist_mission)
        return mission.reward / denominator if denominator > 0 else 1
    
    def get_closest_missions(self, from_i):
        """
        Retourne les missions les plus proches d'une tuile donnée.

        Parameters
        ----------
        from_i : int
            Tuile de départ.

        Returns
        -------
        list of int
            Liste des identifiants de mission les plus proches.
        """
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
                closest_missions = [mission_id]
            elif current_distance == min_distance:
                closest_missions.append(mission_id)
        
        return closest_missions

    def get_high_ratio_missions(self, player_i, robot_i):
        """
        Retourne les missions qui ont le meilleur ratio récompense/distance pour le robot.

        Parameters
        ----------
        player_i : int
            Identifiant du joueur.
        robot_i : int
            Identifiant du robot.

        Returns
        -------
        list of int
            Liste d'ID de missions filtrées par le plus haut ratio.
        """
        mission_ids = self._model.missionsList()
        if len(mission_ids) == 0:
            return []
        
        selected_missions = []
        max_ratio = self.calc_mission_reward_distance_ratio(player_i, robot_i, 0)

        for mission_id in mission_ids:
            current_ratio = self.calc_mission_reward_distance_ratio(player_i, robot_i, mission_id)
            if current_ratio > max_ratio:
                max_ratio = current_ratio
                selected_missions = [mission_id]
            elif current_ratio == max_ratio:
                selected_missions.append(mission_id)
        
        return selected_missions
    
    def get_max_rewarded_mission(self):
        """
        Retourne la ou les missions offrant la plus grosse récompense brute.

        Returns
        -------
        list of int
            Liste d'ID de missions au reward maximal.
        """
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
                selected_missions = [mission_id]
            elif current_reward == max_reward:
                selected_missions.append(mission_id)
        
        return selected_missions
    
    def get_closest_mission(self, i_from):
        """
        Retourne la mission la plus proche (en distance) d'une tuile donnée.

        Parameters
        ----------
        i_from : int
            Tuile de départ.

        Returns
        -------
        Mission or None
            La mission la plus proche, ou None s'il n'y a pas de mission.
        """
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
    
    def get_mission_score(self, robot_id):
        """
        Calcule un score mission (reward/distance) pour le robot spécifié.

        Parameters
        ----------
        robot_id : int
            Identifiant du robot pour lequel calculer le score.

        Returns
        -------
        float
            Score (reward / distance).
        """
        robot_mission_id = self._model.mobileMission(self._id, robot_id)
        if robot_mission_id <= 0:
            robot_reserved_mission_id = self._reservations[robot_id]
            if robot_reserved_mission_id <= 0:
                return 0

        robot_position = self._model.mobilePosition(self._id, robot_id)
        mission = self._model.mission(robot_reserved_mission_id)
        distance = 0

        if robot_mission_id <= 0:
            distance += self._distances[robot_position][mission.start]
        distance += self._distances[mission.start][mission.final]
        
        return float(mission.reward) / float(distance)
    
    def get_prior_robot(self, robot_id, other_robot_id):
        """
        Compare deux robots pour savoir lequel est prioritaire, par exemple
        celui qui a le meilleur score mission.

        Parameters
        ----------
        robot_id : int
            Identifiant du robot 1.
        other_robot_id : int
            Identifiant du robot 2.

        Returns
        -------
        int
            Identifiant du robot prioritaire.
        """
        robot_score = self.calc_priority(robot_id)
        other_robot_score = self.calc_priority(other_robot_id)
        # Le robot avec le score le plus haut est prioritaire
        robot_id_prior = robot_id if robot_score > other_robot_score else other_robot_id
        return robot_id_prior
    
    def get_missions_distances(self, player_id=1, all_mission_ids=None):
        """
        Retourne un tableau des distances pour un joueur (player_id) et tous ses robots
        vers les missions spécifiées.

        Parameters
        ----------
        player_id : int, optional
            Identifiant du joueur. Par défaut 1.
        all_mission_ids : list of int, optional
            Liste d'IDs de missions. Par défaut, None => toutes les missions existantes.

        Returns
        -------
        list
            Tableau 2D indices = [robot_id][mission_id], valeur = distance (ou -1 si invalide).
        """
        result = [[]]
        if all_mission_ids is None:
            all_mission_ids = self._model.missionsList()
        
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

        return result
    
    def get_all_players_missions_distances(self, all_mission_ids=None):
        """
        Calcule la matrice de distances missions pour tous les joueurs et leurs robots.

        Parameters
        ----------
        all_mission_ids : list of int, optional
            Liste des missions. Par défaut, None => toutes les missions existantes.

        Returns
        -------
        list of list
            Tableau 3D indexé par [player_id][robot_id][mission_id].
        """
        result = [[[]]]
        if all_mission_ids is None:
            all_mission_ids = self._model.missionsList()
        
        for player_id in range(1, self._nb_players + 1):
            player_mission_distances = self.get_missions_distances(player_id=player_id, all_mission_ids=all_mission_ids)
            result.append(player_mission_distances)
    
        return result
    
    # ------------------------------------------------------------------------
    # 4. Calcul des Distances et Analyse des Chemins
    # ------------------------------------------------------------------------
    
    def computeDistances(self, iTile):
        """
        Calcule la distance en nombre de sauts pour atteindre chaque tuile depuis iTile.

        Parameters
        ----------
        iTile : int
            Tuile de départ.

        Returns
        -------
        list of int
            Tableau où l'indice représente la tuile et la valeur représente
            le nombre de pas pour l'atteindre.
        """
        # Initialise toutes les distances à 0
        dists= [iTile] +  [0 for _ in range(self._model.map().size())]
        # Première “couronne”
        ringNodes= self._model.map().neighbours(iTile)
        ringDistance= 1

        # Tant qu'il y a des tuiles à visiter
        while len(ringNodes) > 0 :
            nextNodes= []
            for node in ringNodes :
                dists[node]= ringDistance
            for node in ringNodes :
                neighbours= self._model.map().neighbours(node)
                for candidate in neighbours :
                    if dists[candidate] == 0 :
                         nextNodes.append(candidate)
            ringNodes= nextNodes
            ringDistance+= 1
        
        # Correction (la distance de la tuile à elle-même)
        dists[iTile]= 0
        return dists
    
    def init_ennemies_state(self, update=False):
        """
        Initialise ou met à jour l'état des ennemis. Stocke dans un dictionnaire
        les probabilités de passage pour les robots adverses (et le VIP).

        Parameters
        ----------
        update : bool, optional
            Si True, met à jour les positions du VIP et recalcule les potentiels chemins, par défaut False.

        Returns
        -------
        dict
            Le dictionnaire d'état des ennemis.
        """
        # Initialisation d'un dictionnaire vide pour tous les joueurs et robots
        ennemies_state = {}
        
        # VIP
        if self._is_vip_activated:
            ennemies_state[0] = {1:{
                "passage_count": {
                    "total": [0 for _ in range(self._depth)]
                }
            }}
        
        # Autres joueurs
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
            # Mise à jour de la position du VIP et calcul de chemins probables
            if self._is_vip_activated:
                self.refresh_vip_positions()
                vip_next_moves = self.predict_next_vip_movement(self._depth-1)
                self.log(vip_next_moves)
                for i, (_, dest) in enumerate(vip_next_moves):
                    self.add_count_tile(0, 1, dest, i+1)

            # Mise à jour pour chaque joueur/robot
            for player_id in range(1, self._nb_players + 1):
                if player_id == self._id:
                    continue

                for robot_id in range(1, self._nb_robots + 1):
                    robot_position = self._model.mobilePosition(player_id, robot_id)
                    robot_mission = self._model.mobileMission(player_id, robot_id)
                    # Si robot n'a pas de mission en cours, on regarde s'il y a des missions potentielles
                    if robot_mission == 0:
                        for mission_id in self.select_valuable_missions(player_id, robot_id):
                            mission = self._model.mission(mission_id)
                            # On calcule les chemins pour aller de la position du robot au start de la mission
                            self.calc_all_paths(player_id, robot_id, robot_position, mission.start)
                    else:
                        mission = self._model.mission(robot_mission)
                        # On calcule les chemins pour aller du robot au point final de la mission
                        self.calc_all_paths(player_id, robot_id, robot_position, mission.final)

        return self._dict_state

    def add_count_tile(self, i_player, i_robot, i_tile, step_t):
        """
        Incrémente le comptage de passage (probabilité) sur la tuile i_tile pour un robot donné,
        à un instant step_t.

        Parameters
        ----------
        i_player : int
            Identifiant du joueur (ennemi ou VIP).
        i_robot : int
            Identifiant du robot (ou VIP).
        i_tile : int
            Tuile concernée.
        step_t : int
            Indice de temps (profondeur de prédiction).
        """
        if i_tile not in self._dict_state[i_player][i_robot]["passage_count"]:
            self._dict_state[i_player][i_robot]["passage_count"][i_tile] = [0 for _ in range(self._depth)]
        self._dict_state[i_player][i_robot]["passage_count"][i_tile][step_t] += 1
        self._dict_state[i_player][i_robot]["passage_count"]["total"][step_t] += 1
    
    def get_tile_probability(self, i_player, i_robot, i_tile, step_t):
        """
        Calcule la probabilité qu'un robot (ou le VIP) passe par i_tile à l'instant step_t.

        Parameters
        ----------
        i_player : int
            Identifiant du joueur.
        i_robot : int
            Identifiant du robot.
        i_tile : int
            Identifiant de la tuile.
        step_t : int
            Index de temps (profondeur de prédiction).

        Returns
        -------
        float
            La probabilité (entre 0 et 1).
        """
        try:
            numerator = self._dict_state[i_player][i_robot]["passage_count"][i_tile][step_t]
            denominator = self._dict_state[i_player][i_robot]["passage_count"]["total"][step_t]
        except Exception:
            numerator = 0
            denominator = 1
        
        return numerator/(denominator if denominator != 0 else 1)
    
    def calc_all_paths(self, player_i, robot_i, from_i, to_i, step_t=0):
        """
        Calcule de manière récursive tous les chemins possibles d'un robot
        entre deux tuiles, en incrémentant les compteurs de passage pour chaque tuile traversée.

        Parameters
        ----------
        player_i : int
            Identifiant du joueur.
        robot_i : int
            Identifiant du robot.
        from_i : int
            Tuile de départ.
        to_i : int
            Tuile d'arrivée.
        step_t : int, optional
            Profondeur de l'itération (étape), par défaut 0.
        """
        # Condition d'arrêt
        if from_i == to_i:
            return
        if step_t == self._depth:
            return

        # Incrémente le passage sur la tuile courante
        self.add_count_tile(player_i, robot_i, from_i, step_t)
        
        # Cherche les tuiles voisines
        next_tiles = [x for x in self._model._map.neighbours(from_i)]
        # Sélectionne les tuiles qui minimisent la distance au point d'arrivée
        selected_nexts = [next_tiles[0]]
        min_dist = self._distances[next_tiles[0]][to_i]
        for next_tile in next_tiles:
            current_distance = self._distances[next_tile][to_i] 
            if current_distance < min_dist:
                min_dist = current_distance
                selected_nexts = [next_tile]
            elif current_distance == min_dist:
                selected_nexts.append(next_tile)
        
        for tile_i in selected_nexts:
            self.calc_all_paths(player_i, robot_i, tile_i, to_i, step_t+1)
    
    def get_all_paths(self, from_i, to_i, move=None, current_node=None, step_t=0):
        """
        Construit un arbre de chemins possibles entre deux tuiles à une certaine profondeur.

        Parameters
        ----------
        from_i : int
            Tuile de départ.
        to_i : int
            Tuile d'arrivée.
        move : int, optional
            Mouvement associé à ce nœud, par défaut None.
        current_node : Node, optional
            Nœud actuel dans l'arbre de chemins, par défaut None.
        step_t : int, optional
            Étape actuelle, par défaut 0.

        Returns
        -------
        Node
            L'arbre (ou sous-arbre) représentant tous les chemins possibles depuis `from_i` vers `to_i`.
        """
        if from_i == to_i or step_t == self._depth:
            return Node(from_i, move)
        
        if current_node is None:
            current_node = Node(from_i, move)
        
        # Parcourt les tuiles voisines pour construire l'arbre
        next_tiles = [x for x in self._model._map.neighbours(from_i)]
        next_moves = self._model._map.clockBearing(from_i)
        if len(next_tiles) > 0:
            # Sélectionne celles qui minimisent la distance
            selected_nexts = [(next_tiles[0], next_moves[0])]
            min_dist = self._distances[next_tiles[0]][to_i]
            for next_tile, next_move in zip(next_tiles, next_moves):
                current_distance = self._distances[next_tile][to_i] 
                if current_distance < min_dist:
                    min_dist = current_distance
                    selected_nexts = [(next_tile, next_move)]
                elif current_distance == min_dist:
                    selected_nexts.append((next_tile, next_move))
            
            for tile_i, move_i in selected_nexts:
                next_node = current_node.get_node(tile_i)
                current_node.add_node(self.get_all_paths(tile_i, to_i, move_i, next_node, step_t+1))
        
        return current_node

    def get_next_step_from_node(self, head):
        """
        Récupère la liste de couples (move, value) pour tous les nœuds enfants.

        Parameters
        ----------
        head : Node
            Nœud racine.

        Returns
        -------
        list of tuple
            Liste des déplacements (move, tuile) possibles immédiatement.
        """
        return [(next_node.move, next_node.value) for next_node in head._next_nodes]
    
    def detect_potential_collisions(self, player_i, robot_i, path, step_t=0):
        """
        Détecte les collisions potentielles sur le chemin (path) et supprime 
        les nœuds risqués où la probabilité de rencontre avec un ennemi est trop élevée.

        Parameters
        ----------
        player_i : int
            Identifiant du joueur (ici le bot).
        robot_i : int
            Identifiant du robot.
        path : Node
            Nœud racine du chemin.
        step_t : int, optional
            Étape actuelle de la prédiction, par défaut 0.

        Returns
        -------
        tuple
            (path, remove_current) où path est le chemin mis à jour et remove_current 
            indique si le nœud racine doit être supprimé.
        """
        current_node = path
        robot_position = self._model.mobilePosition(player_i, robot_i)
        to_remove_at = []
        remove_current = False

        # Pour chaque nœud enfant
        for i, next_node in enumerate(current_node._next_nodes):
            to_remove = False
            for other_player_id in range(0, self._nb_players + 1):
                if other_player_id == player_i:
                    continue
                # Pour chaque robot adverse
                for other_robot_id in range(1, self._model.numberOfMobiles(other_player_id)):
                    other_robot_position = self._model.mobilePosition(other_player_id, other_robot_id)
                    distance_to_robot = self._distances[robot_position][other_robot_position]

                    # Probabilité de collision
                    if distance_to_robot % 2 == 0:
                        proba_other_robot = self.get_tile_probability(other_player_id, other_robot_id, current_node.value, step_t)
                    else:
                        proba_other_robot= self.get_tile_probability(other_player_id, other_robot_id, current_node.value, step_t+1)

                    if proba_other_robot >= 0.1:
                        to_remove = True
                        self.log(f"remove {next_node.value}")
                        to_remove_at.append(i)
                
                    if to_remove:
                        break
                if to_remove:
                    break
        
        # Supprime les nœuds jugés risqués
        to_remove_at.reverse()
        for i in to_remove_at:
            del current_node._next_nodes[i] 
        
        # Si le nœud courant n'a plus d'enfants, on évalue si on doit le retirer
        if len(current_node._next_nodes) <= 0:
            all_missions = self._model.missions()
            remove_current = True
            for mission in all_missions:
                if mission.start == current_node.value or mission.final == current_node.value:
                    remove_current = False
                if not remove_current:
                    break
        else:
            # On continue la détection de collision récursivement sur les enfants restants
            other_remove = []
            for i, next_node in enumerate(current_node._next_nodes):
                _, remove_node = self.detect_potential_collisions(player_i, robot_i, next_node, step_t+1)
                if remove_node:
                    other_remove.append(i)
            
            other_remove.reverse()
            for i in other_remove:
                del current_node._next_nodes[i]

        return current_node, remove_current

    # ------------------------------------------------------------------------
    # 5. Logique de Décision et Filtrage
    # ------------------------------------------------------------------------

    def get_actions(self, next_steps=None):
        """
        Calcule l'action (moves + missions) à effectuer pour chaque robot, sous forme de string.

        Parameters
        ----------
        next_steps : dict, optional
            Dictionnaire/structure contenant les déplacements possibles
            pré-calculés. Par défaut, None => calcul via `get_next_steps_better`.

        Returns
        -------
        str
            Commande de type "mission x ... move y ..." adaptée au moteur de jeu.
        """
        if next_steps is None:
            next_steps = self.get_next_steps_better()

        move_actions = []
        mission_actions = []
        actions = []
        # Pour chaque robot, on choisit une action parmi les possibilités
        for robot_id in range(1, self._nb_robots + 1):
            current_action = ""
            next_moves = next_steps[robot_id]
            (action_keyword, next_move, _) = random.choice(next_moves)
            current_action = f"{action_keyword} {robot_id} {next_move}"

            if "move" in current_action:
                current_action = current_action.removeprefix("move ")
                move_actions.append(current_action)
            if "mission" in current_action:
                current_action = current_action.removeprefix("mission ")
                mission_actions.append(current_action)
        
        # Concaténation en un seul string
        if len(mission_actions) > 0:
            actions.append("mission " + " ".join(mission_actions))
        if len(move_actions) > 0:
            actions.append("move " + " ".join(move_actions))
        
        actions = " ".join(actions)
        return actions
        
    def get_next_steps_better(self, reservations=None):
        """
        Calcule les déplacements et actions optimaux pour chaque robot, en appliquant
        divers filtres et politiques (VIP, collisions, etc.).

        Parameters
        ----------
        reservations : list, optional
            Liste de missions réservées pour les robots de ce bot. Par défaut, None,
            ce qui déclenche un nouvel assignement de missions.

        Returns
        -------
        list of list
            Structure contenant, pour chaque robot, une liste de (action_type, move, next_tile).
        """
        if reservations is None:
            reservations = self.assign_missions()
        
        # Calcule les mouvements optimaux initiaux
        next_moves=[[] for _ in range(self._nb_robots + 1)]
        next_moves=self.get_optimal_moves(next_moves, reservations=reservations)

        # Applique une série de filtres/contraintes
        next_moves=self.apply_priority(next_moves)
        next_moves=self.apply_vip_policies(next_moves)
        next_moves=self.filter_by_willingess(next_moves)
        next_moves=self.filter_by_any_presence(next_moves)
        next_moves=self.apply_default_move(next_moves)

        return next_moves
    
    def get_optimal_moves(self, next_moves, reservations):
        """
        Détermine les déplacements optimaux pour chaque robot en fonction
        des missions qui leur sont assignées ou déjà actives.

        Parameters
        ----------
        next_moves : list
            Liste vide ou partiellement remplie de déplacements par robot.
        reservations : list
            Liste des missions affectées à chaque robot de ce bot.

        Returns
        -------
        list
            La même liste `next_moves`, enrichie des déplacements planifiés.
        """
        self.log("get_optimal_moves")
        for robot_id in range(1, self._nb_robots + 1):
            robot_position = self._model.mobilePosition(self._id, robot_id)
            robot_mission_id = self._model.mobileMission(self._id, robot_id)

            # Pas de mission réservée => mouvement par défaut
            if reservations[robot_id] < 0:
                self.log(f"\trobot-{robot_id} No reserved mission")
                current_next_moves = [('move', 0, robot_position)]
            
            # Mission réservée, mais non encore activée
            elif robot_mission_id == 0 and reservations[robot_id] > 0:
                robot_mission = self._model.mission(reservations[robot_id])
                self.log(f"\trobot-{robot_id} reserved mission {reservations[robot_id]} go to {robot_mission.start}")

                # Si déjà sur la tuile de départ de la mission
                if robot_position == robot_mission.start:
                    current_next_moves = [('mission', reservations[robot_id], robot_position)]
                else:
                    head = self.get_all_paths(robot_position, robot_mission.start)
                    head, _ = self.detect_potential_collisions(self._id, robot_id, head, 1)
                    current_next_moves = self.get_next_step_from_node(head)
                    current_next_moves = [('move', *move) for move in current_next_moves]
            
            # Mission déjà activée
            else:
                self.log(f"\trobot-{robot_id} activated mission {robot_mission_id}")
                robot_mission = self._model.mission(robot_mission_id)
                # Si déjà sur la tuile final de la mission
                if robot_position == robot_mission.final:
                    current_next_moves = [('mission', robot_mission_id, robot_position)]
                else:
                    head = self.get_all_paths(robot_position, robot_mission.final)
                    head, _ = self.detect_potential_collisions(self._id, robot_id, head, 1)
                    current_next_moves = self.get_next_step_from_node(head)
                    current_next_moves = [('move', *move) for move in current_next_moves]
            
            next_moves[robot_id] = current_next_moves
        
        return next_moves

    def filter_by_any_presence(self, next_moves):
        """
        Filtre les déplacements qui mènent à une tuile déjà occupée par n'importe quel pion.

        Parameters
        ----------
        next_moves : list
            Liste des déplacements potentiels par robot.

        Returns
        -------
        list
            Liste de déplacements modifiée, nettoyée des collisions “incontrôlables”.
        """
        self.log("filter_by_any_presence")
        for robot_id in range(1, self._nb_robots + 1):
            to_remove = []
            # Vérifie chaque destination potentielle
            for i, (action_type, move, next_tile) in enumerate(next_moves[robot_id]):
                if action_type != "move":
                    continue
                if move == 0:  # Rester sur place
                    continue

                pieces = self._model._map.tile(next_tile)._pieces
                # Si la tuile est déjà occupée
                if len(pieces) > 0:
                    self.log(f"\t- robot-{robot_id}: remove ({action_type}, {move}, {next_tile}")
                    if i not in to_remove:
                        to_remove.append(i)
            
            to_remove.sort(reverse=True)
            for i in to_remove:
                next_moves[robot_id].pop(i)
        
        return next_moves

    def filter_by_willingess(self, next_moves):
        """
        Filtre les déplacements qui entraîneraient des collisions directes entre
        deux robots de ce bot. Si un conflit est détecté, le bot privilégie le robot
        qui a le moins de possibilités, ou un autre critère d'arbitrage.

        Parameters
        ----------
        next_moves : list
            Liste des déplacements potentiels par robot.

        Returns
        -------
        list
            Liste de déplacements filtrée pour éviter que deux robots aillent sur la même tuile.
        """
        self.log("filter_by_willingness")

        for robot_id in range(1, self._nb_robots + 1):
            for other_robot_id in range(1, self._nb_robots + 1):
                if robot_id == other_robot_id:
                    continue
                
                to_remove = []
                o_to_remove = []
                # Compare chaque move de robot_id avec chaque move de other_robot_id
                for i, (_, _, tile) in enumerate(next_moves[robot_id]):
                    for j, (_, _, o_tile) in enumerate(next_moves[other_robot_id]):
                        if tile == o_tile:
                            self.log(f"\t robot-{robot_id} conflict with robot-{other_robot_id} on {o_tile}")
                            # On préfère laisser le robot avec le plus de choix possibles ou selon un autre critère
                            if len(next_moves[robot_id]) > len(next_moves[other_robot_id]):
                                if i not in to_remove:
                                    to_remove.append(i)
                            else:
                                if j not in o_to_remove:
                                    o_to_remove.append(j)
                
                # Suppression en ordre décroissant pour pas fausser les indices
                to_remove.sort(reverse=True)
                o_to_remove.sort(reverse=True)

                for i in to_remove:
                    del next_moves[robot_id][i]
                for j in o_to_remove:
                    del next_moves[other_robot_id][j]
        
        return next_moves

    def apply_vip_policies(self, next_moves):
        """
        Applique des politiques spécifiques autour de la position du VIP (éviter de trop s'approcher, etc.).

        Parameters
        ----------
        next_moves : list
            Liste de déplacements par robot.

        Returns
        -------
        list
            Liste de déplacements modifiée selon les règles VIP.
        """
        self.log("apply_vip_policies")
        if self._is_vip_activated:
            for robot_id in range(1, self._nb_robots + 1):
                robot_position = self._model.mobilePosition(self._id, robot_id)
                vip_position = self._model.mobilePosition(0, 1)
                current_dist = self._distances[robot_position][vip_position]

                # Si le robot est très proche du VIP
                if current_dist <= 2:
                    neighbours = [x for x in self._model._map.neighbours(robot_position)]
                    directions = self._model._map.clockBearing(robot_position)

                    # On enlève la direction 0 (rester sur place) de la liste
                    if 0 in directions:
                        i_zero = directions.index(0)
                        del neighbours[i_zero]
                        del directions[i_zero]
                    
                    to_remove_indexes = []
                    # On cherche à éviter de se rapprocher encore plus du VIP
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

                    # On supprime ces mouvements des choix possibles
                    if len(to_remove_dir) > 0:
                        to_remove_indexes = []
                        for  i, (_, next_dir, _) in enumerate(next_moves[robot_id]):
                            if next_dir in to_remove_dir:
                                to_remove_indexes.append(i)
                        
                        to_remove_indexes.sort(reverse=True)
                        for i in to_remove_indexes:
                            del next_moves[robot_id][i]
                
                    # On ajoute éventuellement des directions alternatives (pour s'éloigner du VIP)
                    for direction, neighbour in zip(directions, neighbours):
                        exist = False
                        for _, next_dir, _ in next_moves[robot_id]:
                            if direction == next_dir:
                                exist = True
                                break
                        if not exist:
                            pieces = self._model._map.tile(neighbour)._pieces
                            if len(pieces) == 0:
                                next_moves[robot_id].append(("move", direction, neighbour))
        return next_moves
    
    def apply_priority(self, next_moves):
        """
        Donne une priorité à certains robots en fonction de leur “score mission” et
        restreint les mouvements des autres robots quand ils sont trop proches.

        Parameters
        ----------
        next_moves : list
            Liste de déplacements par robot.

        Returns
        -------
        list
            Liste de déplacements modifiée pour respecter la priorité.
        """
        self.log("apply_priority")
        for robot_id in range(1, self._nb_robots + 1):
            robot_position = self._model.mobilePosition(self._id, robot_id)
            for other_robot_id in range(1, self._nb_robots + 1):
                if robot_id == other_robot_id:
                    continue

                other_robot_position = self._model.mobilePosition(self._id, other_robot_id)
                distance = self._distances[robot_position][other_robot_position]
                if distance > 2:
                    continue
                # On détermine qui est prioritaire
                prior_robot = self.get_prior_robot(robot_id, other_robot_id)
                
                # Cas distance == 1
                if distance == 1 and prior_robot == other_robot_id:
                    neighbours = [x for x in self._model._map.neighbours(robot_position)]
                    directions = self._model._map.clockBearing(robot_position)
                    
                    if 0 in directions:
                        i = directions.index(0)
                        del neighbours[i]
                        del directions[i]
                    
                    tile_to_remove = []
                    to_remove = []
                    # Supprime les tuiles occupées
                    for i, tile in enumerate(neighbours):
                        pieces = self._model._map.tile(tile)._pieces
                        if len(pieces) > 0:
                            tile_to_remove.append(tile)
                            to_remove.append(i)
                    
                    to_remove.sort(reverse=True)
                    for i in to_remove:
                        del neighbours[i]
                        del directions[i]
                    
                    # On supprime également du tableau next_moves
                    to_remove = []
                    for i, (_, _, next_tile) in enumerate(next_moves[robot_id]):
                        if next_tile in tile_to_remove:
                            to_remove.append(i)
                    
                    to_remove.sort(reverse=True)
                    for i in to_remove:
                        del next_moves[robot_id][i]

                    # On rajoute les mouvements restants
                    for tile, dir in zip(neighbours, directions):
                        next_moves[robot_id].append(("move", dir, tile))
                
                # Cas distance == 2
                elif distance == 2 and prior_robot == other_robot_id:
                    conflicting_move = []
                    old_robot_next_moves = [x for x in next_moves[robot_id]]
                    other_robot_next_moves = next_moves[other_robot_id]

                    # Supprime les moves en conflit
                    for i, (_, _, current_next_tile) in enumerate(old_robot_next_moves):
                        for _, _, other_robot_next_tile in other_robot_next_moves:
                            if current_next_tile == other_robot_next_tile:
                                conflicting_move.append(i)
                    
                    conflicting_move.sort(reverse=True)
                    for i in conflicting_move:
                        del next_moves[robot_id][i]
                    
                    # On propose des alternatives s'il reste encore des coups
                    if len(next_moves[robot_id]) > 0:
                        robot_next_moves = []
                        neighbours = [x for x in self._model._map.neighbours(robot_position)]
                        directions = self._model._map.clockBearing(robot_position)
                        
                        current_dist = self._distances[robot_position][other_robot_position]
                        if 0 in directions:
                            i = directions.index(0)
                            del neighbours[i]
                            del directions[i]
                        
                        to_remove = []
                        for i, tile in enumerate(neighbours):
                            next_dist = self._distances[tile][other_robot_position]
                            pieces = self._model._map.tile(tile)._pieces
                            # On évite de se rapprocher ou les tuiles occupées
                            if next_dist <= current_dist or len(pieces) > 0:
                                to_remove.append(i)
                        
                        to_remove.sort(reverse=True)
                        for i in to_remove:
                            del neighbours[i]
                            del directions[i]

                        for tile, dir in zip(neighbours, directions):
                            robot_next_moves.append(("move", dir, tile))
                        
                        next_moves[robot_id] = robot_next_moves
        
        return next_moves

    def apply_default_move(self, next_moves):
        """
        Pour chaque robot n'ayant aucun mouvement possible, ajoute un move “0” (rester sur place).

        Parameters
        ----------
        next_moves : list
            Liste de déplacements potentiels par robot.

        Returns
        -------
        list
            Liste de déplacements complétée par des moves 0 si nécessaire.
        """
        self.log("apply_default_move for:")
        for robot_id in range(1, self._nb_robots+1):
            robot_position = self._model.mobilePosition(self._id, robot_id)
            if len(next_moves[robot_id]) <= 0:
                self.log(f"\t- robot-{robot_id}")
                next_moves[robot_id].append(("move", 0, robot_position))
        return next_moves

    def calc_priority(self, robot_id):
        """
        Calcule la priorité d'un robot (pour appliquer certaines règles de filtrage).

        Parameters
        ----------
        robot_id : int
            Identifiant du robot.

        Returns
        -------
        float
            Un score calculé, ici 10 * get_mission_score(robot_id) + robot_id.
        """
        try:
            robot_id_score = self.get_mission_score(robot_id)
            return 10 * robot_id_score + robot_id
        except Exception:
            return 0

    def log(self, message):
        """
        Affiche un message dans la console si le mode debug est activé.

        Parameters
        ----------
        message : str
            Le message à afficher.
        """
        if self._debug:
            print(message)