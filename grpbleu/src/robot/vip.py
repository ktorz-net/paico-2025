class VIP:
    def __init__(self):
        self.id = 0 # is the id for vip
        self.vip_position = None
        self._last_vip_pos = None
        self._vip_stuck_counter = 0
        self._is_vip_exist = False

    def getVipPosition(self):
        return self.vip_position

    def getVipExistence(self):
        return self._is_vip_exist

    def checkVip(self, model):
        numberOfVips = model.numberOfMobiles(self.id)
        if numberOfVips > 0:
            self._is_vip_exist = True
        else: # Pas d'intéret pour cette partie étant donné que j'initialise à false déjà
            self._is_vip_exist = False

    def setVipPosition(self, position):
        self._last_vip_pos = self.vip_position
        self.vip_position = position

    def updateVipPosition(self, model):
        if self._is_vip_exist:
            position = model.mobilePosition(self.id, 0)
            self.setVipPosition(position)

    def predict_vip_movement(self, game):
        """Prédit les mouvements possibles du VIP avec leurs probabilités

        Le système utilise plusieurs facteurs pour pondérer les probabilités:
        - Distance aux missions disponibles
        - Temps passé sur la même case (détection de blocage)
        - Direction générale du mouvement
        """
        if self.vip_position is None or not (0 <= self.vip_position < len(game.getDistances())):
            return {}

        predictions = {}
        available_missions = game.getModel().freeMissions()

        # Détection si le VIP est bloqué au même endroit
        if hasattr(self, '_last_vip_pos') and self._last_vip_pos == self.vip_position:
            self._vip_stuck_counter = getattr(self, '_vip_stuck_counter', 0) + 1
        else:
            self._vip_stuck_counter = 0
        self._last_vip_pos = self.vip_position

        if not (0 <= self.vip_position < len(game.getDistances())):
            return {}

        # Analyse des cases voisines
        neighbors = game.getModel().map().neighbours(self.vip_position)
        for tile in neighbors:
            if not game.getModel().map().tile(tile) or tile >= len(game.getDistances()):
                continue

            predictions[tile] = 1.0
            # Augmentation de la probabilité si la case rapproche le VIP d'une mission
            for mission_id in available_missions:
                mission = game.getModel().mission(mission_id)
                if (mission.start < len(game.getDistance(tile)) and
                    mission.start < len(game.getDistance(self.vip_position))):
                    # Bonus pour rapprochement du début de mission
                    if game.getDistance(tile, mission.start) < game.getDistance(self.vip_position, mission.start):
                        predictions[tile] *= 1.5
                    # Bonus pour rapprochement de la fin de mission
                    if game.getDistance(tile, mission.final) < game.getDistance(self.vip_position, mission.final):
                        predictions[tile] *= 1.5

            # Bonus si le VIP est bloqué
            if self._vip_stuck_counter > 1:
                predictions[tile] *= 1.2

        # Normalisation des probabilités
        total = sum(predictions.values())
        if total > 0:
            for tile in predictions:
                predictions[tile] /= total
        return predictions

    def avoid_vip(self, game, current_pos, target_pos):
        """Calcule un chemin sûr évitant le VIP

        Cette fonction utilise plusieurs stratégies:
        1. Évitement d'urgence si adjacent au VIP
        2. Recherche de chemins alternatifs si risque de collision
        3. Évaluation de la sécurité des chemins basée sur les prédictions du VIP

        Returns:
            tuple: (moves, path, safety_score)
            - moves: liste des directions à prendre
            - path: liste des cases à traverser
            - safety_score: score de sécurité entre 0 et 1
        """
        # Obtient les prédictions de mouvement du VIP
        vip_predictions = self.predict_vip_movement(game)
        base_move, base_path = game.path(current_pos, target_pos)

        # Gestion d'urgence si adjacent au VIP
        if current_pos != self.vip_position and self.vip_position in game.getModel().map().neighbours(current_pos):
            neighbors = game.getModel().map().neighbours(current_pos)
            clockdirs = game.getModel().map().clockBearing(current_pos)
            best_neighbor = None
            best_score = float('-inf')
            best_dir = None

            # Recherche la meilleure case pour s'éloigner du VIP
            for neighbor, direction in zip(neighbors, clockdirs):
                if not game.getModel().map().tile(neighbor) or neighbor == self.vip_position:
                    continue

                vip_distance = game.getDistance(neighbor, self.vip_position)
                target_distance = game.getDistance(neighbor, target_pos)
                score = vip_distance - (target_distance * 0.5)  # Privilégie l'éloignement du VIP

                if score > best_score:
                    best_score = score
                    best_neighbor = neighbor
                    best_dir = direction

            if best_neighbor is not None:
                # self.logger.info(f"Mouvement d'évitement d'urgence choisi: {best_dir}")
                return [best_dir], [best_neighbor], 0.8

        # Cas critique: même case que le VIP
        if current_pos == self.vip_position:
            # self.logger.critical(f"Robot sur la même case que le VIP! Position: {current_pos}")
            neighbors = game.getModel().map().neighbours(current_pos)
            clockdirs = game.getModel().map().clockBearing(current_pos)
            best_neighbor = None
            best_score = float('-inf')
            best_dir = None

            for neighbor, direction in zip(neighbors, clockdirs):
                if not game.getModel().map().tile(neighbor):
                    continue

                distance_score = -game.getDistance(neighbor, target_pos)
                vip_risk = vip_predictions.get(neighbor, 0)
                total_score = distance_score * (1 - vip_risk)

                if total_score > best_score:
                    best_score = total_score
                    best_neighbor = neighbor
                    best_dir = direction

            if best_neighbor is not None:
                return [best_dir], [best_neighbor], 0.8

        # Utilise le chemin de base s'il est sûr
        if not any(tile in base_path for tile in vip_predictions):
            return base_move, base_path, 1.0

        # Recherche d'alternatives plus sûres
        safety_scores = {}
        alternative_paths = []

        for intermediate in range(game.getModel().map().size()):
            if not game.getModel().map().tile(intermediate):
                continue

            # Skip les détours trop longs
            if (game.getDistance(current_pos, intermediate) +
                game.getDistance(intermediate, target_pos) >=
                game.getDistance(current_pos, target_pos) * 1.5):
                continue

            move1, path1 = game.path(current_pos, intermediate)
            move2, path2 = game.path(intermediate, target_pos)
            full_path = path1 + path2[1:]
            risk = sum(vip_predictions.get(tile, 0) for tile in full_path)
            safety_score = 1.0 / (1.0 + risk)
            safety_scores[tuple(full_path)] = safety_score
            alternative_paths.append((move1, full_path))

        if alternative_paths:
            best_path = max(safety_scores.items(), key=lambda x: x[1])
            best_moves = next(moves for moves, path in alternative_paths
                            if tuple(path) == best_path[0])
            return best_moves, list(best_path[0]), best_path[1]

        return base_move, base_path, 0.3

    def handle_vip_collision(self, game, robot_pos):
        """Gère la collision avec le VIP"""
        neighbors = game.getModel().map().neighbours(robot_pos)
        clockdirs = game.getModel().map().clockBearing(robot_pos)

        for neighbor, direction in zip(neighbors, clockdirs):
            if direction != 0 and game.getModel().map().tile(neighbor):
                return direction
        return None

    def is_moving_away_from_vip(self, game, target_pos, robot_pos):
        """Vérifie si un mouvement éloigne du VIP

        Args:
            target_pos (int): Position cible
            robot_pos (int): Position actuelle du robot
            vip_pos (int): Position du VIP

        Returns:
            bool: True si le mouvement éloigne du VIP
        """
        if self.vip_position is None:
            return True
        return (0 <= target_pos < len(game.getDistances()) and
                0 <= self.vip_position < len(game.getDistance(target_pos)) and
                0 <= robot_pos < len(game.getDistances()) and
                0 <= self.vip_position < len(game.getDistances(robot_pos)) and
                game.getDistance(target_pos, self.vip_position) > game.getDistance(robot_pos, self.vip_position))