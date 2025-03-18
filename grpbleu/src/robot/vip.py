class VIP:
    def __init__(self):
        self.id = 0 # 0 is the id for vip
        self.vip_position = None
        self._last_vip_pos = None
        self._vip_stuck_counter = 0
        self._is_vip_exist = False

    # Vip Existence ----------------------------------------------------------------------------------------------------
    def getVipExistence(self):
        return self._is_vip_exist

    def checkVip(self, model):
        """
            Check the configuration model to determine if a Vip exists or not.
            This method is typically used at the beginning of the game.
        """
        numberOfVips = model.numberOfMobiles(self.id)
        if numberOfVips > 0:
            self._is_vip_exist = True
        else:
            self._is_vip_exist = False

    # Vip Position -----------------------------------------------------------------------------------------------------
    def getVipPosition(self):
        return self.vip_position

    def setVipPosition(self, position):
        """ Set position of the vip + keep the position - 1 """
        self._last_vip_pos = self.vip_position
        self.vip_position = position

    def updateVipPosition(self, model):
        """ Update the position of the vip depending on the current state of the game """
        if self._is_vip_exist:
            position = model.mobilePosition(self.id, 0)
            self.setVipPosition(position)

    # Prediction -------------------------------------------------------------------------------------------------------
    def predict_vip_movement(self, game):
        """Prédit les mouvements possibles du VIP avec leurs probabilités

        The system uses several factors to weigh the probabilities:
        - Distance to available missions
        - Time spent on the same tile (stuck detection)
        - General direction of movement
        """
        if self.vip_position is None or not (0 <= self.vip_position < len(game.getDistances())):
            return {}

        predictions = {}
        available_missions = game.getModel().freeMissions()

        # Detect if the VIP is stuck in the same place
        if hasattr(self, '_last_vip_pos') and self._last_vip_pos == self.vip_position:
            self._vip_stuck_counter = getattr(self, '_vip_stuck_counter', 0) + 1
        else:
            self._vip_stuck_counter = 0
        self._last_vip_pos = self.vip_position

        if not (0 <= self.vip_position < len(game.getDistances())):
            return {}

        # Analyze neighboring tiles
        neighbors = game.getModel().map().neighbours(self.vip_position)
        for tile in neighbors:
            if not game.getModel().map().tile(tile) or tile >= len(game.getDistances()):
                continue

            predictions[tile] = 1.0
            # Increase probability if the tile brings the VIP closer to a mission
            for mission_id in available_missions:
                mission = game.getModel().mission(mission_id)
                if (mission.start < len(game.getDistance(tile)) and
                    mission.start < len(game.getDistance(self.vip_position))):
                    # Bonus for getting closer to the mission start
                    if game.getDistance(tile, mission.start) < game.getDistance(self.vip_position, mission.start):
                        predictions[tile] *= 1.5
                    # Bonus for getting closer to the mission end
                    if game.getDistance(tile, mission.final) < game.getDistance(self.vip_position, mission.final):
                        predictions[tile] *= 1.5

            # Bonus if the VIP is stuck
            if self._vip_stuck_counter > 1:
                predictions[tile] *= 1.2

        # Normalize probabilities
        total = sum(predictions.values())
        if total > 0:
            for tile in predictions:
                predictions[tile] /= total
        return predictions

    def avoid_vip(self, game, current_pos, target_pos):
        """Calculates a safe path avoiding the VIP

        This function uses several strategies:
        1. Emergency avoidance if adjacent to the VIP
        2. Search for alternative paths if there is a risk of collision
        3. Evaluate the safety of paths based on VIP predictions

        Returns:
            tuple: (moves, path, safety_score)
            - moves: list of directions to take
            - path: list of tiles to traverse
            - safety_score: safety score between 0 and 1
        """
        # Get VIP movement predictions
        vip_predictions = self.predict_vip_movement(game)
        base_move, base_path = game.path(current_pos, target_pos)

        # Emergency handling if adjacent to the VIP
        if current_pos != self.vip_position and self.vip_position in game.getModel().map().neighbours(current_pos):
            neighbors = game.getModel().map().neighbours(current_pos)
            clockdirs = game.getModel().map().clockBearing(current_pos)
            best_neighbor = None
            best_score = float('-inf')
            best_dir = None

            # Find the best tile to move away from the VIP
            for neighbor, direction in zip(neighbors, clockdirs):
                if not game.getModel().map().tile(neighbor) or neighbor == self.vip_position:
                    continue
                vip_distance = game.getDistance(neighbor, self.vip_position)
                target_distance = game.getDistance(neighbor, target_pos)
                score = vip_distance - (target_distance * 0.5)  # Prioritize moving away from the VIP
                if score > best_score:
                    best_score = score
                    best_neighbor = neighbor
                    best_dir = direction
            if best_neighbor is not None:
                return [best_dir], [best_neighbor], 0.8

        # Critical case: same tile as the VIP
        if current_pos == self.vip_position:
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

        # Use the base path if it is safe
        if not any(tile in base_path for tile in vip_predictions):
            return base_move, base_path, 1.0

        # Search for safer alternatives
        safety_scores = {}
        alternative_paths = []

        for intermediate in range(game.getModel().map().size()):
            if not game.getModel().map().tile(intermediate):
                continue

            # Skip detours that are too long
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
        """Handles collision with the VIP"""
        neighbors = game.getModel().map().neighbours(robot_pos)
        clockdirs = game.getModel().map().clockBearing(robot_pos)

        for neighbor, direction in zip(neighbors, clockdirs):
            if direction != 0 and game.getModel().map().tile(neighbor):
                return direction
        return None

    def is_moving_away_from_vip(self, game, target_pos, robot_pos):
        """Checks if a move is away from the VIP

        Args:
            target_pos (int): Target position
            robot_pos (int): Current robot position
            vip_pos (int): VIP position

        Returns:
            bool: True if the move is away from the VIP
        """
        if self.vip_position is None:
            return True
        return (0 <= target_pos < len(game.getDistances()) and
                0 <= self.vip_position < len(game.getDistance(target_pos)) and
                0 <= robot_pos < len(game.getDistances()) and
                0 <= self.vip_position < len(game.getDistances(robot_pos)) and
                game.getDistance(target_pos, self.vip_position) > game.getDistance(robot_pos, self.vip_position))
