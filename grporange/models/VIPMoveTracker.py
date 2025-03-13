class VIPMoveTracker:
    def __init__(self):
        self.moves = []
        self.zones = []
        self.update_count = 0
    def record_move(self, direction):
        self.moves.append(direction)
    def record_zone(self, zone):
        self.zones.append(zone)
    def reset(self):
        self.moves = []
    def get_vip_last_move(self, prev_vip, curr_vip, map_instance, other_robots_prev, other_robots_curr, collision_distance=2.0):
        if prev_vip != curr_vip:
            clock_array = map_instance.completeClock(prev_vip)
            for direction in range(13):
                if clock_array[direction] == curr_vip:
                    return direction
            return None
        else:
            vip_center = map_instance.tile(curr_vip).center()
            vip_x = vip_center.x()
            vip_y = vip_center.y()
            for robot_id, prev_pos in other_robots_prev.items():
                curr_pos = other_robots_curr.get(robot_id, None)
                if prev_pos == curr_pos:
                    other_center = map_instance.tile(curr_pos).center()
                    other_x = other_center.x()
                    other_y = other_center.y()
                    dx = vip_x - other_x
                    dy = vip_y - other_y
                    distance = (dx ** 2 + dy ** 2) ** 0.5
                    if distance <= collision_distance:
                        return -1
            return 0
    def update(self, prev_vip, curr_vip, map_instance, other_robots_prev, other_robots_curr, collision_distance=2.0):
        self.update_count += 1
        move = self.get_vip_last_move(prev_vip, curr_vip, map_instance, other_robots_prev, other_robots_curr, collision_distance)
        if self.update_count < 3:
            if move != 0:
                self.record_move(move)
        else:
            if move == 0:
                self.reset()
                if curr_vip not in self.zones:
                    self.record_zone(curr_vip)
            else:
                self.record_move(move)
        return move
    def get_valid_cardinal_directions(self, current_tile, map_instance):
        cardinal_set = {3, 6, 9, 12}
        valid = set()
        clock_array = map_instance.completeClock(current_tile)
        for d in cardinal_set:
            if clock_array[d] != current_tile:
                valid.add(d)
        return valid
    def get_possible_moves(self, current_tile, map_instance):
        valid = self.get_valid_cardinal_directions(current_tile, map_instance)
        return [0] + sorted(list(valid))
    def get_markov_prediction(self):
        if not self.moves:
            return {}
        last_move = self.moves[-1]
        counts = {}
        for move in self.moves:
            counts[move] = counts.get(move, 0) + 1
        total = sum(counts.values())
        probs = {move: count / total for move, count in counts.items()}
        return probs
    def predict_next_move(self, current_tile, map_instance):
        valid = self.get_valid_cardinal_directions(current_tile, map_instance)
        cardinals = {3, 6, 9, 12}
        if not self.moves:
            candidate_set = cardinals
        else:
            last_cardinal_index = None
            for i in range(len(self.moves) - 1, -1, -1):
                if self.moves[i] in cardinals:
                    last_cardinal_index = i
                    break
            if last_cardinal_index is None:
                candidate_set = cardinals
            else:
                last_cardinal = self.moves[last_cardinal_index]
                reversal = {3: 9, 9: 3, 6: 12, 12: 6}[last_cardinal]
                diagonal_count = 0
                for m in self.moves[last_cardinal_index + 1:]:
                    if m not in cardinals and m not in {0, -1}:
                        diagonal_count += 1
                if diagonal_count < 2:
                    candidate_set = cardinals - {reversal}
                else:
                    candidate_set = cardinals
        markov_probs = self.get_markov_prediction()
        weighted = {}
        for move in candidate_set:
            weighted[move] = markov_probs.get(move, 0) + 1e-6
        if 0 in self.get_possible_moves(current_tile, map_instance):
            weighted[0] = 1e-6
        total_weight = sum(weighted.values())
        for move in weighted:
            weighted[move] /= total_weight
        candidate_set = set(weighted.keys())
        possible = candidate_set.intersection(valid)
        sorted_possible = sorted(list(possible), key=lambda m: weighted[m], reverse=True)
        return [0] + sorted_possible

    def predict_next_move_distribution(self, current_tile, map_instance):
        valid = self.get_valid_cardinal_directions(current_tile, map_instance)
        cardinals = {3, 6, 9, 12}
        if not self.moves:
            candidate_set = cardinals
        else:
            last_cardinal_index = None
            for i in range(len(self.moves) - 1, -1, -1):
                if self.moves[i] in cardinals:
                    last_cardinal_index = i
                    break
            if last_cardinal_index is None:
                candidate_set = cardinals
            else:
                last_cardinal = self.moves[last_cardinal_index]
                reversal = {3: 9, 9: 3, 6: 12, 12: 6}[last_cardinal]
                diagonal_count = 0
                for m in self.moves[last_cardinal_index + 1:]:
                    if m not in cardinals and m not in {0, -1}:
                        diagonal_count += 1
                if diagonal_count < 2:
                    candidate_set = cardinals - {reversal}
                else:
                    candidate_set = cardinals
        all_candidates = candidate_set.union({0})
        import math
        alpha = 0.5
        epsilon = 1e-6
        spatial_weights = {}
        for move in all_candidates:
            if move == 0:
                candidate_tile = current_tile
            else:
                candidate_tile = map_instance.clockposition(current_tile, move)
            candidate_center = map_instance.tile(candidate_tile).center()
            zone_weight = 0
            for zone in self.zones:
                zone_center = map_instance.tile(zone).center()
                dx = candidate_center.x() - zone_center.x()
                dy = candidate_center.y() - zone_center.y()
                distance = math.sqrt(dx * dx + dy * dy)
                zone_weight += math.exp(-alpha * distance)
            spatial_weights[move] = zone_weight + epsilon
        markov_probs = self.get_markov_prediction()
        combined_weights = {}
        for move in all_candidates:
            markov_weight = markov_probs.get(move, epsilon)
            combined_weights[move] = markov_weight * spatial_weights[move]
        total_weight = sum(combined_weights.values())
        distribution = {move: combined_weights[move] / total_weight for move in combined_weights}
        final_distribution = {move: distribution[move] for move in distribution if move == 0 or move in valid}
        filtered_total = sum(final_distribution.values())
        if filtered_total > 0:
            final_distribution = {move: final_distribution[move] / filtered_total for move in final_distribution}
        return final_distribution