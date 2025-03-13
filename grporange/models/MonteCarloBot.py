import copy
import math
import random
import time
from collections import deque
from hacka import AbsPlayer
from hacka.games.moveit import GameEngine

def weighted_random_choice(moves, weights):
    total = sum(weights)
    r = random.uniform(0, total)
    cumul = 0
    for move, w in zip(moves, weights):
        cumul += w
        if r <= cumul:
            return move
    return moves[-1]

class PathMCBot(AbsPlayer):
    """
    Bot coordonné avec prédiction probabiliste et gestion des blocages prolongés :
      1) Prédit pour chaque adversaire (p != self._id, p != 0) leur prochain mouvement
         (BFS vers la mission la plus proche pour eux, prob=1).
      2) Pour le VIP, utilise vip_bot.predict_next_move_distribution pour obtenir
         la distribution de mouvements du VIP et en déduire la case la plus probable
         (si la probabilité est supérieure ou égale au seuil vip_prob_threshold).
      3) Ces cases prédites sont ajoutées dans les tuiles bloquées pour le BFS.
      4) Utilise BFS + alternatives + mini-simulation + coordination interne pour décider.
      5) Assemble les commandes en un bloc "mission ..." et un bloc "move ..." sans doublon de préfixe.
      6) Si un robot reste bloqué ("r 0") pendant 2 tours consécutifs, il choisit alors un mouvement aléatoire
         parmi les mouvements disponibles (non "0") pour débloquer la situation, sauf s'il est en coin.
      7) En cas de non disponibilité de missions (toutes prises par l'adversaire), le bot choisit de s'éloigner
         des adversaires en optant pour un mouvement "flee".
      8) Affiche un log de chaque collision détectée si debug est activé.
      9) Fonctionne aussi lorsqu'il est seul avec un VIP (enemy_bot est None).
    """
    def __init__(self, vip_bot, enemy_bot=None, debug=False, alpha=1.0, vip_prob_threshold=0.5):
        super().__init__()
        self.vip_bot = vip_bot
        self.enemy_bot = enemy_bot
        self._id = None
        self._model = None
        self.debug = debug
        self.alpha = alpha
        self.vip_prob_threshold = vip_prob_threshold
        self.no_move_counts = {}  # Suivi du nombre de tours consécutifs de "stay" pour chaque robot

    def wakeUp(self, playerId, numberOfPlayers, gameConfiguration):
        self._id = playerId
        self._model = GameEngine()
        self._model.fromPod(gameConfiguration)
        self.vip_bot.wakeUp(0, numberOfPlayers, gameConfiguration)
        if self.enemy_bot is not None:
            self.enemy_bot.wakeUp(2, numberOfPlayers, gameConfiguration)
        if self.debug:
            print(f"[DEBUG] ProbabilisticPredictiveBot {self._id} wakeUp")
        nb = self._model.numberOfMobiles(self._id)
        for r in range(1, nb + 1):
            self.no_move_counts[r] = 0

    def perceive(self, state):
        self._model.setOnState(state)
        if self.debug:
            self._model.render()
            time.sleep(0.3)
            sc = self._model.score(self._id)
            print(f"[DEBUG] ProbabilisticPredictiveBot {self._id} perceive - tic={self._model.tic()}, score={sc}")

    def sleep(self, result):
        return

    # -----------------------------------------------------------------------
    # is_valid_move et available_moves
    # -----------------------------------------------------------------------
    def is_valid_move(self, state, player, robot, move_command):
        parts = move_command.split()
        if parts[0] == "pass":
            return False
        if parts[0] == "mission":
            return True
        if parts[0] == "move":
            _, srobot, sdir = parts
            robot = int(srobot)
            direction = int(sdir)
        else:
            robot = int(parts[0])
            direction = int(parts[1])
        current_pos = state.mobilePosition(player, robot)
        candidate_tile = state.map().clockposition(current_pos, direction)
        if candidate_tile < 1 or candidate_tile > state.map().size():
            return False
        return True

    def available_moves(self, state, player, robot):
        moves = []
        pos = state.mobilePosition(player, robot)
        directions = state.map().clockBearing(pos)
        for d in directions:
            if d != 0:
                move_cmd = f"move {robot} {d}"
                if self.is_valid_move(state, player, robot, move_cmd):
                    moves.append(move_cmd)
        for mission_id in state.missionsList():
            mis = state.mission(mission_id)
            if mis.start == pos and mis.owner == 0:
                moves.append(f"mission {robot} {mission_id}")
        if self.debug:
            print(f"[DEBUG] available_moves - player={player}, robot={robot}, pos={pos}, moves={moves}")
        return moves

    # -----------------------------------------------------------------------
    # BFS local et next_direction
    # -----------------------------------------------------------------------
    def bfs_path(self, start_tile, goal_tile, blocked_tiles=None):
        if blocked_tiles is None:
            blocked_tiles = set()
        if start_tile == goal_tile:
            return [start_tile]
        queue = deque([start_tile])
        visited = {start_tile: None}
        while queue:
            cur = queue.popleft()
            for nxt in self._model.map().neighbours(cur):
                if nxt in blocked_tiles:
                    continue
                if nxt not in visited:
                    visited[nxt] = cur
                    if nxt == goal_tile:
                        path = [nxt]
                        while path[-1] != start_tile:
                            path.append(visited[path[-1]])
                        path.reverse()
                        return path
                    queue.append(nxt)
        return None

    def next_direction(self, from_tile, to_tile):
        dirs = self._model.map().clockBearing(from_tile)
        neighs = self._model.map().neighbours(from_tile)
        for d, n in zip(dirs, neighs):
            if n == to_tile:
                return d
        return 0

    # -----------------------------------------------------------------------
    # Missions
    # -----------------------------------------------------------------------
    def find_closest_mission(self, tile):
        best_mid = None
        best_dist = 999999
        for mid in self._model.missionsList():
            mis = self._model.mission(mid)
            if mis.owner == 0:
                dist = abs(mis.start - tile)
                if dist < best_dist:
                    best_dist = dist
                    best_mid = mid
        return best_mid

    # -----------------------------------------------------------------------
    # Adversary BFS (1 step) et missions pour adversaires
    # -----------------------------------------------------------------------
    def adversary_bfs_1step(self, pos, goal):
        path = self.bfs_path(pos, goal, blocked_tiles=set())
        if path and len(path) >= 2:
            return path[1]
        return None

    def find_closest_mission_for_adversary(self, p, pos):
        best_mid = None
        best_dist = 999999
        for mid in self._model.missionsList():
            mis = self._model.mission(mid)
            if mis.owner == 0:
                dist = abs(mis.start - pos)
                if dist < best_dist:
                    best_dist = dist
                    best_mid = mid
        return best_mid

    # -----------------------------------------------------------------------
    # VIP distribution et prédiction adversaires
    # -----------------------------------------------------------------------
    def predict_vip_distribution(self, vip_pos):
        if hasattr(self.vip_bot, "predict_next_move_distribution"):
            dist = self.vip_bot.predict_next_move_distribution(vip_pos, self._model.map())
            positions_probs = {}
            for direction, pr in dist.items():
                if direction == 0:
                    positions_probs[vip_pos] = pr
                else:
                    nxt = self._model.map().clockposition(vip_pos, direction)
                    positions_probs[nxt] = pr
            return positions_probs
        else:
            return {vip_pos: 1.0}

    def predict_adversary_positions(self):
        predicted = {}
        vip_count = self._model.numberOfMobiles(0)
        if vip_count > 0:
            vip_pos = self._model.mobilePosition(0, 1)
            vip_dist = self.predict_vip_distribution(vip_pos)
            for t, p in vip_dist.items():
                predicted[t] = predicted.get(t, 0) + p
        for p in range(self._model.numberOfPlayers() + 1):
            if p == self._id or p == 0:
                continue
            nb_r = self._model.numberOfMobiles(p)
            for r in range(1, nb_r + 1):
                adv_pos = self._model.mobilePosition(p, r)
                mmid = self.find_closest_mission_for_adversary(p, adv_pos)
                if mmid is not None:
                    m = self._model.mission(mmid)
                    nxt = self.adversary_bfs_1step(adv_pos, m.start)
                    if nxt is not None:
                        predicted[nxt] = predicted.get(nxt, 0) + 1.0
                    else:
                        predicted[adv_pos] = predicted.get(adv_pos, 0) + 1.0
                else:
                    predicted[adv_pos] = predicted.get(adv_pos, 0) + 1.0
        return predicted

    # -----------------------------------------------------------------------
    # BFS naïf pour un seul move
    # -----------------------------------------------------------------------
    def basic_path_decision(self, robot_id, blocked_tiles=None):
        pos = self._model.mobilePosition(self._id, robot_id)
        mid_active = self._model.mobileMission(self._id, robot_id)
        if mid_active != 0:
            m = self._model.mission(mid_active)
            if pos == m.final:
                return f"mission {robot_id} {mid_active}"
            path = self.bfs_path(pos, m.final, blocked_tiles)
            if not path or len(path) < 2:
                return f"{robot_id} 0"
            d = self.next_direction(pos, path[1])
            return f"{robot_id} {d}"
        else:
            mmid = self.find_closest_mission(pos)
            if mmid is None:
                return f"{robot_id} 0"
            m = self._model.mission(mmid)
            if pos == m.start:
                return f"mission {robot_id} {mmid}"
            path = self.bfs_path(pos, m.start, blocked_tiles)
            if not path or len(path) < 2:
                return f"{robot_id} 0"
            d = self.next_direction(pos, path[1])
            return f"{robot_id} {d}"

    # -----------------------------------------------------------------------
    # Mini-simulation alternative
    # -----------------------------------------------------------------------
    def simulate_alternative(self, alt_path, nb_turns=8, robot_id=1):
        pod = self._model.asPod()
        sim_state = self._model.fromPod(pod)
        idx = 0
        for turn in range(nb_turns):
            if idx < len(alt_path) - 1:
                cur = alt_path[idx]
                nxt = alt_path[idx + 1]
                d = self.next_direction(cur, nxt)
                mv = f"{robot_id} {d}"
                idx += 1
            else:
                mv = f"{robot_id} 0"
            self.mc_simulate_turn(sim_state, mv, robot_id)
        my_score = sim_state.score(self._id)
        opp_score = sim_state.score(1)
        return my_score - opp_score

    def mc_simulate_turn(self, sim_state, move_for_us, robot_id):
        self.apply_move(sim_state, self._id, move_for_us)
        for p in range(sim_state.numberOfPlayers() + 1):
            if p == self._id:
                continue
            if p == 0 and self.vip_bot is not None:
                mv = self.vip_bot.decide()
                self.apply_move(sim_state, p, mv)
            elif p == 2 and self.enemy_bot is not None:
                mv = self.enemy_bot.decide()
                self.apply_move(sim_state, p, mv)
            else:
                opp_moves = []
                nb_r = sim_state.numberOfMobiles(p)
                for r in range(1, nb_r + 1):
                    cands = self.available_moves(sim_state, p, r)
                    if not cands:
                        continue
                    ws = []
                    for c in cands:
                        h = self.heuristic_move(sim_state, p, c)
                        ws.append(math.exp(self.alpha * h))
                    if not ws:
                        continue
                    choice = weighted_random_choice(cands, ws)
                    self.apply_move(sim_state, p, choice)
        sim_state.applyMoveActions()

    # -----------------------------------------------------------------------
    # Reconstitution de chemin avec blocage
    # -----------------------------------------------------------------------
    def reconstitute_path(self, robot_id, blocked_tiles):
        pos = self._model.mobilePosition(self._id, robot_id)
        mid_active = self._model.mobileMission(self._id, robot_id)
        if mid_active != 0:
            m = self._model.mission(mid_active)
            return self.bfs_path(pos, m.final, blocked_tiles)
        else:
            mmid = self.find_closest_mission(pos)
            if mmid is None:
                return None
            return self.bfs_path(pos, self._model.mission(mmid).start, blocked_tiles)

    def detect_blocked(self, path, blocked):
        for i, t in enumerate(path):
            if t in blocked:
                if self.debug:
                    print(f"[DEBUG] detect_blocked: tile {t} at index {i} is blocked")
                return i
        return None

    def generate_detours(self, path, index_block, blocked):
        if index_block == 0:
            return []
        partial = path[:index_block]
        start = partial[-1]
        goal = path[-1]
        local_blocked = {path[index_block]}
        for n in self._model.map().neighbours(path[index_block]):
            local_blocked.add(n)
        block_union = blocked.union(local_blocked)
        alt = self.bfs_path(start, goal, block_union)
        if not alt:
            return []
        return [partial[:-1] + alt]

    # -----------------------------------------------------------------------
    # Final collision check
    # -----------------------------------------------------------------------
    def final_collision_check(self, robot_id, move_str):
        parts = move_str.split()
        if parts[0] == "mission":
            return move_str
        if len(parts) < 2:
            return move_str
        try:
            r = int(parts[0])
            d = int(parts[1])
        except:
            return move_str
        curr = self._model.mobilePosition(self._id, r)
        nxt = self._model.map().clockposition(curr, d)
        for p in range(self._model.numberOfPlayers() + 1):
            nb_r = self._model.numberOfMobiles(p)
            # Ne vérifier que si le joueur possède au moins un mobile
            if nb_r == 0:
                continue
            for rr in range(1, nb_r + 1):
                if self._model.mobilePosition(p, rr) == nxt:
                    if self.debug:
                        print(f"[DEBUG] Collision detected for robot {r}: target tile {nxt} is occupied (player {p}, robot {rr}). Fallback to '{r} 0'")
                    return f"{r} 0"
        return move_str

    # -----------------------------------------------------------------------
    # Get next tile and resolve conflict
    # -----------------------------------------------------------------------
    def get_next_tile(self, robot_id, action):
        parts = action.split()
        if parts[0] == "mission":
            return None
        if len(parts) < 2:
            return None
        try:
            r = int(parts[0])
            d = int(parts[1])
            pos = self._model.mobilePosition(self._id, r)
            nxt = self._model.map().clockposition(pos, d)
            return nxt
        except:
            return None

    def resolve_conflict(self, r, tile_target, blocked_tiles):
        if self.debug:
            print(f"[DEBUG] resolve_conflict: Conflict for robot {r} on tile {tile_target}. Forcing stay.")
        return f"{r} 0"

    # -----------------------------------------------------------------------
    # Flee decision: choisir un mouvement qui s'éloigne des adversaires
    # -----------------------------------------------------------------------
    def flee_decision(self, robot_id):
        available = self.available_moves(self._model, self._id, robot_id)
        if not available:
            return f"{robot_id} 0"
        adv = self.predict_adversary_positions()
        best_move = None
        best_safety = -math.inf
        for move in available:
            if move.startswith("mission"):
                return move
            parts = move.split()
            if parts[0] == "move":
                _, srobot, sdir = parts
            else:
                srobot, sdir = parts
            robot = int(srobot)
            direction = int(sdir)
            current_pos = self._model.mobilePosition(self._id, robot)
            next_tile = self._model.map().clockposition(current_pos, direction)
            safety = math.inf
            for tile, pr in adv.items():
                center_next = self._model.map().tile(next_tile).center()
                center_adv = self._model.map().tile(tile).center()
                dx = center_next.x() - center_adv.x()
                dy = center_next.y() - center_adv.y()
                dist = math.sqrt(dx * dx + dy * dy)
                safety = min(safety, dist)
            if safety > best_safety:
                best_safety = safety
                best_move = move
        if best_move is None:
            return f"{robot_id} 0"
        return best_move

    # -----------------------------------------------------------------------
    # Méthode pour détecter si un robot est en coin
    # -----------------------------------------------------------------------
    def is_in_corner(self, tile):
        center = self._model.map().tile(tile).center()
        all_centers = [self._model.map().tile(t).center() for t in range(1, self._model.map().size() + 1)]
        xs = [c.x() for c in all_centers]
        ys = [c.y() for c in all_centers]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        eps = 0.01
        if (abs(center.x() - min_x) < eps or abs(center.x() - max_x) < eps) and \
           (abs(center.y() - min_y) < eps or abs(center.y() - max_y) < eps):
            return True
        return False

    # -----------------------------------------------------------------------
    # Decide one robot with blocked et stratégie de fuite
    # -----------------------------------------------------------------------
    def decide_one_robot_with_blocked(self, robot_id, blocked_tiles):
        basic_decision = self.basic_path_decision(robot_id, blocked_tiles)
        if basic_decision.endswith(" 0"):
            self.no_move_counts[robot_id] = self.no_move_counts.get(robot_id, 0) + 1
        else:
            self.no_move_counts[robot_id] = 0
        # Si le robot est bloqué depuis 2 tours et n'est pas en coin, utiliser la stratégie de fuite.
        if self.no_move_counts.get(robot_id, 0) >= 2:
            current_tile = self._model.mobilePosition(self._id, robot_id)
            if not self.is_in_corner(current_tile):
                available = self.available_moves(self._model, self._id, robot_id)
                non_stay = [m for m in available if not m.endswith(" 0")]
                if non_stay:
                    rand_move = random.choice(non_stay)
                    if self.debug:
                        print(f"[DEBUG] Robot {robot_id} blocked for 2+ turns, switching to flee move: {rand_move}")
                    self.no_move_counts[robot_id] = 0
                    return rand_move
        mv = self.basic_path_decision(robot_id, blocked_tiles)
        if mv.startswith("mission"):
            return mv
        path = self.reconstitute_path(robot_id, blocked_tiles)
        if not path or len(path) < 2:
            return f"{robot_id} 0"
        idx_block = self.detect_blocked(path, blocked_tiles)
        if idx_block is None:
            return mv
        alt_paths = self.generate_detours(path, idx_block, blocked_tiles)
        if not alt_paths:
            return f"{robot_id} 0"
        best_sc = -9999
        best_alt = None
        pos = self._model.mobilePosition(self._id, robot_id)
        for alt in alt_paths:
            sc = self.simulate_alternative(alt, nb_turns=8, robot_id=robot_id)
            if sc > best_sc:
                best_sc = sc
                best_alt = alt
        if best_alt and len(best_alt) > 1:
            d = self.next_direction(pos, best_alt[1])
            return f"{robot_id} {d}"
        return f"{robot_id} 0"

    # -----------------------------------------------------------------------
    # Final decision assembly
    # -----------------------------------------------------------------------
    def decide(self):
        nb = self._model.numberOfMobiles(self._id)
        if not hasattr(self, "no_move_counts"):
            self.no_move_counts = {r: 0 for r in range(1, nb + 1)}
        robot_ids = list(range(1, nb + 1))
        actions_per_robot = {}
        blocked_this_turn = set()
        adv_predict = self.predict_adversary_positions()
        blocked_pred = {t for (t, pr) in adv_predict.items() if pr >= self.vip_prob_threshold}
        blocked_this_turn.update(blocked_pred)
        allpos = {}
        for rr in robot_ids:
            allpos[rr] = self._model.mobilePosition(self._id, rr)
        for r in robot_ids:
            blocked_local = set(blocked_this_turn)
            for rr in robot_ids:
                if rr != r:
                    blocked_local.add(allpos[rr])
            mv_r = self.decide_one_robot_with_blocked(r, blocked_local)
            mv_r = self.final_collision_check(r, mv_r)
            currpos = allpos[r]
            nxtpos = self.get_next_tile(r, mv_r)
            if nxtpos and nxtpos in blocked_this_turn:
                alt_move = self.resolve_conflict(r, nxtpos, blocked_this_turn)
                mv_r = alt_move
                nxtpos = self.get_next_tile(r, alt_move)
            blocked_this_turn.add(currpos)
            if nxtpos:
                blocked_this_turn.add(nxtpos)
            actions_per_robot[r] = mv_r
            parts = mv_r.split()
            if len(parts) == 2 and parts[0].isdigit():
                ro = int(parts[0])
                di = int(parts[1])
                if ro == r:
                    newtile = self._model.map().clockposition(currpos, di)
                    allpos[r] = newtile
        mission_cmd = []
        move_cmd = []
        for r in robot_ids:
            act = actions_per_robot[r]
            parts = act.split()
            if parts[0] == "mission":
                mission_cmd.append(parts[1])
                mission_cmd.append(parts[2])
            else:
                p2 = act.split()
                if p2[0] == "move":
                    p2 = p2[1:]
                if len(p2) == 2:
                    move_cmd.extend(p2)
                else:
                    move_cmd.append(p2[0])
                    move_cmd.append("0")
        final_cmds = []
        if mission_cmd:
            final_cmds.append("mission " + " ".join(mission_cmd))
        if move_cmd:
            final_cmds.append("move " + " ".join(move_cmd))
        decision = " ".join(final_cmds).strip()
        if decision.startswith("move move"):
            decision = "move" + decision[len("move move"):]
        if decision == "":
            decision = "pass"
        if self.debug:
            print(f"[DEBUG] final decision => {decision}")
        return decision

    def apply_move(self, state, player, move_command):
        cmds = move_command.split("move")
        for cmd in cmds:
            cmd = cmd.strip()
            if not cmd:
                continue
            if cmd.startswith("mission"):
                parts = cmd.split()
                r = int(parts[1])
                mid = int(parts[2])
                state.missionAction(player, r, mid)
            else:
                parts = cmd.split()
                if parts[0] == "pass":
                    continue
                r = int(parts[0])
                d = int(parts[1])
                state.setMoveAction(player, r, d)
        state.applyMoveActions()