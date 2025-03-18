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

class MultiDangerPathMCBot(AbsPlayer):
    """
    Bot coordonné qui combine :
      1) Un BFS pour aller vers les missions.
      2) Mini-simulation pour contourner des obstacles (generate_detours + simulate_alternative).
      3) Coordination élémentaire des robots (on décide dans l'ordre,
         bloque la case courante + la case visée pour éviter collision interne).
      4) Bloque la tuile occupée par tous les adversaires dans BFS (ou du moins la tuile du VIP).
      5) "Adversary danger check": si un adversaire est adjacent
         et peut bouger sur notre tuile, on tente un mouvement d'évitement
         (sinon on fait BFS).
      6) Fallback "r 0" si rien n'est possible.
      7) Assemblage final "mission ... move ...".
    """

    def __init__(self, vip_bot, enemy_bot, debug=False, alpha=1.0):
        super().__init__()
        self.vip_bot = vip_bot      # Contrôle player=0 (VIP) ou non
        self.enemy_bot = enemy_bot  # Contrôle player=2
        self._id = None
        self._model = None
        self.debug = debug
        self.alpha = alpha

    def wakeUp(self, playerId, numberOfPlayers, gameConfiguration):
        self._id = playerId
        self._model = GameEngine()
        self._model.fromPod(gameConfiguration)

        self.vip_bot.wakeUp(0, numberOfPlayers, gameConfiguration)
        self.enemy_bot.wakeUp(2, numberOfPlayers, gameConfiguration)

        if self.debug:
            print(f"[DEBUG] MultiDangerPathMCBot {self._id} wakeUp - nbPlayers={numberOfPlayers}")

    def perceive(self, state):
        self._model.setOnState(state)
        if self.debug:
            sc = self._model.score(self._id)
            print(f"[DEBUG] MultiDangerPathMCBot {self._id} perceive - tic={self._model.tic()}, score={sc}")

    def sleep(self, result):
        return
        #print(f"MultiDangerPathMCBot {self._id} ended with result: {result}")

    # -----------------------------------------------------------------------
    # BFS
    # -----------------------------------------------------------------------

    def bfs_path(self, start_tile, goal_tile, blocked_tiles=None, block_adversaries=True):
        """
        BFS de start_tile à goal_tile en ignorant blocked_tiles.
        Si block_adversaries=True, on ajoute aussi la tuile de tous les adversaires
        (player != self._id) dans blocked_tiles,
        pour éviter de marcher sur un adversaire statique ce tour-ci.
        """
        if blocked_tiles is None:
            blocked_tiles = set()

        if block_adversaries:
            # Pour chaque adversaire (player != self._id),
            # on ajoute la position de ses robots comme bloquées.
            for p in range(self._model.numberOfPlayers()+1):
                if p != self._id:
                    nb_r = self._model.numberOfMobiles(p)
                    for r in range(1, nb_r+1):
                        adv_pos = self._model.mobilePosition(p, r)
                        blocked_tiles.add(adv_pos)

        if start_tile == goal_tile:
            return [start_tile]

        from collections import deque
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
            m = self._model.mission(mid)
            if m.owner == 0:
                dist = abs(m.start - tile)
                if dist < best_dist:
                    best_dist = dist
                    best_mid = mid
        return best_mid

    # -----------------------------------------------------------------------
    # Heuristique, is_valid_move, apply_move
    # -----------------------------------------------------------------------

    def heuristic_move(self, state, player, move):
        parts = move.split()
        if parts[0] == "pass":
            return -1.0
        if parts[0] == "mission":
            return 2.0
        if parts[0] == "move":
            _, srobot, sdir = parts
            robot = int(srobot)
            direction = int(sdir)
        else:
            robot = int(parts[0])
            direction = int(parts[1])

        curr = state.mobilePosition(player, robot)
        nxt = state.map().clockposition(curr, direction)
        # +1 si rapproche mission.start
        dist_b = 9999
        dist_a = 9999
        for mid in state.missionsList():
            mis = state.mission(mid)
            db = abs(mis.start - curr)
            da = abs(mis.start - nxt)
            if db < dist_b:
                dist_b = db
            if da < dist_a:
                dist_a = da
        score = 0.0
        if dist_a < dist_b:
            score += 1.0
        else:
            score -= 0.5
        return score

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

        pos = state.mobilePosition(player, robot)
        nxt = state.map().clockposition(pos, direction)
        sz = state.map().size()
        if nxt < 1 or nxt > sz:
            return False
        return True

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

    # -----------------------------------------------------------------------
    # BFS naive => un seul move
    # -----------------------------------------------------------------------

    def basic_path_decision(self, robot_id, blocked_tiles=None):
        """
        BFS naive => "r d" ou "mission r mid" ou "r 0".
        On ignore collisions dyn, sauf qu'on applique blocked_tiles + block_adversaries.
        """
        pos = self._model.mobilePosition(self._id, robot_id)
        mid_active = self._model.mobileMission(self._id, robot_id)

        if mid_active != 0:
            m = self._model.mission(mid_active)
            if pos == m.final:
                return f"mission {robot_id} {mid_active}"
            path = self.bfs_path(pos, m.final, blocked_tiles=blocked_tiles, block_adversaries=True)
            if not path or len(path) < 2:
                return f"{robot_id} 0"
            d = self.next_direction(pos, path[1])
            return f"{robot_id} {d}"
        else:
            mid = self.find_closest_mission(pos)
            if mid is None:
                return f"{robot_id} 0"
            mm = self._model.mission(mid)
            if pos == mm.start:
                return f"mission {robot_id} {mid}"
            path = self.bfs_path(pos, mm.start, blocked_tiles=blocked_tiles, block_adversaries=True)
            if not path or len(path) < 2:
                return f"{robot_id} 0"
            d = self.next_direction(pos, path[1])
            return f"{robot_id} {d}"

    # -----------------------------------------------------------------------
    #  Mini-simulation
    # -----------------------------------------------------------------------

    def simulate_alternative(self, alt_path, nb_turns=8, robot_id=1):
        pod = self._model.asPod()
        sim_state = self._model.fromPod(pod)
        idx = 0
        for turn in range(nb_turns):
            if idx < len(alt_path)-1:
                cur = alt_path[idx]
                nxt = alt_path[idx+1]
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
        for p in range(0, sim_state.numberOfPlayers()+1):
            if p == self._id:
                continue
            if p == 0 and self.vip_bot is not None:
                mv = self.vip_bot.decide()
                self.apply_move(sim_state, p, mv)
            elif p == 2 and self.enemy_bot is not None:
                mv = self.enemy_bot.decide()
                self.apply_move(sim_state, p, mv)
            else:
                # random pondéré
                opp_moves = []
                nb_r = sim_state.numberOfMobiles(p)
                for r in range(1, nb_r+1):
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
    #  Reconstitution path => block => alt
    # -----------------------------------------------------------------------

    def reconstitute_path(self, robot_id, blocked_tiles):
        pos = self._model.mobilePosition(self._id, robot_id)
        mid_active = self._model.mobileMission(self._id, robot_id)
        if mid_active != 0:
            m = self._model.mission(mid_active)
            goal = m.final
        else:
            mid = self.find_closest_mission(pos)
            if mid is None:
                return None
            goal = self._model.mission(mid).start
        return self.bfs_path(pos, goal, blocked_tiles=blocked_tiles, block_adversaries=True)

    def detect_blocked(self, path, blocked):
        for i, tile in enumerate(path):
            if tile in blocked:
                return i
        return None

    def generate_detours(self, path, index_block, blocked):
        if index_block == 0:
            return []
        partial = path[:index_block]
        start = partial[-1]
        goal = path[-1]
        local_blocked = set([path[index_block]])
        for n in self._model.map().neighbours(path[index_block]):
            local_blocked.add(n)
        block_union = blocked.union(local_blocked)
        alt = self.bfs_path(start, goal, blocked_tiles=block_union, block_adversaries=True)
        if not alt:
            return []
        return [partial[:-1] + alt]

    # -----------------------------------------------------------------------
    #  final_collision_check
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
        for p in range(0, self._model.numberOfPlayers()+1):
            nb_r = self._model.numberOfMobiles(p)
            for rr in range(1, nb_r+1):
                if self._model.mobilePosition(p, rr) == nxt:
                    return f"{r} 0"
        return move_str

    # -----------------------------------------------------------------------
    #  Danger check: si un adversaire adjacent peut bouger sur nous
    # -----------------------------------------------------------------------

    def danger_check_adversaries(self, robot_id):
        """
        Vérifie si un adversaire (player != self._id, p.ex. vip ou enemy)
        est adjacent + peut se déplacer sur la tuile de 'robot_id'.
        Si oui, on tente un move d'évitement (une direction voisine).
        """
        pos = self._model.mobilePosition(self._id, robot_id)

        # Pour chaque adversaire
        for p in range(self._model.numberOfPlayers()+1):
            if p == self._id:
                continue
            nb_r = self._model.numberOfMobiles(p)
            for rr in range(1, nb_r+1):
                adv_pos = self._model.mobilePosition(p, rr)
                # On regarde si pos figure dans la liste des tuiles où adv peut bouger
                adv_dirs = self._model.map().clockBearing(adv_pos)
                adv_ngh = self._model.map().neighbours(adv_pos)
                # + le "stay in place" => 0
                possible_tiles = set()
                for d, n in zip(adv_dirs, adv_ngh):
                    possible_tiles.add(n)
                #  on ignore / ou on ajoute adv_pos si c'est "stay"
                #  ex: s'il n'y a pas un 'colliding stay' ?

                if pos in possible_tiles:
                    # => Danger : cet adversaire POURRAIT bouger sur nous
                    # On tente un move d'évitement
                    directions = self._model.map().clockBearing(pos)
                    neighbours = self._model.map().neighbours(pos)
                    for d, n in zip(directions, neighbours):
                        if d == 0:
                            continue
                        # On évite d'aller sur adv_pos
                        if n == adv_pos:
                            continue
                        # On fabrique "robot_id d"
                        cand_move = f"{robot_id} {d}"
                        # On peut check is_valid_move, final_collision_check, etc.
                        # ex:
                        return cand_move
                    # pas de direction => on pass
                    return f"{robot_id} 0"

        return None  # pas de danger immediat

    # -----------------------------------------------------------------------
    #  decide_one_robot_with_blocked
    # -----------------------------------------------------------------------

    def decide_one_robot_with_blocked(self, robot_id, blocked_tiles):
        """
        1) Danger check => on tente un move d'évitement si un adversaire
           peut nous rentrer dedans,
        2) BFS naive => path => alt => fallback,
        3) "r 0" si rien.
        """
        # Danger check
        avoid_move = self.danger_check_adversaries(robot_id)
        if avoid_move:
            # check valid
            # is_valid + final_collision_check
            if self.is_valid_move(self._model, self._id, robot_id, avoid_move):
                safe_avoid = self.final_collision_check(robot_id, avoid_move)
                return safe_avoid

        # BFS normal
        mv_basic = self.basic_path_decision(robot_id, blocked_tiles)
        if mv_basic.startswith("mission"):
            return mv_basic

        path = self.reconstitute_path(robot_id, blocked_tiles)
        if not path or len(path) < 2:
            return f"{robot_id} 0"

        idx_block = self.detect_blocked(path, blocked_tiles)
        if idx_block is None:
            return mv_basic

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
    #  get_next_tile + resolve_conflict
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
        return f"{r} 0"

    # -----------------------------------------------------------------------
    #  DECIDE
    # -----------------------------------------------------------------------

    def decide(self):
        nb = self._model.numberOfMobiles(self._id)
        robot_ids = list(range(1, nb+1))
        actions_per_robot = {}

        blocked_this_turn = set()
        all_positions = {}
        for rr in robot_ids:
            all_positions[rr] = self._model.mobilePosition(self._id, rr)

        for r in robot_ids:
            # blocked_local = blocked_this_turn + positions des autres robots
            blocked_local = set(blocked_this_turn)
            for rr in robot_ids:
                if rr != r:
                    blocked_local.add(all_positions[rr])

            mv_r = self.decide_one_robot_with_blocked(r, blocked_local)

            # collision check final
            mv_r = self.final_collision_check(r, mv_r)

            curr_pos = all_positions[r]
            nxt_pos = self.get_next_tile(r, mv_r)

            # resoud un conflit direct
            if nxt_pos and nxt_pos in blocked_this_turn:
                alt_move = self.resolve_conflict(r, nxt_pos, blocked_this_turn)
                mv_r = alt_move
                nxt_pos = self.get_next_tile(r, alt_move)

            # on bloque
            blocked_this_turn.add(curr_pos)
            if nxt_pos:
                blocked_this_turn.add(nxt_pos)

            actions_per_robot[r] = mv_r

            # on met à jour la position
            ps = mv_r.split()
            if len(ps) == 2 and ps[0].isdigit():
                ro = int(ps[0])
                di = int(ps[1])
                if ro == r:
                    newtile = self._model.map().clockposition(curr_pos, di)
                    all_positions[r] = newtile

        # assemble mission / move
        mission_cmds = []
        move_cmds = []
        for r in robot_ids:
            act = actions_per_robot[r]
            parts = act.split()
            if parts[0] == "mission":
                mission_cmds.append(parts[1])
                mission_cmds.append(parts[2])
            else:
                p2 = act.split()
                if len(p2) == 2:
                    move_cmds.append(p2[0])
                    move_cmds.append(p2[1])
                else:
                    move_cmds.append(p2[0])
                    move_cmds.append("0")

        final_cmds = []
        if mission_cmds:
            final_cmds.append("mission " + " ".join(mission_cmds))
        if move_cmds:
            final_cmds.append("move " + " ".join(move_cmds))

        decision = " ".join(final_cmds)
        if self.debug:
            print(f"[DEBUG] final decision => {decision}")
        return decision