from operator import length_hint


class Path:
    def __init__(self, iTile: int, iTarget: int, game):
        self.iTile, self.iTarget = iTile, iTarget
        self.game = game

    def findDaWay(self):
        shortestPaths = self.multiPath()
        if len(shortestPaths) > 0:
            length = len(shortestPaths[0][0])
            validShortPaths = [path for path in shortestPaths if self.filterPaths(path)]
        else:
            length = 0
            validShortPaths = []
        if len(validShortPaths) == 0:
            while length <= 25:
                length += 1
                lengthPaths = self.multiLengthPath(length)
                validLengthPaths = [path for path in lengthPaths if self.filterPaths(path)]
                if len(validLengthPaths) > 0:
                    path = validLengthPaths[0]
                    return path
            return [0],[0]
        else:
            path = validShortPaths[0]
            return path

    def filterPaths(self, path: tuple):
        for player in self.game.getPlayers():
            for robot in player.getRobots():
                if robot.getPosition() in path[1]:
                    return False
        return True

    def moveEverywhere(self, tile: int, target: int) -> tuple:
        """Version améliorée retournant tous les mouvements optimaux possibles

        Args:
            iTile (int): Position de départ
            iTarget (int): Position cible

        Returns:
            tuple: (list[int], list[int]) Liste des directions et des tuiles suivantes
        """
        if tile == target:
            return [0], [tile]

        clockdirs = list(self.game.getModel().map().clockBearing(tile))
        nextTiles = list(self.game.getModel().map().neighbours(tile))
        if 0 in clockdirs:
            clockdirs.remove(0)
        if tile in nextTiles:
            nextTiles.remove(tile)
        return clockdirs,nextTiles

    def multiLengthPath(self, length) -> list:
        """Calcule le chemin optimal entre deux tuiles

        Args:
            iTile (int): Position de départ
            iTarget (int): Position cible

        Returns:
            list: list(tuple) Liste des mouvements et chemin
        """
        tile, target = self.iTile, self.iTarget
        dirs, nexts = self.moveEverywhere(tile, target)
        paths = [([dir], [next]) for dir, next in zip(dirs, nexts)]
        inMemory = [tile]
        inMemory.extend(nexts)
        inMemory.remove(target) if target in inMemory else None
        index = 0
        while index < len(paths):
            path = paths[index]
            dirPath, nextPath = path

            while nextPath[-1] != target:

                inMemory.remove(target) if target in inMemory else None
                if len(nextPath) > length:
                    paths.remove(path)
                    index -= 1
                    break

                dirs, nexts = self.moveEverywhere(nextPath[-1], target)

                filteredMoves = [(dirIn, nextIn) for dirIn, nextIn in zip(dirs, nexts) if nextIn not in inMemory]
                if not filteredMoves:
                    paths.remove(path)
                    index -= 1
                    break

                dirNext, nextNext = filteredMoves.pop(0)
                for dirIn, nextIn in filteredMoves:
                    inMemory.append(nextIn)
                    paths.append((dirPath + [dirIn], nextPath + [nextIn]))
                path[0].append(dirNext)
                path[1].append(nextNext)
                inMemory.append(nextNext)
            index += 1
        return paths


    def moveToward(self, tile: int, target: int) -> tuple:
        """Version améliorée retournant tous les mouvements optimaux possibles

        Args:
            iTile (int): Position de départ
            iTarget (int): Position cible

        Returns:
            tuple: (list[int], list[int]) Liste des directions et des tuiles suivantes
        """
        # If no need to move:
        if tile == target :
            return 0, tile
        # Get candidates:
        clockdirs= list(self.game._model.map().clockBearing(tile))
        nextTiles= list(self.game._model.map().neighbours(tile))
        # Test all candidates:
        selectedDir = []
        selectedNext = []
        for clock, iterTile in zip( clockdirs, nextTiles ) :
            if self.game._distances[iterTile][target] < self.game._distances[tile][target] :
                selectedDir.append(clock)
                selectedNext.append(iterTile)
        # Return the selected candidates:
        return selectedDir, selectedNext

    def multiPath(self) -> list:
        def recursive_path( de, current_tile, target, current_move, current_path):
            # Base case: If the current tile is the target, return the completed path
            if current_tile == target:
                return [(current_move, current_path)]

            if de > 20 :
                return [(current_move, current_path)]

            # Get the possible moves and tiles from moveToward
            clocks, tiles = self.moveToward(current_tile, target)
            # If moveToward returns multiple options, recurse for each option
            paths = []
            for clock, tile in zip( clocks, tiles ) :
                # Recurse with the new tile, adding the current move and tile to the path
                de= de+1
                paths.extend(
                    recursive_path( de, tile, target, current_move + [clock], current_path + [tile])
                )
            return paths
        # Start the recursion with the initial tile and empty paths
        r= recursive_path( 0, self.iTile, self.iTarget, [], [])
        return r
