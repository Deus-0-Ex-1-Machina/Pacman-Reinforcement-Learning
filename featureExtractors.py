# featureExtractors.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

def closestTarget(pos, targets, walls):
    """
    BFS distance from pos to the nearest tile in the set targets.
    Returns None if no target is reachable.
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        if (pos_x, pos_y) in targets:
            return dist
        for nbr_x, nbr_y in Actions.getLegalNeighbors((pos_x, pos_y), walls):
            fringe.append((nbr_x, nbr_y, dist+1))
    return None

# A scared ghost is only chased if Pac-Man can reach it with this many turns
# to spare before its scaredTimer runs out (guards against the ghost turning
# dangerous again during the chase).
SCARED_MARGIN = 1

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a Pacman that avoids dangerous ghosts and
    chases scared (edible) ghosts:
    - whether food will be eaten (only when no edible ghost is catchable)
    - how far away the next food is (same gating)
    - whether a dangerous ghost is one step away
    - whether a catchable scared ghost is one step away
    - how far away the nearest catchable scared ghost is
    - whether the next move captures a catchable scared ghost (unscaled)

    A scared ghost is "catchable" if its remaining timer exceeds the Pac-Man
    BFS distance to it plus SCARED_MARGIN. Scared ghosts that cannot be
    reached in time (timer about to expire) are treated as dangerous.
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghostStates = state.getGhostStates()
        area = float(walls.width * walls.height)

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        nextPos = (next_x, next_y)

        dangerous = []  # ghost positions Pac-Man must avoid
        catchable = []  # (tile, distance) for scared ghosts Pac-Man can catch in time
        for gs in ghostStates:
            gpos = gs.getPosition()
            if gs.scaredTimer == 0:
                dangerous.append(gpos)
                continue
            tile = (int(gpos[0] + 0.5), int(gpos[1] + 0.5))
            d = closestTarget(nextPos, set([tile]), walls)
            if d is None:
                continue  # unreachable: neither a threat nor a target
            if gs.scaredTimer > d + SCARED_MARGIN:
                catchable.append((tile, d))
            else:
                # timer will run out before we get there: treat as dangerous
                dangerous.append(gpos)

        # count the number of dangerous ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum(nextPos in Actions.getLegalNeighbors(g, walls) for g in dangerous)

        # count the number of catchable scared ghosts 1-step away
        features["#-of-scared-ghosts-1-step-away"] = sum(d <= 1 for _, d in catchable)

        if catchable:
            # Priority: while an edible ghost is catchable, chase it instead of food
            scaredDist = min(d for _, d in catchable)
            features["closest-scared-ghost"] = float(scaredDist) / area
        else:
            # if there is no danger of ghosts then add the food feature
            if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
                features["eats-food"] = 1.0

            dist = closestFood(nextPos, food, walls)
            if dist is not None:
                # make the distance a number less than one otherwise the update
                # will diverge wildly
                features["closest-food"] = float(dist) / area

        features.divideAll(10.0)

        # Capture indicator is added after scaling so its weight is not shrunk
        # relative to the other ghost features.
        if any(d == 0 for _, d in catchable):
            features["captures-scared-ghost"] = 1.0

        return features
