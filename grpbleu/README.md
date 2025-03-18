![SmartBot](https://media.discordapp.net/attachments/1338893102882095104/1350104650904240149/smartbot.webp?ex=67d58670&is=67d434f0&hm=c35a34540833a1876ca5550628162940838303cca02e3c29c27b21a44992c542&=&format=webp&width=877&height=877)

# PAICO

## Project

https://github.com/users/AlexisEgea/projects/3  

## Group 

Alexis Egea  
Elouan Fronville  
Zakaria Si Salah  


# AI Bot for the Game "Move It"

This project implements an AI bot for the game "Move It," developed in Python. The bot is designed to control the players' robots, manage missions, calculate optimal paths, and handle interactions with VIPs in a strategic game environment.

## Project Structure

```
├── src/                       # Source code directory
│   ├── bot.py                 # Main bot implementation
│   ├── game/                  # Game mechanics and logic
│   │   ├── game.py            # Basic game functionalities
│   │   ├── path.py            # Pathfinding algorithms
│   │   └── __init__.py
│   ├── mission/               # Mission management
│   │   ├── mission.py         # Mission representation
│   │   ├── mission_manager.py # Mission assignment and management
│   │   └── __init__.py
│   ├── player/                # Player-related components
│   │   ├── player.py          # Player representation and robot management
│   │   └── __init__.py
│   ├── robot/                 # Robot components
│   │   ├── robot.py           # Robot logic and behavior
│   │   ├── vip.py             # VIP management and movement prediction
│   │   └── __init__.py
│   ├── utils/                 # Utility functions
│   │   ├── action_builder.py  # Helper for building game actions
│   │   ├── display.py         # Visualization utilities
│   │   ├── evaluation.py      # Game state evaluation functions
│   │   └── __init__.py
│
├── team.py                    # Team information and entry point
└── tests/                     # Tests directory
```

## Main Components

### Game

The `Game` class serves as the central component that orchestrates the game state, including:
- Building and initializing the game model
- Managing player and robot initialization
- Handling distance calculations between tiles
- Calculating optimal paths

The game uses a distance matrix for efficient pathfinding, with methods such as:
- `computeDistances`: Calculates distances from a single tile to all others using a flood fill algorithm
- `computeAllDistances`: Pre-computes the complete distance matrix for the map
- `path`: Determines the optimal path between two tiles

### Path

The `Path` class implements advanced pathfinding algorithms:
- `findDaWay`: Main method to find the most optimal path
- `multiPath`: Recursive algorithm to find all possible shortest paths
- `multiLengthPath`: Finds paths of a specific length
- `moveToward`: Calculates optimal movements toward a target
- `filterPaths`: Filters paths to avoid collisions with other robots

### Mission System

Missions are a key element of the game:
- `Mission`: Represents individual missions with start, end, reward, and owner
- `MissionManager`: Manages mission assignment, tracking, and prioritization
- Mission assignment uses evaluation metrics to find the best robot-mission pairs

### Robot Management

The bot controls multiple robots:
- `Robot`: Represents an individual robot with position and mission tracking
- `Player`: Manages a collection of robots for a specific player
- Robots can select missions, execute them, and track progress

### VIP Management

The `VIP` class provides advanced functionalities for interacting with VIPs:
- Tracking VIP positions and predicting movements
- Collision avoidance strategies
- Evaluating path safety around VIP presence

## Key Algorithms

1. **Distance Calculation**: Flood fill algorithm to calculate distances between tiles
2. **Pathfinding**: Multiple algorithms to find optimal and alternative paths
3. **Mission Assignment**: Evaluation-based system to match robots with optimal missions
4. **VIP Movement Prediction**: Probabilistic model to predict VIP movements
5. **Collision Avoidance**: Intelligent routing to minimize collision risk with VIPs and other robots

## Usage

The bot operates as follows:
1. Initializes the game state
2. Computes the distance matrix for efficient pathfinding
3. Assigns missions to robots based on strategic evaluation
4. Calculates optimal paths while avoiding collisions
5. Executes movements and adapts to changing game conditions

## Dependencies

- The project uses the `hacka.games.moveit` package, which provides the main game engine functionalities

---
