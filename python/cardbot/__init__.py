"""cardbot: training and evaluation for trick-taking card-game bots.

The C++ side (cardbot-framework/cpp) holds the games' engines, the batched
self-play environment and the search; this package trains the networks the
search uses (docs/DESIGN.md, steps 4 to 8) and evaluates them. A game brings
a pybind11 module made with cardbot/python.h; everything here works on it.
"""

__version__ = "0.1.0"
