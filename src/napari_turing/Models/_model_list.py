from enum import Enum
from .FitzHughNagumo import FitzHughNagumo
from .Brusselator import Brusselator
from .GrayScott import GrayScott
from .GameOfLife import GameOfLife
from .Oregonator import Oregonator

class AvailableModels(Enum):
    FitzHughNagumo = FitzHughNagumo
    Brusselator = Brusselator
    GrayScott = GrayScott
    GameOfLife = GameOfLife
    Oregonator = Oregonator
