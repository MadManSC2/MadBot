import sc2, sys
from __init__ import run_ladder_game
from sc2 import Race, Difficulty
from sc2.player import Bot, Computer

# Load bot
from MadBot import MadBot
bot = Bot(Race.Protoss, MadBot())

# Start game
if __name__ == '__main__':
    if "--LadderServer" in sys.argv:
        # Ladder game started by LadderManager
        print("Starting ladder game...")
        result, opponentid = run_ladder_game(bot)
        print(result, " against opponent ", opponentid)
    else:
        # Local game
        print("Starting local game...")
        maps = [
            'Automaton LE',
            'Blueshift LE',
            'Cerulean Fall LE',
            'Kairos Junction LE',
            'Para Site LE',
            'Port Aleksander LE',
            'Stasis LE',
            'Darkness Sanctuary LE'
        ]

        races = [
            Race.Protoss,
            Race.Terran,
            Race.Zerg
        ]

        selected_map = 3
        race = 0
        sc2.run_game(sc2.maps.get(maps[selected_map]), [
            Bot(Race.Protoss, MadBot()),
            Computer(races[race], Difficulty.VeryHard)
        ], realtime=False)
