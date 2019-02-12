## Disclaimer ##
# I started this project to learn both python and machine learning (ML) at the same time.
# This means that not only my code might be messy or sub-optimal, but also my ML techniques might be pretty basic.
# Yet, I tried to create a bot, that mostly relies on hard-coded build order, which are good in particular situations or against certain races.
# This version was meant to randomly choose a build order and gather scouting data at approximately 2 minutes elapsed game time, which is stored together with the result of the game.
# This training data will be used to train a neural network to choose the optimal build order based on the scouting information of the game instead of randomly choosing it right at the start.
# Hopefully, this will increase win rates a lot (it certainly did against the build in computer opponent, but thats a different story).
# The next step would be to identify more milestones within a typical game in which major choices had to be made and let a different network make these choices.
# Possible candidates might be: Attack now or keep defending/expanding, alter unit composition (e.g. Colossi instead of Immortals), Commit to a attack or retreat, etc.
# Thanks to sentdex, CreepyBot, Cannon-Lover and TapiocaBot for some inspiration

# Version 1.3: Due to the current Rush/Cheese Meta, I implemented a more defensive build order and in return deactivated the 2-Base Immortal BO.

# Version 1.4: Switched from randomly choosen build orders to scouting based build order. Yet, still not completely with a neural network but with basic rules, provided by a neural network.

# Version 1.5: Added a simple neural network to chose build orders based on scouting information. Local tests with hundreds of games revealed that win rates compared to random chosing increased
# from 44% to 71%. Bot used locally: YoBot, Tyr, Tyrz, 5minBot, BlinkerBot, NaughtyBot, SarsaBot, SeeBot, ramu, Micromachine, Kagamine, AviloBot, EarlyAggro, Voidstar, ReeBot

# Version 1.6: Adapted early game rush defense in order to deal better with 12 pools (e.g. by CheeZerg). Trained a new neural network with 730 games against the newest versions of most bots available.
# Also refined scouting on 4 player maps and tuned the late game emergency strat to prevent ties.

# Version 1.6.1: Bugfixes and new Model

# Version 1.7: Added a One-Base defence into Void-Ray build in order to deal with other very aggressive builds

# Version 1.7.1: Bugfixes and improved Voidray micro

# Version 1.7.2: Newly trained model

# Version 1.7.3 - 4: Small Bugfixes

# Version 1.7.5: Slightly improved Rush defence

# Version 1.8: Improved scouting with more scouting parameters, new model and various bug fixes / small improvements (see changelog)
import sc2
import sc2.units
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, STARGATE, DARKSHRINE, \
 CYBERNETICSCORE, ZEALOT, STALKER, OBSERVER, ROBOTICSFACILITY, COLOSSUS, DARKTEMPLAR, \
 ROBOTICSBAY, RESEARCH_EXTENDEDTHERMALLANCE, RESEARCH_WARPGATE, SENTRY, SHIELDBATTERY, \
 CHRONOBOOSTENERGYCOST, EFFECT_CHRONOBOOSTENERGYCOST, MORPH_WARPGATE, RALLY_UNITS, \
 WARPGATE, WARPGATETRAIN_ZEALOT, WARPGATETRAIN_STALKER, IMMORTAL, ADEPT, VOIDRAY, \
 PHOTONCANNON, FORGE, TWILIGHTCOUNCIL, RESEARCH_PROTOSSGROUNDARMOR, MORPH_ARCHON, ARCHON_WARP_TARGET, \
 RESEARCH_PROTOSSGROUNDWEAPONS, RESEARCH_PROTOSSSHIELDS, RESEARCH_CHARGE, RESEARCH_ADEPTRESONATINGGLAIVES, \
 RESEARCH_BLINK, CHARGE, BLINKTECH, PROTOSSGROUNDWEAPONSLEVEL1, PROTOSSGROUNDWEAPONSLEVEL2, \
 PROTOSSGROUNDWEAPONSLEVEL3, PROTOSSGROUNDARMORSLEVEL1, PROTOSSGROUNDARMORSLEVEL2, \
 PROTOSSGROUNDARMORSLEVEL3, PROTOSSSHIELDSLEVEL1, PROTOSSSHIELDSLEVEL2, \
 PROTOSSSHIELDSLEVEL3, FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL1, FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL2, \
 FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL3, FORGERESEARCH_PROTOSSGROUNDARMORLEVEL1, FORGERESEARCH_PROTOSSGROUNDARMORLEVEL2, \
 FORGERESEARCH_PROTOSSGROUNDARMORLEVEL3, FORGERESEARCH_PROTOSSSHIELDSLEVEL1, \
 GUARDIANSHIELD, GUARDIANSHIELD_GUARDIANSHIELD, HARVEST_RETURN, EFFECT_VOIDRAYPRISMATICALIGNMENT
from sc2.ids.unit_typeid import UnitTypeId
import random
import numpy as np
import time
import math
import keras

HEADLESS = False

class MadBot(sc2.BotAI):
    def __init__(self, use_model=True):
        self.combinedActions = []
        self.MAX_WORKERS = 44
        self.do_something_after = 0
        self.do_something_scout = 0
        self.do_something_after_exe = 0
        self.MAX_EXE = 2
        self.MAX_GATES = 3
        self.MAX_ROBOS = 1
        self.MAX_GAS = 4
        self.first_gas_taken = False
        self.early_game_finished = False
        self.warpgate_started = False
        self.lance_started = False
        self.blink_started = False
        self.charge_started = False
        self.first_attack = False
        self.first_attack_finished = False
        self.gathered = False
        self.second_attack = False
        self.final_attack = False
        self.second_gathered = False
        self.proxy_built = False
        self.first_pylon_built = False
        self.armor_upgrade = 0
        self.weapon_upgrade = 0

        self.scout = []
        self.remembered_enemy_units = []
        self.remembered_enemy_units_by_tag = {}
        self.units_to_ignore = [UnitTypeId.KD8CHARGE, UnitTypeId.EGG, UnitTypeId.LARVA, UnitTypeId.OVERLORD, UnitTypeId.BROODLING, UnitTypeId.INTERCEPTOR, UnitTypeId.CREEPTUMOR, UnitTypeId.CREEPTUMORBURROWED, UnitTypeId.CREEPTUMORQUEEN, UnitTypeId.CREEPTUMORMISSILE]
        self.units_to_ignore_defend = [UnitTypeId.KD8CHARGE, UnitTypeId.REAPER, UnitTypeId.BANELING, UnitTypeId.EGG, UnitTypeId.LARVA, UnitTypeId.OVERLORD, UnitTypeId.BROODLING,
                                UnitTypeId.INTERCEPTOR, UnitTypeId.CREEPTUMOR, UnitTypeId.CREEPTUMORBURROWED,
                                UnitTypeId.CREEPTUMORQUEEN, UnitTypeId.CREEPTUMORMISSILE]

        self.defend_around = [PYLON, NEXUS, ASSIMILATOR]
        self.threat_proximity = 11
        self.prg = []
        self.prg2 = []
        self.back_home_early = False
        self.defend_early = False
        self.back_home = False
        self.defend = False
        self.k = 0

        self.train_data = []
        self.scout_data = []
        self.build_order = []

        self.won = False

        self.model = keras.models.load_model("MadAI_09_02_2019")


        # Only run once at game start
    async def on_game_start(self):
        # self.build_order = random.randrange(0, 5)
        #
        # if self.build_order == 0:
        #     print('--- 2-Base Colossus BO chosen ---')
        # elif self.build_order == 3:
        #     print('--- 2-Base Adept/Immortal BO chosen ---')
        # elif self.build_order == 2:
        #     print('--- 4-Gate Proxy BO chosen ---')
        # elif self.build_order == 1:
        #     print('--- One-Base Defend into DT BO chosen ---')
        # elif self.build_order == 4:
        #     print('--- One-Base Defend into Voidrays BO chosen ---')
        # else:
        #     print('--- ???????? ---')

        for worker in self.workers:
            closest_mineral_patch = self.state.mineral_field.closest_to(worker)
            self.combinedActions.append(worker.gather(closest_mineral_patch))
            await self.do_actions(self.combinedActions)

        # Save base locations for later
        self.enemy_natural = await self.find_enemy_natural()
        # print('Enemy Natural @', self.enemy_natural)

    def on_end(self, game_result):
        print("OnGameEnd() was called.")
        # if str(game_result) == "Result.Victory":
        #     result = 1
        # else:
        #     result = 0
        # if self.early_game_finished:
        #    self.train_data.append([self.build_order, result, self.scout_data[0], self.scout_data[1], self.scout_data[2],
        #                            self.scout_data[3], self.scout_data[4], self.scout_data[5], self.scout_data[6], self.scout_data[7],
        #                            self.scout_data[8], self.scout_data[9], self.scout_data[10], self.scout_data[11],
        #                            self.scout_data[12], self.scout_data[13], self.scout_data[14], self.scout_data[15],
        #                            self.scout_data[16], self.scout_data[17], self.scout_data[18], self.scout_data[19],
        #                            self.scout_data[20], self.scout_data[21], self.scout_data[22], self.scout_data[23]])
        # np.save("data/{}_full.npy".format(str(int(time.time()))), np.array(self.train_data))

    async def on_step(self, iteration):

        self.combinedActions = []

        if iteration == 0:
            print("Game started")
            await self.on_game_start()

        #Early Game
        if not self.early_game_finished:
            #print('--- Early Game Started ---')
            await self.build_workers()
            await self.build_pylons()
            await self.build_assimilators()
            await self.build_first_gates()
            await self.first_gate_units()
            await self.chrono_boost()
            await self.defend_early_rush()
            await self.distribute_workers()
            await self.remember_enemy_units()
        #Mid Game
        elif self.early_game_finished and not self.first_attack_finished and self.build_order == 0:
            self.MAX_GATES = 7
            await self.build_workers()
            await self.build_pylons()
            await self.distribute_workers()
            await self.build_assimilators()
            await self.chrono_boost()
            await self.defend_early_rush()
            await self.expand()
            await self.scout_obs()
            await self.morph_warpgates()
            await self.micro_units()
            await self.build_proxy_pylon_2base_colossus()
            await self.two_base_colossus_buildings()
            await self.two_base_colossus_upgrade()
            await self.two_base_colossus_offensive_force()
            await self.two_base_colossus_unit_control()
            await self.remember_enemy_units()
            await self.game_won()
        elif self.early_game_finished and self.build_order == 3:
            self.MAX_WORKERS = 32
            self.MAX_GAS = 2
            await self.build_workers()
            await self.build_pylons()
            await self.distribute_workers()
            await self.build_assimilators()
            await self.chrono_boost()
            await self.defend_early_rush()
            await self.expand()
            await self.morph_warpgates()
            await self.micro_units()
            await self.immortal_adept_buildings()
            await self.immortal_adept_offensive_force()
            await self.immortal_adept_unit_control()
            await self.build_proxy_pylon()
            await self.remember_enemy_units()
            await self.game_won()
        elif self.early_game_finished and self.build_order == 2:
            self.MAX_WORKERS = 20
            self.MAX_GATES = 4
            await self.build_workers()
            await self.build_pylons()
            await self.distribute_workers()
            await self.build_assimilators()
            await self.chrono_boost()
            await self.defend_early_rush()
            await self.morph_warpgates()
            await self.micro_units()
            await self.build_proxy_pylon_four_gate()
            await self.four_gate_buildings()
            await self.four_gate_offensive_force()
            await self.four_gate_unit_control()
            await self.remember_enemy_units()
            await self.game_won()
        elif self.early_game_finished and self.build_order == 1:
            self.MAX_WORKERS = 20
            await self.build_workers()
            await self.build_pylons()
            await self.chrono_boost()
            await self.defend_early_rush()
            await self.morph_warpgates()
            await self.build_assimilators()
            await self.micro_units()
            await self.build_proxy_pylon_dt()
            await self.one_base_dt_buildings()
            await self.one_base_dt_offensive_force()
            await self.dt_unit_control()
            await self.distribute_workers()
            await self.remember_enemy_units()
            await self.game_won()
        elif self.early_game_finished and self.build_order == 4:
            self.MAX_WORKERS = 20
            self.MAX_GATES = 2
            await self.build_workers()
            await self.build_pylons()
            await self.chrono_boost()
            await self.defend_early_rush()
            await self.morph_warpgates()
            await self.build_assimilators()
            await self.micro_units()
            await self.one_base_vr_buildings()
            await self.one_base_vr_offensive_force()
            await self.vr_unit_control()
            await self.distribute_workers()
            await self.remember_enemy_units()
            await self.game_won()
        # Late Game
        elif self.first_attack_finished and self.build_order == 0:
            self.MAX_WORKERS = 50
            self.MAX_EXE = 3  # Increase Exes to 3 TODO: and build a new one every ~3 Minutes
            self.MAX_GATES = 8
            await self.build_workers()
            await self.build_pylons()
            await self.distribute_workers()
            await self.build_assimilators()
            await self.chrono_boost()
            await self.expand()
            await self.scout_obs()
            await self.morph_warpgates()
            await self.micro_units()
            await self.game_won()
            if len(self.units(NEXUS)) == 3:
                await self.two_base_colossus_offensive_force()
                await self.two_base_colossus_unit_control_lategame()
                await self.two_base_colossus_upgrade_lategame()
        # Destroy Terran BM after 30min
        if self.time/60 >= 20:
            self.build_order = 99
            self.MAX_EXE = 4
            self.MAX_GAS = 6
            self.MAX_WORKERS = 50
            await self.distribute_workers()
            await self.build_pylons()
            await self.expand()
            await self.build_assimilators()
            await self.destroy_lifted_buildings()

    async def game_won(self):
        if not self.won and len(self.known_enemy_units(NEXUS)) + len(self.known_enemy_units(UnitTypeId.COMMANDCENTER)) + len(self.known_enemy_units(UnitTypeId.ORBITALCOMMAND)) + len(self.known_enemy_units(UnitTypeId.HATCHERY)) + len(self.known_enemy_units(UnitTypeId.LAIR)) + len(self.known_enemy_units(UnitTypeId.HIVE)) == 0:
            self.won = True
            self.train_data.append(
                [self.build_order, self.scout_data[0], self.scout_data[1], self.scout_data[2],
                 self.scout_data[3], self.scout_data[4], self.scout_data[5], self.scout_data[6], self.scout_data[7],
                 self.scout_data[8], self.scout_data[9], self.scout_data[10], self.scout_data[11],
                 self.scout_data[12], self.scout_data[13], self.scout_data[14], self.scout_data[15],
                 self.scout_data[16], self.scout_data[17], self.scout_data[18], self.scout_data[19],
                 self.scout_data[20], self.scout_data[21], self.scout_data[22], self.scout_data[23],
                 self.scout_data[24], self.scout_data[25], self.scout_data[26], #])
                 self.scout_data[27], self.scout_data[28], self.scout_data[29],self.scout_data[30],
                 self.scout_data[31], self.scout_data[32], self.scout_data[33], self.scout_data[34],
                 self.scout_data[35], self.scout_data[36], self.scout_data[37], self.scout_data[38],
                 self.scout_data[39], self.scout_data[40], self.scout_data[41], self.scout_data[42],
                 self.scout_data[43], self.scout_data[44], self.scout_data[45], self.scout_data[46],
                 self.scout_data[47], self.scout_data[48], self.scout_data[49], self.scout_data[50],
                 self.scout_data[51], self.scout_data[52], self.scout_data[53]])

            np.save("data/{}.npy".format(str(int(time.time()))), np.array(self.train_data))

    async def build_workers(self):
        if (len(self.units(NEXUS)) * 22) > len(self.units(PROBE)) and len(self.units(PROBE)) < self.MAX_WORKERS:
            for nexus in self.units(NEXUS).ready.noqueue:
                if self.can_afford(PROBE):
                    await self.do(nexus.train(PROBE))

    async def build_pylons(self):
        if self.supply_used <= 14 and self.can_afford(PYLON) and not self.first_pylon_built:
            pylon_placement_positions = self.main_base_ramp.corner_depots
            nexus = self.units(NEXUS)
            pylon_placement_positions = {d for d in pylon_placement_positions if nexus.closest_distance_to(d) > 1}
            target_pylon_location = pylon_placement_positions.pop()
            await self.build(PYLON, near=target_pylon_location)
            # Say GL HF!

            self.first_pylon_built = True
        elif 17 < self.supply_used <= 20 and not self.already_pending(PYLON) and self.already_pending(GATEWAY) and self.supply_left < 6:
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=self.units(NEXUS).first.position.towards(self.game_info.map_center, 5))

        elif self.units(NEXUS).amount > 1 and self.can_afford(PYLON) and not self.already_pending(PYLON) and self.supply_used < 36 and self.supply_left < 12:
            await self.build(PYLON, near=self.units(NEXUS)[1].position.towards(self.game_info.map_center, 5))
        elif 20 < self.supply_used < 61 and self.supply_left < 8 and not self.already_pending(PYLON):
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=nexuses.random.position.random_on_distance(random.randrange(1, 15)), max_distance=10, random_alternative=False, placement_step=2)
        elif 60 < self.supply_used < 188 and self.supply_left < 16 and not self.already_pending(PYLON):
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=nexuses.random.position.towards(self.game_info.map_center, random.randrange(1, 20)), max_distance=10, random_alternative=False, placement_step=2)

    async def build_assimilators(self):
        if len(self.units(ASSIMILATOR)) < self.MAX_GAS:
            for nexus in self.units(NEXUS).ready:
                vaspenes = self.state.vespene_geyser.closer_than(15.0, nexus)
                for vaspene in vaspenes:
                    if not self.can_afford(ASSIMILATOR):
                        break
                    worker = self.select_build_worker(vaspene.position)
                    if worker is None:
                        break
                    if not self.units(ASSIMILATOR).closer_than(1.0, vaspene).exists:
                        if self.supply_used >= 17 and not self.first_gas_taken:
                            await self.do(worker.build(ASSIMILATOR, vaspene))

                            self.first_gas_taken = True
                            break
                        elif self.supply_used >= 25 and self.first_gas_taken:
                            await self.do(worker.build(ASSIMILATOR, vaspene))

                            break
                        elif self.supply_used >= 35:
                            await self.do(worker.build(ASSIMILATOR, vaspene))
        #if self.units(ASSIMILATOR).ready.exists: # Sofort 3 Arbeiter zuweisen wenn Assimilator fertig

    async def build_first_gates(self):
        if self.units(PYLON).ready.exists:
            if self.can_afford(GATEWAY) and self.supply_used >= 16 and len(self.units(GATEWAY)) < 1 and not self.already_pending(GATEWAY):
                await self.build(GATEWAY, near=self.main_base_ramp.barracks_correct_placement)
                # # Use Gateway Probe as Scout
                # self.scout = self.units(PROBE).furthest_to(self.units(NEXUS).first)
                # move_to = self.enemy_start_locations[0].random_on_distance(random.randrange(1, 3))
                # print(move_to)
                # await self.do(self.scout.move(move_to))

            if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE):
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=self.units(PYLON).ready.closest_to(self.units(NEXUS).first), max_distance=10, random_alternative=False, placement_step=5)


            if self.units(CYBERNETICSCORE).ready.exists and self.can_afford(
                    RESEARCH_WARPGATE) and not self.warpgate_started:
                ccore = self.units(CYBERNETICSCORE).ready.first
                await self.do(ccore(RESEARCH_WARPGATE))
                self.warpgate_started = True


    async def first_gate_units(self):
        if self.units(GATEWAY).ready.exists:
            for gw in self.units(GATEWAY).ready.noqueue:
                if self.can_afford(ZEALOT) and self.supply_used <= 22:
                    await self.do(gw.train(ZEALOT))
                elif self.can_afford(ZEALOT) and self.supply_left > 1 and not self.units(CYBERNETICSCORE).ready.exists and (len(self.known_enemy_units(UnitTypeId.HATCHERY)) >= 1 or len(self.remembered_enemy_units.of_type(UnitTypeId.ZERGLING)) > 1):
                    await self.do(gw.train(ZEALOT))
                elif self.minerals > 525 and self.supply_left > 1:
                    await self.do(gw.train(ZEALOT))
                if self.units(CYBERNETICSCORE).ready.exists and self.can_afford(STALKER) and self.supply_left > 0:
                    if len(self.known_enemy_units(UnitTypeId.HATCHERY)) >= 1 or len(self.remembered_enemy_units.of_type(UnitTypeId.ZERGLING)) > 1:
                        await self.do(gw.train(ZEALOT))
                    else:
                        await self.do(gw.train(STALKER))
                    print('--- Early Game Finished --- @:', self.time)
                    workercount = len(self.remembered_enemy_units.of_type({UnitTypeId.DRONE, UnitTypeId.PROBE, UnitTypeId.SCV}))

                    enemy_pylon_pos = []
                    for pylon in range(len(self.known_enemy_units(PYLON))):
                        enemy_pylon_pos.append(self.known_enemy_units(PYLON)[pylon].position)
                    enemy_gateway_pos = []
                    for gateway in range(len(self.known_enemy_units(GATEWAY))):
                        enemy_gateway_pos.append(self.known_enemy_units(GATEWAY)[gateway].position)
                    enemy_forge_pos = []
                    for forge in range(len(self.known_enemy_units(FORGE))):
                        enemy_forge_pos.append(self.known_enemy_units(FORGE)[forge].position)
                    enemy_cannon_pos = []
                    for cannon in range(len(self.known_enemy_units(PHOTONCANNON))):
                        enemy_cannon_pos.append(self.known_enemy_units(PHOTONCANNON)[cannon].position)
                    enemy_depot_pos = []
                    for depot in range(len(self.known_enemy_units(UnitTypeId.SUPPLYDEPOT))):
                        enemy_depot_pos.append(self.known_enemy_units(UnitTypeId.SUPPLYDEPOT)[depot].position)
                    enemy_depotlow_pos = []
                    for depotlow in range(len(self.known_enemy_units(UnitTypeId.SUPPLYDEPOTLOWERED))):
                        enemy_depotlow_pos.append(self.known_enemy_units(UnitTypeId.SUPPLYDEPOTLOWERED)[depotlow].position)
                    enemy_bunker_pos = []
                    for bunker in range(len(self.known_enemy_units(UnitTypeId.BUNKER))):
                        enemy_bunker_pos.append(self.known_enemy_units(UnitTypeId.BUNKER)[bunker].position)
                    enemy_barracks_pos = []
                    for barracks in range(len(self.known_enemy_units(UnitTypeId.BARRACKS))):
                        enemy_barracks_pos.append(self.known_enemy_units(UnitTypeId.BARRACKS)[barracks].position)
                    enemy_factory_pos = []
                    for factory in range(len(self.known_enemy_units(UnitTypeId.FACTORY))):
                        enemy_factory_pos.append(self.known_enemy_units(UnitTypeId.FACTORY)[factory].position)
                    enemy_pool_pos = []
                    for pool in range(len(self.known_enemy_units(UnitTypeId.SPAWNINGPOOL))):
                        enemy_pool_pos.append(self.known_enemy_units(UnitTypeId.SPAWNINGPOOL)[pool].position)
                    enemy_spine_pos = []
                    for spine in range(len(self.known_enemy_units(UnitTypeId.SPINECRAWLER))):
                        enemy_spine_pos.append(self.known_enemy_units(UnitTypeId.SPINECRAWLER)[spine].position)

                    if len(self.known_enemy_units(PYLON)) >= 1:
                        pylon1_pos = enemy_pylon_pos[0][0] + enemy_pylon_pos[0][1]
                    else:
                        pylon1_pos = 0
                    if len(self.known_enemy_units(PYLON)) >= 2:
                        pylon2_pos = enemy_pylon_pos[1][0] + enemy_pylon_pos[1][1]
                    else:
                        pylon2_pos = 0
                    if len(self.known_enemy_units(PYLON)) >= 3:
                        pylon3_pos = enemy_pylon_pos[2][0] + enemy_pylon_pos[2][1]
                    else:
                        pylon3_pos = 0
                    if len(self.known_enemy_units(GATEWAY)) >= 1:
                        gate1_pos = enemy_gateway_pos[0][0] + enemy_gateway_pos[0][1]
                    else:
                        gate1_pos = 0
                    if len(self.known_enemy_units(GATEWAY)) >= 2:
                        gate2_pos = enemy_gateway_pos[1][0] + enemy_gateway_pos[1][1]
                    else:
                        gate2_pos = 0
                    if len(self.known_enemy_units(FORGE)) >= 1:
                        forge1_pos = enemy_forge_pos[0][0] + enemy_forge_pos[0][1]
                    else:
                        forge1_pos = 0
                    if len(self.known_enemy_units(PHOTONCANNON)) >= 1:
                        cannon1_pos = enemy_cannon_pos[0][0] + enemy_cannon_pos[0][1]
                    else:
                        cannon1_pos = 0
                    if len(self.known_enemy_units(PHOTONCANNON)) >= 2:
                        cannon2_pos = enemy_cannon_pos[1][0] + enemy_cannon_pos[1][1]
                    else:
                        cannon2_pos = 0
                    if len(self.known_enemy_units(PHOTONCANNON)) >= 3:
                        cannon3_pos = enemy_cannon_pos[2][0] + enemy_cannon_pos[2][1]
                    else:
                        cannon3_pos = 0
                    if len(self.known_enemy_units(PHOTONCANNON)) >= 4:
                        cannon4_pos = enemy_cannon_pos[3][0] + enemy_cannon_pos[3][1]
                    else:
                        cannon4_pos = 0
                    if len(self.known_enemy_units(UnitTypeId.SUPPLYDEPOT)) >= 1:
                        depot1_pos = enemy_depot_pos[0][0] + enemy_depot_pos[0][1]
                    else:
                        depot1_pos = 0
                    if len(self.known_enemy_units(UnitTypeId.SUPPLYDEPOT)) >= 2:
                        depot2_pos = enemy_depot_pos[1][0] + enemy_depot_pos[1][1]
                    else:
                        depot2_pos = 0
                    if len(self.known_enemy_units(UnitTypeId.SUPPLYDEPOT)) >= 3:
                        depot3_pos = enemy_depot_pos[2][0] + enemy_depot_pos[2][1]
                    else:
                        depot3_pos = 0
                    if len(self.known_enemy_units(UnitTypeId.SUPPLYDEPOTLOWERED)) >= 1:
                        depotlow1_pos = enemy_depotlow_pos[0][0] + enemy_depotlow_pos[0][1]
                    else:
                        depotlow1_pos = 0
                    if len(self.known_enemy_units(UnitTypeId.SUPPLYDEPOTLOWERED)) >= 2:
                        depotlow2_pos = enemy_depotlow_pos[1][0] + enemy_depotlow_pos[1][1]
                    else:
                        depotlow2_pos = 0
                    if len(self.known_enemy_units(UnitTypeId.SUPPLYDEPOTLOWERED)) >= 3:
                        depotlow3_pos = enemy_depotlow_pos[2][0] + enemy_depotlow_pos[2][1]
                    else:
                        depotlow3_pos = 0
                    if len(self.known_enemy_units(UnitTypeId.BUNKER)) >= 1:
                        bunker1_pos = enemy_bunker_pos[0][0] + enemy_bunker_pos[0][1]
                    else:
                        bunker1_pos = 0
                    if len(self.known_enemy_units(UnitTypeId.BARRACKS)) >= 1:
                        barracks1_pos = enemy_barracks_pos[0][0] + enemy_barracks_pos[0][1]
                    else:
                        barracks1_pos = 0
                    if len(self.known_enemy_units(UnitTypeId.BARRACKS)) >= 2:
                        barracks2_pos = enemy_barracks_pos[1][0] + enemy_barracks_pos[1][1]
                    else:
                        barracks2_pos = 0
                    if len(self.known_enemy_units(UnitTypeId.BARRACKS)) >= 3:
                        barracks3_pos = enemy_barracks_pos[2][0] + enemy_barracks_pos[2][1]
                    else:
                        barracks3_pos = 0
                    if len(self.known_enemy_units(UnitTypeId.FACTORY)) >= 1:
                        factory1_pos = enemy_factory_pos[0][0] + enemy_factory_pos[0][1]
                    else:
                        factory1_pos = 0
                    if len(self.known_enemy_units(UnitTypeId.SPAWNINGPOOL)) >= 1:
                        pool1_pos = enemy_pool_pos[0][0] + enemy_pool_pos[0][1]
                    else:
                        pool1_pos = 0
                    if len(self.known_enemy_units(UnitTypeId.SPINECRAWLER)) >= 1:
                        spine1_pos = enemy_spine_pos[0][0] + enemy_spine_pos[0][1]
                    else:
                        spine1_pos = 0
                    if len(self.known_enemy_units(UnitTypeId.SPINECRAWLER)) >= 2:
                        spine2_pos = enemy_spine_pos[1][0] + enemy_spine_pos[1][1]
                    else:
                        spine2_pos = 0

                    self.scout_data = [len(self.known_enemy_units(NEXUS)),
                                       len(self.known_enemy_units(PYLON)),
                                       len(self.known_enemy_units(GATEWAY)),
                                       len(self.known_enemy_units(CYBERNETICSCORE)),
                                       len(self.known_enemy_units(ASSIMILATOR)),
                                       len(self.known_enemy_units(UnitTypeId.COMMANDCENTER)),
                                       len(self.known_enemy_units(UnitTypeId.ORBITALCOMMAND)),
                                       len(self.known_enemy_units(UnitTypeId.SUPPLYDEPOT)),
                                       len(self.known_enemy_units(UnitTypeId.SUPPLYDEPOTLOWERED)),
                                       len(self.known_enemy_units(UnitTypeId.BARRACKS)),
                                       len(self.known_enemy_units(UnitTypeId.TECHLAB)),
                                       len(self.known_enemy_units(UnitTypeId.REACTOR)),
                                       len(self.known_enemy_units(UnitTypeId.REFINERY)),
                                       len(self.known_enemy_units(UnitTypeId.FACTORY)),
                                       len(self.known_enemy_units(UnitTypeId.HATCHERY)),
                                       len(self.known_enemy_units(UnitTypeId.SPINECRAWLER)),
                                       len(self.known_enemy_units(UnitTypeId.SPAWNINGPOOL)),
                                       len(self.known_enemy_units(UnitTypeId.ROACHWARREN)),
                                       len(self.known_enemy_units(UnitTypeId.EXTRACTOR)),
                                       workercount,
                                       len(self.remembered_enemy_units.of_type(UnitTypeId.ZEALOT)),
                                       len(self.remembered_enemy_units.of_type(UnitTypeId.STALKER)),
                                       len(self.remembered_enemy_units.of_type(UnitTypeId.MARINE)),
                                       len(self.remembered_enemy_units.of_type(UnitTypeId.REAPER)),
                                       len(self.remembered_enemy_units.of_type(UnitTypeId.ZERGLING)),
                                       len(self.remembered_enemy_units.of_type(UnitTypeId.ROACH)),
                                       len(self.known_enemy_units(UnitTypeId.PHOTONCANNON)),
                                       len(self.known_enemy_units(UnitTypeId.BUNKER)),
                                       len(self.known_enemy_units(FORGE)),
                                       self.enemy_start_locations[0][0] + self.enemy_start_locations[0][1],
                                       pylon1_pos, pylon2_pos, pylon3_pos,
                                       gate1_pos, gate2_pos,
                                       forge1_pos, cannon1_pos, cannon2_pos, cannon3_pos, cannon4_pos,
                                       depot1_pos, depot2_pos, depot3_pos, depotlow1_pos, depotlow2_pos, depotlow3_pos,
                                       bunker1_pos,
                                       barracks1_pos, barracks2_pos, barracks3_pos,
                                       factory1_pos,
                                       pool1_pos, spine1_pos, spine2_pos]

                    # print(self.scout_data)
                    self.early_game_finished = True

                    choice_data = [self.scout_data[0], self.scout_data[1], self.scout_data[2], self.scout_data[3],
                                   self.scout_data[4], self.scout_data[5], self.scout_data[6], self.scout_data[7],
                                   self.scout_data[8], self.scout_data[9], self.scout_data[10], self.scout_data[11],
                                   self.scout_data[12], self.scout_data[13], self.scout_data[14], self.scout_data[15],
                                   self.scout_data[16], self.scout_data[17], self.scout_data[18], self.scout_data[19],
                                   self.scout_data[20], self.scout_data[21], self.scout_data[22], self.scout_data[23],
                                   self.scout_data[24], self.scout_data[25], self.scout_data[26],
                                   self.scout_data[27], self.scout_data[28], self.scout_data[29], self.scout_data[30],
                                   self.scout_data[31], self.scout_data[32], self.scout_data[33], self.scout_data[34],
                                   self.scout_data[35], self.scout_data[36], self.scout_data[37], self.scout_data[38],
                                   self.scout_data[39], self.scout_data[40], self.scout_data[41], self.scout_data[42],
                                   self.scout_data[43], self.scout_data[44], self.scout_data[45], self.scout_data[46],
                                   self.scout_data[47], self.scout_data[48], self.scout_data[49], self.scout_data[50],
                                   self.scout_data[51], self.scout_data[52], self.scout_data[53]]

                    new_choice_data = np.array(choice_data).reshape(-1, 54, 1)

                    prediction = self.model.predict(new_choice_data)
                    choice = np.argmax(prediction[0])
                    certainty = prediction[0][choice]
                    print(prediction[0])

                    self.build_order = choice

                    if self.build_order == 0:
                        print('--- 2-Base Colossus BO chosen ---')
                        await self.chat_send("(glhf) MadBot v1.8: 2-Base Colossus BO chosen! Certainties: 2-Base Colossus: " + str(round(prediction[0][0]*100, 2)) + "%; " +
                                             "1-Base DTs: " + str(round(prediction[0][1]*100, 2)) + "%; " +
                                             "4-Gate Proxy: " + str(round(prediction[0][2] * 100, 2)) + "%; " +
                                             "2-Base Immortals: " + str(round(prediction[0][3] * 100, 2)) + "%; " +
                                             "1-Base Voidrays: " + str(round(prediction[0][4] * 100, 2)) + "%")
                    elif self.build_order == 1:
                        print('--- One-Base Defend into DT BO chosen ---')
                        await self.chat_send("(glhf) MadBot v1.8: Rush-Defend into DT BO chosen! Certainties: 2-Base Colossus: " + str(round(prediction[0][0]*100, 2)) + "%; " +
                                             "1-Base DTs: " + str(round(prediction[0][1]*100, 2)) + "%; " +
                                             "4-Gate Proxy: " + str(round(prediction[0][2] * 100, 2)) + "%; " +
                                             "2-Base Immortals: " + str(round(prediction[0][3] * 100, 2)) + "%; " +
                                             "1-Base Voidrays: " + str(round(prediction[0][4] * 100, 2)) + "%")
                    elif self.build_order == 2:
                        print('--- 4-Gate Proxy BO chosen ---')
                        await self.chat_send("(glhf) MadBot v1.8: 4-Gate Proxy BO chosen! Certainties: 2-Base Colossus: " + str(round(prediction[0][0]*100, 2)) + "%; " +
                                             "1-Base DTs: " + str(round(prediction[0][1]*100, 2)) + "%; " +
                                             "4-Gate Proxy: " + str(round(prediction[0][2] * 100, 2)) + "%; " +
                                             "2-Base Immortals: " + str(round(prediction[0][3] * 100, 2)) + "%; " +
                                             "1-Base Voidrays: " + str(round(prediction[0][4] * 100, 2)) + "%")
                    elif self.build_order == 3:
                        print('--- 2-Base Adept/Immortal BO chosen ---')
                        await self.chat_send("(glhf) MadBot v1.8: 2-Base Adept/Immortal BO chosen! Certainties: 2-Base Colossus: " + str(round(prediction[0][0]*100, 2)) + "%; " +
                                             "1-Base DTs: " + str(round(prediction[0][1]*100, 2)) + "%; " +
                                             "4-Gate Proxy: " + str(round(prediction[0][2] * 100, 2)) + "%; " +
                                             "2-Base Immortals: " + str(round(prediction[0][3] * 100, 2)) + "%; " +
                                             "1-Base Voidrays: " + str(round(prediction[0][4] * 100, 2)) + "%")
                    elif self.build_order == 4:
                        print('--- One-Base Defend into Voidrays BO chosen ---')
                        await self.chat_send("(glhf) MadBot v1.8: Rush-Defend into Voidrays BO chosen! Certainties: 2-Base Colossus: " + str(round(prediction[0][0]*100, 2)) + "%; " +
                                             "1-Base DTs: " + str(round(prediction[0][1]*100, 2)) + "%; " +
                                             "4-Gate Proxy: " + str(round(prediction[0][2] * 100, 2)) + "%; " +
                                             "2-Base Immortals: " + str(round(prediction[0][3] * 100, 2)) + "%; " +
                                             "1-Base Voidrays: " + str(round(prediction[0][4] * 100, 2)) + "%")
                    else:
                        self.build_order = random.randrange(0, 5)
                        await self.chat_send("(glhf) MadBot v1.8: Neural Network broke, choosing random Build Order")

    async def defend_early_rush(self):
        # defend if there is a 12 pool or worker rush
        if len(self.units(ZEALOT)) < 2:
            for zl in self.units(ZEALOT).idle:
                self.combinedActions.append(zl.attack(self.units(NEXUS).closest_to(self.game_info.map_center).position.towards(self.game_info.map_center, random.randrange(8, 10))))

        if (len(self.units(ZEALOT)) + len(self.units(STALKER)) + len(self.units(ADEPT)) < 2 and self.known_enemy_units) or self.back_home_early:
            threats = []
            for structure_type in self.defend_around:
                for structure in self.units(structure_type):
                    threats += self.known_enemy_units.filter(lambda unit: unit.type_id not in self.units_to_ignore_defend).closer_than(11, structure.position)
                    # print(threats)
                    if len(threats) > 0:
                        break
                if len(threats) > 0:
                    break
            #print(len(threats))
            # Full rush incoming. Pull all probes
            if len(threats) >= 7:
                #print('Full')
                self.defend_early = True
                self.back_home_early = True
                defence_target = threats[0].position.random_on_distance(random.randrange(1, 3))
                for pr in self.units(PROBE):
                    if pr.shield_percentage > 0.1:
                        self.combinedActions.append(pr.attack(defence_target))
                    elif self.units(NEXUS).exists:
                        self.combinedActions.append(pr.gather(self.state.mineral_field.closest_to(self.units(NEXUS).first)))
                for zl in self.units(ZEALOT):
                    self.combinedActions.append(zl.attack(defence_target))
            # 12 Pool or some kind of stuff
            elif 1 < len(threats) < 7 and len(self.prg) == 0:
                #print('Half')
                self.defend_early = True
                self.back_home_early = True
                defence_target = threats[0].position.random_on_distance(random.randrange(1, 3))
                self.prg = self.units(PROBE).random_group_of(round(len(self.units(PROBE))/2))
                for pr in self.prg:
                    if pr.shield_percentage > 0.1:
                        self.combinedActions.append(pr.attack(defence_target))
                    elif self.units(NEXUS).exists:
                        self.combinedActions.append(pr.gather(self.state.mineral_field.closest_to(self.units(NEXUS).first)))
                for zl in self.units(ZEALOT):
                    self.combinedActions.append(zl.attack(defence_target))
            # Just some harass. Pull only two probes
            elif len(threats) == 1 and not self.defend_early and len(self.prg2) == 0 and len(self.units(PROBE)) > 1:
                #print('Two')
                self.defend_early = True
                self.back_home_early = True
                defence_target = threats[0].position.random_on_distance(random.randrange(1, 3))
                self.prg2 = self.units(PROBE).random_group_of(2)
                for pr in self.prg2:
                    self.combinedActions.append(pr.attack(defence_target))
                for zl in self.units(ZEALOT):
                    self.combinedActions.append(zl.attack(defence_target))
            # Threat is gone for now. Go back to work
            elif (len(self.prg) > 0 or len(self.prg2) > 0) and len(threats) == 0 and self.back_home_early:
                #print('Back1')
                if self.units(NEXUS).exists:
                    if len(self.prg) > 0:
                        for pr in self.prg:
                            self.combinedActions.append(pr.gather(self.state.mineral_field.closest_to(self.units(NEXUS).first)))
                        for zl in self.units(ZEALOT):
                            self.combinedActions.append(zl.move(
                                self.units(NEXUS).closest_to(self.game_info.map_center).position.towards(
                                    self.game_info.map_center, random.randrange(8, 10))))
                        self.prg = []
                    elif len(self.prg2) > 0:
                        for pr in self.prg2:
                            self.combinedActions.append(pr.gather(self.state.mineral_field.closest_to(self.units(NEXUS).first)))
                        for zl in self.units(ZEALOT):
                            self.combinedActions.append(zl.move(
                                self.units(NEXUS).closest_to(self.game_info.map_center).position.towards(
                                    self.game_info.map_center, random.randrange(8, 10))))
                        self.prg2 = []

                self.defend_early = False
                self.back_home_early = False
            # Everything is fine again. Go back to work
            elif len(threats) == 0 and self.back_home_early:
                #print('Back2')
                self.back_home_early = False
                self.defend_early = False
                if self.units(NEXUS).exists:
                    for pr in self.units(PROBE):
                        self.combinedActions.append(pr.gather(self.state.mineral_field.closest_to(self.units(NEXUS).first)))
                    for zl in self.units(ZEALOT):
                        self.combinedActions.append(zl.move(
                            self.units(NEXUS).closest_to(self.game_info.map_center).position.towards(
                                self.game_info.map_center, random.randrange(8, 10))))

            # Some Cheese detected (e.g. YoBot & NaugthyBot). Pull some Probes!
            # if self.known_enemy_structures.closer_than(100, self.units(NEXUS).first) and len(self.prg2) == 0:
            #     self.prg2 = self.units(PROBE).random_group_of(4)
            #     for pr2 in self.prg2:
            #         # if len(self.known_enemy_units.of_type({PYLON, UnitTypeId.SCV})) > 0:
            #         #     self.combinedActions.append(pr2.attack(self.known_enemy_units.of_type(
            #         #         {PYLON, UnitTypeId.SCV}).closest_to(pr2.position)))
            #         #     print('Attacking Drone')
            #         # else:
            #         self.combinedActions.append(pr2.attack(
            #                 self.known_enemy_structures.closest_to(self.units(NEXUS).first).position.random_on_distance(
            #                     random.randrange(1, 3))))
            #         print('Attacking Base')
            # elif len(self.prg2) > 0 and len(self.known_enemy_structures.closer_than(120, self.units(NEXUS).first)) == 0:
            #     for pr2 in self.prg2:
            #         self.combinedActions.append(pr2.gather(self.state.vespene_geyser.closest_to(self.units(NEXUS).first)))
            #     self.prg2 = []

        await self.do_actions(self.combinedActions)

    async def remember_enemy_units(self):

        if self.first_pylon_built and self.units(PYLON).exists and self.scout == []:
            self.scout = [self.units(PROBE).furthest_to(self.units(NEXUS).first)]
            # print(self.scout)
            # print(self.enemy_start_locations)
        elif len(self.scout) == 1 and len(self.enemy_start_locations) == 1 and not (self.first_attack or self.proxy_built):
            if self.time > self.do_something_scout:
                wait = 500
                self.do_something_scout = self.time + wait
                for scout in self.scout:
                    # print('Sending Scout')
                    move_to1 = self.enemy_start_locations[0].random_on_distance(random.randrange(1, 5))
                    move_to2 = self.enemy_natural.random_on_distance(random.randrange(1, 5))
                    move_to3 = self.enemy_start_locations[0].random_on_distance(random.randrange(5, 10))
                    move_to4 = self.enemy_natural.random_on_distance(random.randrange(5, 10))
                    move_to5 = self.enemy_start_locations[0].towards(self.game_info.map_center, random.randrange(-10, -1))
                    move_to6 = self.enemy_natural.towards(self.game_info.map_center, random.randrange(-10, -1))
                    move_to7 = self.enemy_start_locations[0].random_on_distance(random.randrange(1, 15))
                    move_to8 = self.enemy_natural.random_on_distance(random.randrange(1, 15))
                    move_to9 = self.enemy_start_locations[0].random_on_distance(random.randrange(10, 20))
                    move_to10 = self.enemy_natural.random_on_distance(random.randrange(10, 20))
                    self.combinedActions.append(scout.move(move_to1))
                    self.combinedActions.append(scout.move(move_to2, queue=True))
                    self.combinedActions.append(scout.move(move_to3, queue=True))
                    self.combinedActions.append(scout.move(move_to4, queue=True))
                    self.combinedActions.append(scout.move(move_to5, queue=True))
                    self.combinedActions.append(scout.move(move_to6, queue=True))
                    self.combinedActions.append(scout.move(move_to7, queue=True))
                    self.combinedActions.append(scout.move(move_to8, queue=True))
                    self.combinedActions.append(scout.move(move_to9, queue=True))
                    self.combinedActions.append(scout.move(move_to10, queue=True))

        elif len(self.scout) == 1 and len(self.enemy_start_locations) > 1 and not (self.first_attack or self.proxy_built):
            if self.time > self.do_something_after:
                'TODO: Far from perfect. Needs more work!'
                self.k = self.k - 1
                pos = [0, 2, 1]
                wait = 50
                self.do_something_after = self.time + wait
                if self.k >= 0:
                    move_to = self.enemy_start_locations[pos[self.k]]
                    # print(move_to)
                    for scout in self.scout:
                        self.combinedActions.append(scout.move(move_to))
                # else:
                #     move_to = random.sample(list(self.enemy_start_locations), k=1)[0]
                #     print('2')


        # Look through all currently seen units and add them to list of remembered units (override existing)
        for unit in self.known_enemy_units:
            unit.is_known_this_step = True
            self.remembered_enemy_units_by_tag[unit.tag] = unit

        # Convert to an sc2 Units object and place it in self.remembered_enemy_units
        self.remembered_enemy_units = sc2.units.Units([], self._game_data)
        for tag, unit in list(self.remembered_enemy_units_by_tag.items()):
            # Make unit.is_seen = unit.is_visible
            if unit.is_known_this_step:
                unit.is_seen = unit.is_visible # There are known structures that are not visible
                unit.is_known_this_step = False # Set to false for next step
            else:
                unit.is_seen = False

            # Units that are not visible while we have friendly units nearby likely don't exist anymore, so delete them
            if not unit.is_seen and self.units.closer_than(7, unit).exists:
                del self.remembered_enemy_units_by_tag[tag]
                continue

            self.remembered_enemy_units.append(unit)

        await self.do_actions(self.combinedActions)

    async def expand(self):
        if self.units(NEXUS).exists and self.units(NEXUS).amount < self.MAX_EXE and self.can_afford(NEXUS) and self.time > self.do_something_after_exe:
            self.do_something_after_exe = self.time + 20
            location = await self.get_next_expansion()

            await self.build(NEXUS, near=location, max_distance=10, random_alternative=False,
                             placement_step=1)
            #await self.expand_now()

    async def scout_obs(self):
        if len(self.units(OBSERVER)) == 1:
            obs = self.units(OBSERVER)[0]
            if (self.first_attack or self.gathered) and self.units(COLOSSUS).ready.exists:
                target = self.units(COLOSSUS).ready.closest_to(self.enemy_start_locations[0]).position.towards(self.enemy_start_locations[0], random.randrange(5, 7))
            elif self.units(NEXUS).exists and self.units(STALKER).exists:
                target = self.units(STALKER).random
            else:
                target = self.game_info.map_center
            self.combinedActions.append(obs.move(target))

        elif len(self.units(OBSERVER)) == 2:
            scout = self.units(OBSERVER)[1]
            if scout.is_idle:
                #move_to = self.enemy_start_locations[0].towards(self.game_info.map_center, random.randrange(1, 20))
                move_to = self.enemy_start_locations[0].random_on_distance(random.randrange(20, 40))
                self.combinedActions.append(scout.move(move_to))

        if len(self.units(OBSERVER)) < 2 and not self.lance_started:
            for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
                if self.can_afford(OBSERVER) and self.supply_left > 0:
                    await self.do(rf.train(OBSERVER))

    async def chrono_boost(self):
        if self.units(NEXUS).ready.exists:
            nexus = self.units(NEXUS).ready.random
            if not self.units(GATEWAY).ready.exists and not self.units(WARPGATE).ready.exists:
                if not nexus.has_buff(CHRONOBOOSTENERGYCOST) and self.supply_used > 14:
                    abilities = await self.get_available_abilities(nexus)
                    if EFFECT_CHRONOBOOSTENERGYCOST in abilities:
                        await self.do(nexus(EFFECT_CHRONOBOOSTENERGYCOST, nexus))
            elif self.charge_started > 0 and self.time - self.charge_started <= 100:
                twi = self.units(TWILIGHTCOUNCIL).ready.first
                if not twi.has_buff(CHRONOBOOSTENERGYCOST):
                    abilities = await self.get_available_abilities(nexus)
                    if EFFECT_CHRONOBOOSTENERGYCOST in abilities:
                        await self.do(nexus(EFFECT_CHRONOBOOSTENERGYCOST, twi))
            elif not self.units(CYBERNETICSCORE).ready.exists and self.units(GATEWAY).ready.exists:
                gate = self.units(GATEWAY).ready.first
                if not nexus.has_buff(CHRONOBOOSTENERGYCOST) and not self.units(GATEWAY).ready.noqueue:
                    abilities = await self.get_available_abilities(nexus)
                    if EFFECT_CHRONOBOOSTENERGYCOST in abilities:
                        await self.do(nexus(EFFECT_CHRONOBOOSTENERGYCOST, gate))
            elif self.units(WARPGATE).ready.exists and not self.units(ROBOTICSFACILITY).ready.exists and not self.units(STARGATE).ready.exists and not self.units(GATEWAY).ready.exists:
                warpgate = self.units(WARPGATE).ready.random
                if not warpgate.has_buff(CHRONOBOOSTENERGYCOST):
                    abilities = await self.get_available_abilities(nexus)
                    if EFFECT_CHRONOBOOSTENERGYCOST in abilities:
                        await self.do(nexus(EFFECT_CHRONOBOOSTENERGYCOST, warpgate))
            elif not self.units(ROBOTICSFACILITY).ready.exists and not self.units(STARGATE).ready.exists and self.units(CYBERNETICSCORE).ready.exists and self.early_game_finished:
                ccore = self.units(CYBERNETICSCORE).ready.first
                if not ccore.has_buff(CHRONOBOOSTENERGYCOST):
                    abilities = await self.get_available_abilities(nexus)
                    if EFFECT_CHRONOBOOSTENERGYCOST in abilities:
                        await self.do(nexus(EFFECT_CHRONOBOOSTENERGYCOST, ccore))
            elif self.units(ROBOTICSFACILITY).ready.exists:
                robo = self.units(ROBOTICSFACILITY).ready.first
                if not robo.has_buff(CHRONOBOOSTENERGYCOST) and not self.units(ROBOTICSFACILITY).ready.noqueue:
                    abilities = await self.get_available_abilities(nexus)
                    if EFFECT_CHRONOBOOSTENERGYCOST in abilities:
                        await self.do(nexus(EFFECT_CHRONOBOOSTENERGYCOST, robo))
            elif self.units(STARGATE).ready.exists:
                star = self.units(STARGATE).ready.first
                if not star.has_buff(CHRONOBOOSTENERGYCOST) and not self.units(STARGATE).ready.noqueue:
                    abilities = await self.get_available_abilities(nexus)
                    if EFFECT_CHRONOBOOSTENERGYCOST in abilities:
                        await self.do(nexus(EFFECT_CHRONOBOOSTENERGYCOST, star))

    async def morph_warpgates(self):
        for gateway in self.units(GATEWAY).ready:
            abilities = await self.get_available_abilities(gateway)
            if MORPH_WARPGATE in abilities and self.can_afford(MORPH_WARPGATE):
                await self.do(gateway(MORPH_WARPGATE))

        if len(self.known_enemy_units.of_type(UnitTypeId.REAPER)) >= 2:
            if self.units(CYBERNETICSCORE).ready.exists and self.early_game_finished:
                if self.units(NEXUS).exists:
                    nexus = self.units(NEXUS).random
                else:
                    nexus = self.units.structure.random
                if self.units(SHIELDBATTERY).closer_than(6, nexus).amount < 1:
                    if self.units(PYLON).ready.closer_than(5, nexus).amount < 1:
                        if self.can_afford(PYLON) and not self.already_pending(PYLON):
                            await self.build(PYLON, near=nexus)
                    else:
                        if self.can_afford(SHIELDBATTERY) and not self.already_pending(SHIELDBATTERY):
                            await self.build(SHIELDBATTERY, near=nexus.position.towards(self.game_info.map_center, random.randrange(-5, -1)))
        elif len(self.known_enemy_units.of_type(UnitTypeId.BANSHEE)) >= 1:
            if self.units(CYBERNETICSCORE).ready.exists and self.early_game_finished:
                if self.units(NEXUS).exists:
                    nexus = self.units(NEXUS).random
                else:
                    nexus = self.units.structure.random
                if self.units(SHIELDBATTERY).closer_than(7, nexus).amount < 2 and len(self.units(SHIELDBATTERY)) < 3:
                    if self.units(PYLON).ready.closer_than(5, nexus).amount < 1:
                        if self.can_afford(PYLON) and not self.already_pending(PYLON):
                            await self.build(PYLON, near=nexus)
                    else:
                        if self.can_afford(SHIELDBATTERY) and not self.already_pending(SHIELDBATTERY):
                            await self.build(SHIELDBATTERY, near=nexus.position.towards(self.game_info.map_center, random.randrange(-5, -1)))

    async def micro_units(self):
        # Some Cheese detected (e.g. YoBot & NaugthyBot). Pull some Probes!
        # if self.time < 240 and self.known_enemy_structures.closer_than(120, self.units(NEXUS).first) and len(self.prg2) > 0:
        #     for pr2 in self.prg2:
        #         self.combinedActions.append(pr2.attack(
        #             self.known_enemy_structures.closest_to(self.units(NEXUS).first).position.random_on_distance(
        #                 random.randrange(1, 3))))
        # elif len(self.prg2) > 0 and not self.known_enemy_structures.closer_than(120, self.units(NEXUS).first):
        #     for pr2 in self.prg2:
        #         self.combinedActions.append(pr2.gather(self.state.vespene_geyser.closest_to(self.units(NEXUS).first)))
        #     self.prg2 = []

        # Stalker-Micro without Blink
        for st in self.units(STALKER):
            # Unit is damaged severely, retreat if possible!
            if st.shield_percentage < 0.1:
                threats = self.known_enemy_units.not_structure.filter(lambda unit: unit.type_id not in self.units_to_ignore).closer_than(st.ground_range + st.radius, st.position)
                if threats.exists and st.position != threats.closest_to(st).position:
                    distance = await self._client.query_pathing(st.position, st.position.towards(threats.closest_to(st).position, -2))
                    if distance is None:
                        # Path is blocked, fight for your life!
                        self.combinedActions.append(st.attack(threats.closest_to(st.position)))

                    else:
                        self.combinedActions.append(st.move(st.position.towards(threats.closest_to(st).position, -2)))

            # Unit is under fire! If possible, kite enemy to minimize damage
            elif st.is_attacking and st.shield_percentage < 1:
                threats = self.known_enemy_units.not_structure.filter(lambda unit: unit.type_id not in self.units_to_ignore).closer_than(st.ground_range + st.radius, st.position)
                if threats.exists:
                    if st.ground_range + st.radius > threats.closest_to(st).ground_range + threats.closest_to(st).radius:
                        if st.ground_range + st.radius > st.distance_to(threats.closest_to(st)):
                            if st.weapon_cooldown > 0 and st.position != threats.closest_to(st).position:
                                distance = await self._client.query_pathing(st.position, st.position.towards(threats.closest_to(st).position, -1))
                                if distance is None:
                                    # Path is blocked, fight for your life!
                                    self.combinedActions.append(st.attack(threats.closest_to(st.position)))

                                else:
                                    self.combinedActions.append(st.move(st.position.towards(threats.closest_to(st).position, -1)))
            # Snipe targets which are a oneshot
            else:
                threats = self.known_enemy_units.filter(lambda unit: unit.type_id not in self.units_to_ignore).closer_than(st.ground_range + st.radius, st.position)
                for threat in threats:
                    if threat.health <= 13:
                        #print('Attacking prefered Enemy', threat, 'with health:', threat.health)
                        self.combinedActions.append(st.attack(threat))


        # Adept-Micro
        for ad in self.units(ADEPT):
            # Unit is damaged severely, retreat if possible!
            if ad.shield_percentage < 0.1:
                threats = self.known_enemy_units.not_structure.filter(lambda unit: unit.type_id not in self.units_to_ignore).closer_than(ad.ground_range + ad.radius, ad.position)
                if threats.exists and ad.position != threats.closest_to(ad).position:
                    distance = await self._client.query_pathing(ad.position, ad.position.towards(threats.closest_to(ad).position, -1))
                    if distance is None:
                        # Path is blocked, fight for your life!
                        self.combinedActions.append(ad.attack(threats.closest_to(ad.position)))
                        #print('- Adept is Blocked! Fighting! -')
                    else:
                        self.combinedActions.append(ad.move(ad.position.towards(threats.closest_to(ad).position, -1)))
                        #print('- Microing Adept! -')
            # Unit is under fire! If possible, kite enemy to minimize damage
            elif ad.is_attacking and ad.shield_percentage < 1:
                threats = self.known_enemy_units.not_structure.filter(lambda unit: unit.type_id not in self.units_to_ignore).closer_than(ad.ground_range + ad.radius, ad.position)
                if threats.exists:
                    if ad.ground_range + ad.radius > threats.closest_to(ad).ground_range + threats.closest_to(ad).radius:
                        if ad.ground_range + ad.radius > ad.distance_to(threats.closest_to(ad)):
                            if ad.weapon_cooldown > 0 and ad.position != threats.closest_to(ad).position:
                                distance = await self._client.query_pathing(ad.position, ad.position.towards(threats.closest_to(ad).position, -1))
                                if distance is None:
                                    # Path is blocked, fight for your life!
                                    self.combinedActions.append(ad.attack(threats.closest_to(ad.position)))
                                    #print('- Adept is Blocked! Fighting! -')
                                else:
                                    #print('- Kiting Enemy -')
                                    self.combinedActions.append(ad.move(ad.position.towards(threats.closest_to(ad).position, -1)))
            # Snipe targets which are a oneshot
            else:
                threats = self.known_enemy_units.filter(lambda unit: unit.type_id not in self.units_to_ignore).closer_than(ad.ground_range + ad.radius, ad.position)
                for threat in threats:
                    if threat.health <= 10:
                        # print('Attacking prefered Enemy', threat, 'with health:', threat.health)
                        self.combinedActions.append(ad.attack(threat))

            # Prioritize targets with a specific armor type
            # elif self.first_attack:
            #     threats = self.known_enemy_units.closer_than(st.ground_range + st.radius, st.position)
            #     for threat in threats:
            #         if threat.is_armored:
            #             print('- Attacking prefered Enemy -', threat)
            #             self.combinedActions.append(st.attack(threat))

        # Immortal-Micro
        for im in self.units(IMMORTAL):
            # Unit is damaged severely, retreat if possible!
            if im.shield_percentage < 0.1:
                threats = self.known_enemy_units.filter(lambda unit: unit.type_id not in self.units_to_ignore).closer_than(im.ground_range + im.radius, im.position)
                if threats.exists and im.position != threats.closest_to(im).position:
                    distance = await self._client.query_pathing(im.position, im.position.towards(threats.closest_to(im).position, -2))
                    if distance is None:
                        # Path is blocked, fight for your life!
                        self.combinedActions.append(im.attack(threats.closest_to(im.position)))
                        #print('- Immortal is Blocked! Fighting! -')
                    else:
                        self.combinedActions.append(
                            im.move(im.position.towards(threats.closest_to(im).position, -2)))
                        #print('- Microing Immortal! -')
            # Unit is under fire! If possible, kite enemy to minimize damage
            elif im.is_attacking and im.shield_percentage < 1:
                threats = self.known_enemy_units.filter(lambda unit: unit.type_id not in self.units_to_ignore).closer_than(im.ground_range + im.radius, im.position)
                if threats.exists:
                    if im.ground_range + im.radius > threats.closest_to(im).ground_range + threats.closest_to(
                            im).radius:
                        if im.ground_range + im.radius > im.distance_to(threats.closest_to(im)):
                            if im.weapon_cooldown > 0 and im.position != threats.closest_to(im).position:
                                distance = await self._client.query_pathing(im.position, im.position.towards(
                                    threats.closest_to(im).position, -2))
                                if distance is None:
                                    # Path is blocked, fight for your life!
                                    self.combinedActions.append(im.attack(threats.closest_to(im.position)))
                                    #print('- Immortal is Blocked! Fighting! -')
                                else:
                                    #print('- Kiting Enemy -')
                                    self.combinedActions.append(
                                        im.move(im.position.towards(threats.closest_to(im).position, -2)))
            # Prioritize targets with a specific armor type
            elif self.first_attack:
                threats = self.known_enemy_units.filter(lambda unit: unit.type_id not in self.units_to_ignore).closer_than(im.ground_range + im.radius, im.position)
                for threat in threats:
                    if threat.is_armored:
                        self.combinedActions.append(im.attack(threat))

        # Colossus-Micro
        for cl in self.units(COLOSSUS):
            # Unit is damaged severely, retreat if possible!
            if cl.shield_percentage < 0.1:
                threats = self.known_enemy_units.not_structure.filter(lambda unit: unit.type_id not in self.units_to_ignore).closer_than(cl.ground_range + cl.radius, cl.position)
                if threats.exists and cl.position != threats.closest_to(cl).position:
                    distance = await self._client.query_pathing(cl.position,
                                                                cl.position.towards(threats.closest_to(cl).position,
                                                                                    -2))
                    if distance is None:
                        # Path is blocked, fight for your life!
                        self.combinedActions.append(cl.attack(threats.closest_to(cl.position)))
                        #print('- Colossus is Blocked! Fighting! -')
                    else:
                        self.combinedActions.append(
                            cl.move(cl.position.towards(threats.closest_to(cl).position, -2)))
                        #print('- Microing Colossus! -')
            # Unit is under fire! If possible, kite enemy to minimize damage
            elif cl.is_attacking and cl.shield_percentage <= 1:
                threats = self.known_enemy_units.not_structure.filter(lambda unit: unit.type_id not in self.units_to_ignore).closer_than(cl.ground_range + cl.radius, cl.position)
                if threats.exists:
                    if cl.ground_range + cl.radius > threats.closest_to(cl).ground_range + threats.closest_to(
                            cl).radius:
                        if cl.ground_range + cl.radius > cl.distance_to(threats.closest_to(cl)):
                            if cl.weapon_cooldown > 0 and cl.position != threats.closest_to(cl).position:
                                distance = await self._client.query_pathing(cl.position, cl.position.towards(
                                    threats.closest_to(cl).position, -2))
                                if distance is None:
                                    # Path is blocked, fight for your life!
                                    self.combinedActions.append(cl.attack(threats.closest_to(cl.position)))
                                    #print('- Colossus is Blocked! Fighting! -')
                                else:
                                    #print('- Kiting Enemy -')
                                    self.combinedActions.append(
                                        cl.move(cl.position.towards(threats.closest_to(cl).position, -2)))

        # Sentry-Micro
        for se in self.units(SENTRY):
            # Unit is damaged severely, retreat if possible!
            threats = self.known_enemy_units.not_structure.filter(lambda unit: unit.type_id not in self.units_to_ignore).closer_than(10, se.position)
            if se.shield_percentage < 0.1:
                if threats.exists and se.position != threats.closest_to(se).position:
                    distance = await self._client.query_pathing(se.position,
                                                                se.position.towards(threats.closest_to(se).position,
                                                                                    -1))
                    if distance is None:
                        # Path is blocked, fight for your life!
                        self.combinedActions.append(se.attack(threats.closest_to(se.position)))
                        # print('- Colossus is Blocked! Fighting! -')
                    else:
                        self.combinedActions.append(
                            se.move(se.position.towards(threats.closest_to(se).position, -1)))
                        # print('- Microing Colossus! -')
            if threats.amount > 4 and not se.has_buff(GUARDIANSHIELD):
                if await self.can_cast(se, GUARDIANSHIELD_GUARDIANSHIELD):
                    await self.do(se(GUARDIANSHIELD_GUARDIANSHIELD))
                    break

        # Voidray-Micro
        for vr in self.units(VOIDRAY):
            # Unit is damaged severely, retreat if possible!
            if vr.shield_percentage < 0.1:
                threats = self.known_enemy_units.filter(lambda unit: unit.type_id not in self.units_to_ignore).closer_than(vr.air_range + vr.radius, vr.position)
                if threats.exists and vr.position != threats.closest_to(vr).position:
                    distance = await self._client.query_pathing(vr.position, vr.position.towards(
                        threats.closest_to(vr).position, -2))
                    if distance is None:
                        # Path is blocked, fight for your life!
                        self.combinedActions.append(vr.attack(threats.closest_to(vr.position)))
                        # print('- Immortal is Blocked! Fighting! -')
                    else:
                        self.combinedActions.append(
                            vr.move(vr.position.towards(threats.closest_to(vr).position, -2)))
                        # print('- Microing Voidray! -')
            # Unit is under fire! If possible, kite enemy to minimize damage
            # elif vr.is_attacking and vr.shield_percentage < 0.5:
            #     threats = self.known_enemy_units.filter(lambda unit: unit.type_id not in self.units_to_ignore).closer_than(vr.air_range + vr.radius, vr.position)
            #     if threats.exists:
            #         if vr.air_range + vr.radius > threats.closest_to(vr).air_range + threats.closest_to(
            #                 vr).radius:
            #             if vr.air_range + vr.radius > vr.distance_to(threats.closest_to(vr)):
            #                 if vr.weapon_cooldown > 0 and vr.position != threats.closest_to(vr).position:
            #                     distance = await self._client.query_pathing(vr.position, vr.position.towards(
            #                         threats.closest_to(vr).position, -2))
            #                     if distance is None:
            #                         # Path is blocked, fight for your life!
            #                         self.combinedActions.append(vr.attack(threats.closest_to(vr.position)))
            #                         # print('- Voidray is Blocked! Fighting! -')
            #                     else:
            #                         # print('- Kiting Enemy -')
            #                         self.combinedActions.append(
            #                             vr.move(vr.position.towards(threats.closest_to(vr).position, -2)))
            # Prioritize targets with a specific armor type
            elif not vr.is_attacking:
                threats = self.known_enemy_units.filter(lambda unit: unit.type_id not in self.units_to_ignore).closer_than(vr.air_range + vr.radius + 2, vr.position)
                # print(threats)
                if threats.exists:
                    for threat in threats:
                        if threat.is_armored and threat.can_attack_air:
                            self.combinedActions.append(vr.attack(threat))
                            # print('Attacking Armored & Airdamage: ', threat)
                            if await self.can_cast(vr, EFFECT_VOIDRAYPRISMATICALIGNMENT):
                                await self.do(vr(EFFECT_VOIDRAYPRISMATICALIGNMENT))
                                #print('Full Damage!')
                        elif threat.can_attack_air:
                            self.combinedActions.append(vr.attack(threat))
                            # print('Attacking Airdamage: ', threat)
                            if await self.can_cast(vr, EFFECT_VOIDRAYPRISMATICALIGNMENT):
                                await self.do(vr(EFFECT_VOIDRAYPRISMATICALIGNMENT))
                                #print('Full Damage!')
                        elif threat.is_armored:
                            self.combinedActions.append(vr.attack(threat))
                            # print('Attacking Armored: ', threat)
                            if await self.can_cast(vr, EFFECT_VOIDRAYPRISMATICALIGNMENT):
                                await self.do(vr(EFFECT_VOIDRAYPRISMATICALIGNMENT))
                                #print('Full Damage!')
                        else:
                            self.combinedActions.append(vr.attack(threat))
                            # print('Attacking else: ', threat)

        await self.do_actions(self.combinedActions)

        # Survive Base-Trade
        if self.units(NEXUS).amount == 0:
                if self.can_afford(PYLON) and self.units(STALKER).exists:
                    await self.build(PYLON, near=self.units(STALKER).random.position)

    # Specific Functions for Two Base Colossus Build Order

    async def build_proxy_pylon_2base_colossus(self):
        if self.lance_started and not self.proxy_built and self.can_afford(PYLON):
            p = self.game_info.map_center.towards(self.enemy_start_locations[0], 17)
            await self.build(PYLON, near=p)
            self.proxy_built = True

    async def two_base_colossus_buildings(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random

            if (len(self.units(GATEWAY)) + len(self.units(WARPGATE))) < (self.time / 60) and (len(self.units(GATEWAY)) + len(self.units(WARPGATE))) < self.MAX_GATES and self.units(ROBOTICSFACILITY).ready.exists:
                if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
                    await self.build(GATEWAY, near=pylon, max_distance=10, random_alternative=False, placement_step=5)
                    #print('Gate #', len(self.units(GATEWAY))+1, 'build @:', self.time)

            if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE):
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=self.units(PYLON).ready.closest_to(self.units(NEXUS).first))

            if self.units(CYBERNETICSCORE).ready.exists and len(self.units(NEXUS)) > 1:
                if len(self.units(ROBOTICSFACILITY)) < self.MAX_ROBOS:
                    if self.can_afford(ROBOTICSFACILITY) and not self.already_pending(ROBOTICSFACILITY):
                        await self.build(ROBOTICSFACILITY, near=pylon, max_distance=10, random_alternative=False, placement_step=5)

            if self.units(ROBOTICSFACILITY).ready.exists:
                if len(self.units(ROBOTICSBAY)) < 1:
                    if self.can_afford(ROBOTICSBAY) and not self.already_pending(ROBOTICSBAY):
                        await self.build(ROBOTICSBAY, near=self.units(PYLON).ready.closest_to(self.units(NEXUS).first), max_distance=10, random_alternative=False, placement_step=5)

    async def two_base_colossus_upgrade(self):
        if self.units(ROBOTICSBAY).ready.exists and self.can_afford(
                RESEARCH_EXTENDEDTHERMALLANCE) and not self.lance_started and self.units(COLOSSUS).ready:
            bay = self.units(ROBOTICSBAY).ready.first
            await self.do(bay(RESEARCH_EXTENDEDTHERMALLANCE))
            self.lance_started = True


    async def two_base_colossus_offensive_force(self):
        for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
            if self.units(ROBOTICSBAY).ready.exists and self.can_afford(COLOSSUS) and self.supply_left > 5:
                await self.do(rf.train(COLOSSUS))

        for gw in self.units(GATEWAY).ready.noqueue:
            if self.units(GATEWAY).ready.exists and self.minerals > 600 and self.supply_left > 1:
                await self.do(gw.train(ZEALOT))
            if self.units(ROBOTICSFACILITY).ready.exists and not self.already_pending(ROBOTICSBAY) and not self.units(ROBOTICSBAY).ready.exists:
                break
            elif self.units(ROBOTICSBAY).ready.exists and self.units(ROBOTICSFACILITY).ready.noqueue:
                break
            elif self.units(CYBERNETICSCORE).ready.exists and self.can_afford(STALKER) and self.supply_left > 1:
                await self.do(gw.train(STALKER))
            elif self.units(GATEWAY).ready.exists and self.minerals > 225 and self.supply_left > 1:
                await self.do(gw.train(ZEALOT))

        for wg in self.units(WARPGATE).ready:
            abilities = await self.get_available_abilities(wg)
            if WARPGATETRAIN_ZEALOT in abilities:
                pylon = self.units(PYLON).ready.closest_to(self.enemy_start_locations[0])
                pos = pylon.position.to2.random_on_distance(random.randrange(1, 6))
                warp_place = await self.find_placement(WARPGATETRAIN_ZEALOT, pos, placement_step=1)
                if self.units(WARPGATE).ready.exists and self.minerals > 600 and self.supply_left > 1:
                    await self.do(wg.warp_in(ZEALOT, warp_place))
                if self.units(ROBOTICSFACILITY).ready.exists and not self.already_pending(ROBOTICSBAY) and not self.units(ROBOTICSBAY).ready.exists:
                    break
                elif self.units(ROBOTICSBAY).ready.exists and self.units(ROBOTICSFACILITY).ready.noqueue:
                    break
                elif self.units(CYBERNETICSCORE).ready.exists and self.can_afford(SENTRY) and self.units(STALKER).amount/(self.units(SENTRY).amount+1) > 6 and self.supply_left > 1:
                    await self.do(wg.warp_in(SENTRY, warp_place))
                elif self.units(CYBERNETICSCORE).ready.exists and self.can_afford(STALKER) and self.supply_left > 1:
                    await self.do(wg.warp_in(STALKER, warp_place))
                elif self.units(WARPGATE).ready.exists and self.minerals > 425 and self.supply_left > 1:
                    await self.do(wg.warp_in(ZEALOT, warp_place))
                elif self.units(WARPGATE).ready.exists and self.vespene > 400 and self.supply_left > 1:
                    await self.do(wg.warp_in(SENTRY, warp_place))

    async def two_base_colossus_unit_control(self):

        # defend as long as there are not 2 colossi, then attack
        if len(self.units(COLOSSUS).ready) <= 1 and not self.first_attack:
            threats = []
            for structure_type in self.defend_around:
                for structure in self.units(structure_type):
                    threats += self.known_enemy_units.filter(lambda unit: unit.type_id not in self.units_to_ignore).closer_than(self.threat_proximity, structure.position)
                    if len(threats) > 0:
                        break
                if len(threats) > 0:
                    break
            if len(threats) > 0 and not self.defend:
                self.defend = True
                self.back_home = True
                defence_target = threats[0].position.random_on_distance(random.randrange(1, 3))
                for cl in self.units(COLOSSUS):
                    self.combinedActions.append(cl.attack(defence_target))
                for se in self.units(SENTRY):
                    self.combinedActions.append(se.attack(defence_target))
                for st in self.units(STALKER):
                    self.combinedActions.append(st.attack(defence_target))
                for zl in self.units(ZEALOT):
                    self.combinedActions.append(zl.attack(defence_target))
            elif len(threats) > 0 and self.defend:
                defence_target = threats[0].position.random_on_distance(random.randrange(1, 3))
                for cl in self.units(COLOSSUS).idle:
                    self.combinedActions.append(cl.attack(defence_target))
                for se in self.units(SENTRY).idle:
                    self.combinedActions.append(se.attack(defence_target))
                for st in self.units(STALKER).idle:
                    self.combinedActions.append(st.attack(defence_target))
                for zl in self.units(ZEALOT).idle:
                    self.combinedActions.append(zl.attack(defence_target))
            elif len(threats) == 0 and self.back_home:
                self.back_home = False
                self.defend = False
                defence_target = self.units(NEXUS).closest_to(self.game_info.map_center).position.towards(self.game_info.map_center, random.randrange(5, 10))
                for cl in self.units(COLOSSUS):
                    self.combinedActions.append(cl.attack(defence_target))
                for se in self.units(SENTRY):
                    self.combinedActions.append(se.attack(defence_target))
                for st in self.units(STALKER):
                    self.combinedActions.append(st.attack(defence_target))
                for zl in self.units(ZEALOT):
                    self.combinedActions.append(zl.attack(defence_target))

        # attack_enemy_start
        elif len(self.units(COLOSSUS).ready) > 1 or (self.first_attack and not self.first_attack_finished):

            if self.time > self.do_something_after:
                all_enemy_base = self.known_enemy_structures
                if all_enemy_base.exists and self.units(NEXUS).exists:
                    next_enemy_base = all_enemy_base.closest_to(self.units(NEXUS).first)
                    attack_target = next_enemy_base.position.random_on_distance(random.randrange(1, 5))
                    gather_target = self.game_info.map_center.towards(self.enemy_start_locations[0], 17)
                elif all_enemy_base.exists:
                    next_enemy_base = all_enemy_base.closest_to(self.game_info.map_center)
                    attack_target = next_enemy_base.position.random_on_distance(random.randrange(1, 5))
                    gather_target = self.game_info.map_center.towards(self.enemy_start_locations[0], 17)
                else:
                    attack_target = self.game_info.map_center.random_on_distance(random.randrange(12, 70 + int(self.time / 60)))
                    gather_target = self.game_info.map_center.towards(self.enemy_start_locations[0], 20)

                if self.gathered and not self.first_attack:
                    for cl in self.units(COLOSSUS):
                        self.combinedActions.append(cl.attack(attack_target))
                    for se in self.units(SENTRY):
                        self.combinedActions.append(se.attack(attack_target))
                    for st in self.units(STALKER):
                        self.combinedActions.append(st.attack(attack_target))
                    for zl in self.units(ZEALOT):
                        self.combinedActions.append(zl.attack(attack_target))
                    self.first_attack = True
                    print('--- First Attack started --- @: ', self.time, 'with Stalkers: ', len(self.units(STALKER)), 'and Zealots: ', len(self.units(ZEALOT)))
                if gather_target and not self.first_attack:
                    for cl in self.units(COLOSSUS):
                        self.combinedActions.append(cl.attack(gather_target))
                    for se in self.units(SENTRY):
                        self.combinedActions.append(se.attack(gather_target))
                    for st in self.units(STALKER):
                        self.combinedActions.append(st.attack(gather_target))
                    for zl in self.units(ZEALOT):
                        self.combinedActions.append(zl.attack(gather_target))
                    wait = 35
                    self.do_something_after = self.time + wait
                    self.gathered = True
                if self.first_attack:
                    for cl in self.units(COLOSSUS).idle:
                        self.combinedActions.append(cl.attack(attack_target))
                    for se in self.units(SENTRY).idle:
                        self.combinedActions.append(se.attack(attack_target))
                    for st in self.units(STALKER).idle:
                        self.combinedActions.append(st.attack(attack_target))
                    for zl in self.units(ZEALOT).idle:
                        self.combinedActions.append(zl.attack(attack_target))

        # seek & destroy
        if self.first_attack and not self.known_enemy_structures.exists and self.time > self.do_something_after and self.time/60 > 10:

            for cl in self.units(COLOSSUS).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(cl.attack(attack_target))
            for se in self.units(SENTRY).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(se.attack(attack_target))
            for st in self.units(STALKER).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(st.attack(attack_target))
            for zl in self.units(ZEALOT).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(zl.attack(attack_target))
            self.do_something_after = self.time + 5

        if self.first_attack and len(self.units(COLOSSUS).ready) < 2:

            lategame_choice = random.randrange(0, 2)
            if lategame_choice == 0:
                self.first_attack_finished = True
                self.first_attack = False
                print('Lategame started @:', self.time)
            else:
                self.first_attack_finished = False
                self.first_attack = False
                print('Fully committing')


        # execuite actions
        await self.do_actions(self.combinedActions)

    async def two_base_colossus_unit_control_lategame(self):

        # defend as long as there is no +2 Armor-Upgrade or supply < 200
        if self.armor_upgrade < 2 and self.supply_used < 190:
            threats = []
            for structure_type in self.defend_around:
                for structure in self.units(structure_type):
                    threats += self.known_enemy_units.filter(lambda unit: unit.type_id not in self.units_to_ignore).closer_than(self.threat_proximity, structure.position)
                    if len(threats) > 0:
                        break
                if len(threats) > 0:
                    break
            if len(threats) > 0 and not self.defend:
                self.defend = True
                self.back_home = True
                defence_target = threats[0].position.random_on_distance(random.randrange(1, 3))
                for cl in self.units(COLOSSUS):
                    self.combinedActions.append(cl.attack(defence_target))
                for se in self.units(SENTRY):
                    self.combinedActions.append(se.attack(defence_target))
                for st in self.units(STALKER):
                    self.combinedActions.append(st.attack(defence_target))
                for zl in self.units(ZEALOT):
                    self.combinedActions.append(zl.attack(defence_target))
            elif len(threats) > 0 and self.defend:
                defence_target = threats[0].position.random_on_distance(random.randrange(1, 3))
                for cl in self.units(COLOSSUS).idle:
                    self.combinedActions.append(cl.attack(defence_target))
                for se in self.units(SENTRY).idle:
                    self.combinedActions.append(se.attack(defence_target))
                for st in self.units(STALKER).idle:
                    self.combinedActions.append(st.attack(defence_target))
                for zl in self.units(ZEALOT).idle:
                    self.combinedActions.append(zl.attack(defence_target))
            elif len(threats) == 0 and self.back_home:
                self.back_home = False
                self.defend = False
                defence_target = self.units(NEXUS).closest_to(self.game_info.map_center).position.towards(
                    self.game_info.map_center, random.randrange(5, 10))
                for cl in self.units(COLOSSUS):
                    self.combinedActions.append(cl.attack(defence_target))
                for se in self.units(SENTRY):
                    self.combinedActions.append(se.attack(defence_target))
                for st in self.units(STALKER):
                    self.combinedActions.append(st.attack(defence_target))
                for zl in self.units(ZEALOT):
                    self.combinedActions.append(zl.attack(defence_target))

        # attack_enemy_start
        elif self.armor_upgrade >= 2 or self.supply_used > 190:

            all_enemy_base = self.known_enemy_structures
            if all_enemy_base.exists and self.units(NEXUS).exists:
                next_enemy_base = all_enemy_base.closest_to(self.units(NEXUS).first)
                attack_target = next_enemy_base.position.random_on_distance(random.randrange(1, 5))
            elif all_enemy_base.exists:
                next_enemy_base = all_enemy_base.closest_to(self.game_info.map_center)
                attack_target = next_enemy_base.position.random_on_distance(random.randrange(1, 5))
            else:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))

            if not self.second_attack:
                for cl in self.units(COLOSSUS):
                    self.combinedActions.append(cl.attack(attack_target))
                for se in self.units(SENTRY):
                    self.combinedActions.append(se.attack(attack_target))
                for st in self.units(STALKER):
                    self.combinedActions.append(st.attack(attack_target))
                for zl in self.units(ZEALOT):
                    self.combinedActions.append(zl.attack(attack_target))
                self.second_attack = True
                print('--- Second Attack started --- @: ', self.time, 'with Stalkers: ',
                      len(self.units(STALKER)), 'and Zealots: ', len(self.units(ZEALOT)))
            if self.second_attack:

                for cl in self.units(COLOSSUS).idle:
                    self.combinedActions.append(cl.attack(attack_target))
                for se in self.units(SENTRY).idle:
                    self.combinedActions.append(se.attack(attack_target))
                for st in self.units(STALKER).idle:
                    self.combinedActions.append(st.attack(attack_target))
                for zl in self.units(ZEALOT).idle:
                    self.combinedActions.append(zl.attack(attack_target))

        # seek & destroy
        if self.second_attack and self.time/60 > 20:

            for cl in self.units(COLOSSUS).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(cl.attack(attack_target))
            for se in self.units(SENTRY).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(se.attack(attack_target))
            for st in self.units(STALKER).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(st.attack(attack_target))
            for zl in self.units(ZEALOT).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(zl.attack(attack_target))
            self.do_something_after = self.time + 5

        # execuite actions
        await self.do_actions(self.combinedActions)

    async def two_base_colossus_upgrade_lategame(self):
        if self.units(NEXUS).exists:
            nexus = self.units(NEXUS).random
        else:
            nexus = self.units.structure.random
        # Build two Forges for double upgrades
        if self.units(FORGE).amount < 2 and not self.already_pending(FORGE):
            if self.can_afford(FORGE):
                await self.build(FORGE, near=self.units(PYLON).ready.random)

        # Always build a cannon in mineral line for defense
        if self.units(FORGE).ready.exists:
            if self.units(PHOTONCANNON).closer_than(10, nexus).amount < 1:
                if self.units(PYLON).ready.closer_than(5, nexus).amount < 1:
                    if self.can_afford(PYLON) and not self.already_pending(PYLON):
                        await self.build(PYLON, near=nexus)
                else:
                    if self.can_afford(PHOTONCANNON) and not self.already_pending(PHOTONCANNON):
                        await self.build(PHOTONCANNON, near=nexus.position.towards(self.game_info.map_center, random.randrange(-10, -1)))

            forge = self.units(FORGE).ready.random

            # Only if we're not upgrading anything yet
            if forge.noqueue and self.can_afford(FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL1): # Das can_afford triggert nicht richtig
                #abilities = await self.get_available_abilities(forge)
                #print('Abilities:', abilities)
                if self.can_afford(FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL1) and self.weapon_upgrade == 0:  #and FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL1 in abilities:
                    await self.do(forge(RESEARCH_PROTOSSGROUNDWEAPONS))
                    self.weapon_upgrade += 1
                elif self.can_afford(FORGERESEARCH_PROTOSSGROUNDARMORLEVEL1) and self.armor_upgrade == 0: #and FORGERESEARCH_PROTOSSGROUNDARMORLEVEL1 in abilities:
                    await self.do(forge(RESEARCH_PROTOSSGROUNDARMOR))
                    self.armor_upgrade += 1
                elif self.can_afford(FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL2)and self.weapon_upgrade == 1: #and FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL2 in abilities:
                    await self.do(forge(RESEARCH_PROTOSSGROUNDWEAPONS))
                    self.weapon_upgrade += 1
                elif self.can_afford(FORGERESEARCH_PROTOSSGROUNDARMORLEVEL2) and self.armor_upgrade == 1: #and FORGERESEARCH_PROTOSSGROUNDARMORLEVEL2 in abilities:
                    await self.do(forge(RESEARCH_PROTOSSGROUNDARMOR))
                    self.armor_upgrade += 1
                elif self.can_afford(FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL3) and self.weapon_upgrade == 2: #and FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL3 in abilities:
                    await self.do(forge(RESEARCH_PROTOSSGROUNDWEAPONS))
                    self.weapon_upgrade += 1
                elif self.can_afford(FORGERESEARCH_PROTOSSGROUNDARMORLEVEL3) and self.armor_upgrade == 2: #and FORGERESEARCH_PROTOSSGROUNDARMORLEVEL3 in abilities:
                    await self.do(forge(RESEARCH_PROTOSSGROUNDARMOR))
                    self.armor_upgrade += 1
                if (self.armor_upgrade == 3 or self.weapon_upgrade == 3) and self.can_afford(FORGERESEARCH_PROTOSSSHIELDSLEVEL1): #and FORGERESEARCH_PROTOSSSHIELDSLEVEL1 in abilities:
                    await self.do(forge(RESEARCH_PROTOSSSHIELDS))

        # Build a Twilight Council
        if not self.units(TWILIGHTCOUNCIL).exists and not self.already_pending(TWILIGHTCOUNCIL):
            if self.can_afford(TWILIGHTCOUNCIL) and self.units(CYBERNETICSCORE).ready.exists:
                await self.build(TWILIGHTCOUNCIL, near=self.units(PYLON).ready.random)

        if self.units(TWILIGHTCOUNCIL).ready.exists and self.can_afford(RESEARCH_BLINK) and not self.blink_started:
            twi = self.units(TWILIGHTCOUNCIL).ready.first
            await self.do(twi(RESEARCH_BLINK))
            self.blink_started = True
        elif self.units(TWILIGHTCOUNCIL).ready.exists and self.can_afford(RESEARCH_CHARGE) and not self.charge_started:
            twi = self.units(TWILIGHTCOUNCIL).ready.first
            await self.do(twi(RESEARCH_CHARGE))
            self.charge_started = True

        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random

            if (len(self.units(GATEWAY)) + len(self.units(WARPGATE))) < (self.time / 60) and (len(self.units(GATEWAY)) + len(self.units(WARPGATE))) < self.MAX_GATES and self.units(ROBOTICSFACILITY).ready.exists:
                if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
                    await self.build(GATEWAY, near=pylon)

        if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE):
            if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                await self.build(CYBERNETICSCORE, near=self.units(PYLON).ready.closest_to(self.units(NEXUS).first))

# Specific Functions for Two Base Immortal Adept Push

    async def immortal_adept_buildings(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random

            if len(self.units(IMMORTAL).ready) >= 1 and not self.units(ROBOTICSFACILITY).ready.noqueue and self.MAX_GATES <= 6:
                self.MAX_GATES = 7

            if (self.already_pending(ROBOTICSFACILITY) or self.units(ROBOTICSFACILITY).ready) and (len(self.units(GATEWAY)) + len(self.units(WARPGATE))) < self.MAX_GATES:
                if self.can_afford(GATEWAY):
                    await self.build(GATEWAY, near=pylon, max_distance=10, random_alternative=False, placement_step=5)

            if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE):
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=self.units(PYLON).ready.closest_to(self.units(NEXUS).first))

            if self.units(CYBERNETICSCORE).ready.exists and len(self.units(NEXUS)) > 1:
                if len(self.units(ROBOTICSFACILITY)) < self.MAX_ROBOS:
                    if self.can_afford(ROBOTICSFACILITY) and not self.already_pending(ROBOTICSFACILITY):
                        await self.build(ROBOTICSFACILITY, near=pylon, max_distance=10, random_alternative=False, placement_step=5)

    async def build_proxy_pylon(self):
        if len(self.units(IMMORTAL).ready) >= 1 and not self.proxy_built and self.can_afford(PYLON):
            p = self.game_info.map_center.towards(self.enemy_start_locations[0], 17)
            await self.build(PYLON, near=p)
            self.proxy_built = True

    async def immortal_adept_offensive_force(self):
        for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
            if len(self.units(IMMORTAL).ready) >= 2 and len(self.units(OBSERVER).ready) == 0 and self.can_afford(OBSERVER) and self.supply_left > 1:
                await self.do(rf.train(OBSERVER))
            elif self.can_afford(IMMORTAL) and self.supply_left > 1:
                await self.do(rf.train(IMMORTAL))

        for gw in self.units(GATEWAY).ready.noqueue:
            if self.units(GATEWAY).ready.exists and self.minerals > 600 and self.supply_left > 1:
                await self.do(gw.train(ZEALOT))
            if self.units(ROBOTICSBAY).ready.exists and self.units(ROBOTICSFACILITY).ready.noqueue:
                break
            elif self.units(CYBERNETICSCORE).ready.exists and self.can_afford(STALKER) and self.supply_left > 1:
                await self.do(gw.train(STALKER))
            elif self.units(GATEWAY).ready.exists and self.minerals > 325 and self.supply_left > 1:
                await self.do(gw.train(ZEALOT))

        for wg in self.units(WARPGATE).ready:
            abilities = await self.get_available_abilities(wg)
            if WARPGATETRAIN_ZEALOT in abilities:
                pylon = self.units(PYLON).ready.closest_to(self.enemy_start_locations[0])
                pos = pylon.position.to2.random_on_distance(random.randrange(1, 6))
                warp_place = await self.find_placement(WARPGATETRAIN_ZEALOT, pos, placement_step=1)
                if self.units(WARPGATE).ready.exists and self.minerals > 600 and self.supply_left > 1:
                    await self.do(wg.warp_in(ZEALOT, warp_place))
                elif self.units(WARPGATE).ready.exists and self.vespene > 400 and self.supply_left > 1:
                    await self.do(wg.warp_in(SENTRY, warp_place))
                if self.units(ROBOTICSFACILITY).ready.exists and self.units(ROBOTICSFACILITY).ready.noqueue:
                    break
                elif self.units(CYBERNETICSCORE).ready.exists and self.can_afford(SENTRY) and (self.units(STALKER).amount+self.units(ADEPT).amount)/(self.units(SENTRY).amount+1) > 6 and self.supply_left > 1:
                    await self.do(wg.warp_in(SENTRY, warp_place))
                elif self.units(CYBERNETICSCORE).ready.exists and self.can_afford(STALKER) and self.supply_left > 1:
                    if len(self.remembered_enemy_units.of_type({UnitTypeId.CARRIER})) > 0:
                        await self.do(wg.warp_in(STALKER, warp_place))
                    else:
                        build_what = random.randrange(0, 5)
                        # print('Build What:', build_what)
                        if build_what < 3:
                            await self.do(wg.warp_in(ADEPT, warp_place))
                        else:
                            await self.do(wg.warp_in(STALKER, warp_place))
                elif self.units(WARPGATE).ready.exists and self.minerals > 325 and self.supply_left > 1:
                    await self.do(wg.warp_in(ZEALOT, warp_place))

    async def immortal_adept_unit_control(self):

        # defend as long as there are not 2 immortals, then attack
        if len(self.units(IMMORTAL).ready) <= 1 and not self.first_attack:
            threats = []
            for structure_type in self.defend_around:
                for structure in self.units(structure_type):
                    threats += self.known_enemy_units.filter(lambda unit: unit.type_id not in self.units_to_ignore).closer_than(self.threat_proximity, structure.position)
                    if len(threats) > 0:
                        break
                if len(threats) > 0:
                    break
            if len(threats) > 0 and not self.defend:
                self.defend = True
                self.back_home = True
                defence_target = threats[0].position.random_on_distance(random.randrange(1, 3))
                for cl in self.units(IMMORTAL):
                    self.combinedActions.append(cl.attack(defence_target))
                for se in self.units(SENTRY):
                    self.combinedActions.append(se.attack(defence_target))
                for st in self.units(STALKER):
                    self.combinedActions.append(st.attack(defence_target))
                for ad in self.units(ADEPT):
                    self.combinedActions.append(ad.attack(defence_target))
                for zl in self.units(ZEALOT):
                    self.combinedActions.append(zl.attack(defence_target))
            elif len(threats) > 0 and self.defend:
                defence_target = threats[0].position.random_on_distance(random.randrange(1, 3))
                for cl in self.units(IMMORTAL).idle:
                    self.combinedActions.append(cl.attack(defence_target))
                for se in self.units(SENTRY).idle:
                    self.combinedActions.append(se.attack(defence_target))
                for st in self.units(STALKER).idle:
                    self.combinedActions.append(st.attack(defence_target))
                for ad in self.units(ADEPT).idle:
                    self.combinedActions.append(ad.attack(defence_target))
                for zl in self.units(ZEALOT).idle:
                    self.combinedActions.append(zl.attack(defence_target))
            elif len(threats) == 0 and self.back_home:
                self.back_home = False
                self.defend = False
                defence_target = self.units(NEXUS).closest_to(self.game_info.map_center).position.towards(
                    self.game_info.map_center, random.randrange(5, 10))
                for cl in self.units(IMMORTAL):
                    self.combinedActions.append(cl.attack(defence_target))
                for se in self.units(SENTRY):
                    self.combinedActions.append(se.attack(defence_target))
                for st in self.units(STALKER):
                    self.combinedActions.append(st.attack(defence_target))
                for ad in self.units(ADEPT):
                    self.combinedActions.append(ad.attack(defence_target))
                for zl in self.units(ZEALOT):
                    self.combinedActions.append(zl.attack(defence_target))

        # attack_enemy_start
        elif len(self.units(IMMORTAL).ready) > 1:

            if self.time > self.do_something_after:
                all_enemy_base = self.known_enemy_structures
                if all_enemy_base.exists and self.units(NEXUS).exists:
                    next_enemy_base = all_enemy_base.closest_to(self.units(NEXUS).first)
                    attack_target = next_enemy_base.position.random_on_distance(random.randrange(1, 5))
                    gather_target = self.game_info.map_center.towards(self.enemy_start_locations[0], 17)
                elif all_enemy_base.exists:
                    next_enemy_base = all_enemy_base.closest_to(self.game_info.map_center)
                    attack_target = next_enemy_base.position.random_on_distance(random.randrange(1, 5))
                    gather_target = self.game_info.map_center.towards(self.enemy_start_locations[0], 17)
                else:
                    attack_target = self.game_info.map_center.random_on_distance(random.randrange(12, 70 + int(self.time / 60)))
                    gather_target = self.game_info.map_center.towards(self.enemy_start_locations[0], 20)

                if self.gathered and not self.first_attack:
                    for cl in self.units(IMMORTAL):
                        self.combinedActions.append(cl.attack(attack_target))
                    for st in self.units(STALKER):
                        self.combinedActions.append(st.attack(attack_target))
                    for se in self.units(SENTRY):
                        self.combinedActions.append(se.attack(attack_target))
                    for zl in self.units(ZEALOT):
                        self.combinedActions.append(zl.attack(attack_target))
                    for ad in self.units(ADEPT):
                        self.combinedActions.append(ad.attack(attack_target))
                    for ob in self.units(OBSERVER):
                        self.combinedActions.append(ob.move(attack_target))
                    self.first_attack = True
                    print('--- First Attack started --- @: ', self.time, 'with Stalkers: ', len(self.units(STALKER)), 'and Adepts: ', len(self.units(ADEPT)), 'and Zealots: ', len(self.units(ZEALOT)))
                if gather_target and not self.first_attack:
                    for cl in self.units(IMMORTAL):
                        self.combinedActions.append(cl.attack(gather_target))
                    for st in self.units(STALKER):
                        self.combinedActions.append(st.attack(gather_target))
                    for se in self.units(SENTRY):
                        self.combinedActions.append(se.attack(gather_target))
                    for zl in self.units(ZEALOT):
                        self.combinedActions.append(zl.attack(gather_target))
                    for ad in self.units(ADEPT):
                        self.combinedActions.append(ad.attack(gather_target))
                    wait = 38
                    self.do_something_after = self.time + wait
                    self.gathered = True
                if self.first_attack:
                    for cl in self.units(IMMORTAL).idle:
                        self.combinedActions.append(cl.attack(attack_target))
                    for st in self.units(STALKER).idle:
                        self.combinedActions.append(st.attack(attack_target))
                    for se in self.units(SENTRY).idle:
                        self.combinedActions.append(se.attack(attack_target))
                    for zl in self.units(ZEALOT).idle:
                        self.combinedActions.append(zl.attack(attack_target))
                    for ad in self.units(ADEPT).idle:
                        self.combinedActions.append(ad.attack(attack_target))
                    if len(self.units(IMMORTAL)) == 0:
                        self.first_attack = False
                        self.gathered = False

            if len(self.units(OBSERVER).ready) >= 1 and self.units(IMMORTAL).ready.exists:
                for ob in self.units(OBSERVER):
                    self.combinedActions.append(ob.move(self.units(IMMORTAL).ready.closest_to(self.enemy_start_locations[0]).position.towards(self.enemy_start_locations[0], random.randrange(5, 7))))

        # seek & destroy
        if self.first_attack and not self.known_enemy_structures.exists and self.time > self.do_something_after:

            for cl in self.units(IMMORTAL).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(cl.attack(attack_target))
            for st in self.units(STALKER).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(st.attack(attack_target))
            for zl in self.units(ZEALOT).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(zl.attack(attack_target))
            for ad in self.units(ADEPT).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(ad.attack(attack_target))
            self.do_something_after = self.time + 5

        # execuite actions
        await self.do_actions(self.combinedActions)

# Specific Functions for Four Gate Proxy Build

    async def build_proxy_pylon_four_gate(self):
        if (len(self.units(GATEWAY)) + len(self.units(WARPGATE))) >= 3 and not self.proxy_built and self.can_afford(PYLON):
            # p = self.game_info.map_center.towards(self.enemy_start_locations[0], 27)
            p = self.game_info.map_center.towards(self.enemy_start_locations[0], 17)
            await self.build(PYLON, near=p)
            self.proxy_built = True


    async def four_gate_buildings(self):
        if self.units(PYLON).ready.exists and self.units(NEXUS).ready.exists:
            pylon = self.units(PYLON).ready.random

            if (len(self.units(GATEWAY)) + len(self.units(WARPGATE))) < self.MAX_GATES:
                if self.can_afford(GATEWAY):
                    await self.build(GATEWAY, near=pylon, max_distance=10, random_alternative=True, placement_step=5)

        if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE):
            if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                await self.build(CYBERNETICSCORE, near=self.units(PYLON).ready.closest_to(self.units(NEXUS).first))

    async def four_gate_offensive_force(self):

        if self.units(GATEWAY).ready.exists:
            for gw in self.units(GATEWAY).ready.noqueue:
                if self.units(CYBERNETICSCORE).ready.exists and self.can_afford(STALKER) and self.supply_left > 1:
                    if str(self.enemy_race) == "Race.Zerg":
                        build_what = random.randrange(0, 2)
                        if build_what == 0:
                            await self.do(gw.train(STALKER))
                        else:
                            await self.do(gw.train(ZEALOT))
                    else:
                        await self.do(gw.train(STALKER))

        for wg in self.units(WARPGATE).ready:
            abilities = await self.get_available_abilities(wg)
            if WARPGATETRAIN_ZEALOT in abilities:
                pylon = self.units(PYLON).ready.closest_to(self.enemy_start_locations[0])
                pos = pylon.position.to2.random_on_distance(random.randrange(1, 6))
                warp_place = await self.find_placement(WARPGATETRAIN_ZEALOT, pos, placement_step=1)
                if self.units(CYBERNETICSCORE).ready.exists and self.can_afford(SENTRY) and (self.units(STALKER).amount+self.units(ADEPT).amount)/(self.units(SENTRY).amount+1) > 10 and self.supply_left > 1:
                    await self.do(wg.warp_in(SENTRY, warp_place))
                elif self.units(CYBERNETICSCORE).ready.exists and self.can_afford(STALKER) and self.supply_left > 1:
                    if str(self.enemy_race) == "Race.Zerg":
                        build_what = random.randrange(1, 3)
                        if build_what == 0:
                            await self.do(wg.warp_in(ADEPT, warp_place))
                        elif build_what == 1:
                            await self.do(wg.warp_in(STALKER, warp_place))
                        else:
                            await self.do(wg.warp_in(ZEALOT, warp_place))
                    else:
                        build_what = random.randrange(0, 2)
                        if build_what == 0:
                            await self.do(wg.warp_in(ADEPT, warp_place))
                        elif build_what == 1:
                            await self.do(wg.warp_in(STALKER, warp_place))
                        else:
                            await self.do(wg.warp_in(ZEALOT, warp_place))
                elif self.units(WARPGATE).ready.exists and self.minerals > 325 and self.supply_left > 1:
                    await self.do(wg.warp_in(ZEALOT, warp_place))
                elif self.units(WARPGATE).ready.exists and self.vespene > 200 and self.supply_left > 1:
                    await self.do(wg.warp_in(SENTRY, warp_place))

    async def four_gate_unit_control(self):

        # defend nexus if there is no proxy pylon
        if not self.gathered and len(self.units(STALKER)) + len(self.units(ZEALOT)) + len(self.units(ADEPT)) < 10:
            threats = []
            for structure_type in self.defend_around:
                for structure in self.units(structure_type):
                    threats += self.known_enemy_units.filter(lambda unit: unit.type_id not in self.units_to_ignore).closer_than(self.threat_proximity, structure.position)
                    if len(threats) > 0:
                        break
                if len(threats) > 0:
                    break
            if len(threats) > 0 and not self.defend:
                self.defend = True
                self.back_home = True
                defence_target = threats[0].position.random_on_distance(random.randrange(1, 3))
                for se in self.units(SENTRY):
                    self.combinedActions.append(se.attack(defence_target))
                for st in self.units(STALKER):
                    self.combinedActions.append(st.attack(defence_target))
                for ad in self.units(ADEPT):
                    self.combinedActions.append(ad.attack(defence_target))
                for zl in self.units(ZEALOT):
                    self.combinedActions.append(zl.attack(defence_target))
            elif len(threats) > 0 and self.defend:
                defence_target = threats[0].position.random_on_distance(random.randrange(1, 3))
                for se in self.units(SENTRY).idle:
                    self.combinedActions.append(se.attack(defence_target))
                for st in self.units(STALKER).idle:
                    self.combinedActions.append(st.attack(defence_target))
                for ad in self.units(ADEPT).idle:
                    self.combinedActions.append(ad.attack(defence_target))
                for zl in self.units(ZEALOT).idle:
                    self.combinedActions.append(zl.attack(defence_target))
            elif len(threats) == 0 and self.back_home:
                self.back_home = False
                self.defend = False
                defence_target = self.units(NEXUS).closest_to(self.game_info.map_center).position.towards(
                    self.game_info.map_center, random.randrange(5, 10))
                for se in self.units(SENTRY):
                    self.combinedActions.append(se.attack(defence_target))
                for st in self.units(STALKER):
                    self.combinedActions.append(st.attack(defence_target))
                for ad in self.units(ADEPT):
                    self.combinedActions.append(ad.attack(defence_target))
                for zl in self.units(ZEALOT):
                    self.combinedActions.append(zl.attack(defence_target))

        # attack_enemy_start
        elif self.proxy_built:

            if self.time > self.do_something_after:
                all_enemy_base = self.known_enemy_structures
                if all_enemy_base.exists and self.units(NEXUS).exists:
                    next_enemy_base = all_enemy_base.closest_to(self.units(NEXUS).first)
                    attack_target = next_enemy_base.position.random_on_distance(random.randrange(1, 5))
                    gather_target = next_enemy_base.position.towards(self.units(NEXUS).first.position, 40)
                elif all_enemy_base.exists:
                    next_enemy_base = all_enemy_base.closest_to(self.game_info.map_center)
                    attack_target = next_enemy_base.position.random_on_distance(random.randrange(1, 5))
                    gather_target = self.game_info.map_center.towards(self.enemy_start_locations[0], 17)
                else:
                    attack_target = self.game_info.map_center.random_on_distance(random.randrange(12, 70 + int(self.time / 60)))
                    gather_target = self.game_info.map_center.towards(self.enemy_start_locations[0], 20)
                if self.gathered and not self.first_attack and len(self.units(STALKER)) + len(self.units(ZEALOT)) + len(self.units(ADEPT)) >= 10:
                    for st in self.units(STALKER):
                        self.combinedActions.append(st.attack(attack_target))
                    for se in self.units(SENTRY):
                        self.combinedActions.append(se.attack(attack_target))
                    for zl in self.units(ZEALOT):
                        self.combinedActions.append(zl.attack(attack_target))
                    for ad in self.units(ADEPT):
                        self.combinedActions.append(ad.attack(attack_target))
                    self.first_attack = True
                    print('--- First Attack started --- @: ', self.time, 'with Stalkers: ', len(self.units(STALKER)), 'and Adepts: ', len(self.units(ADEPT)), 'and Zealots: ', len(self.units(ZEALOT)))
                if gather_target and not self.first_attack and len(self.units(STALKER)) + len(self.units(ZEALOT)) + len(self.units(ADEPT)) >= 6:
                    for st in self.units(STALKER):
                        self.combinedActions.append(st.attack(gather_target))
                    for se in self.units(SENTRY):
                        self.combinedActions.append(se.attack(gather_target))
                    for zl in self.units(ZEALOT):
                        self.combinedActions.append(zl.attack(gather_target))
                    for ad in self.units(ADEPT):
                        self.combinedActions.append(ad.attack(gather_target))
                    wait = 32
                    self.do_something_after = self.time + wait
                    self.gathered = True
                if self.first_attack:
                    for st in self.units(STALKER).idle:
                        self.combinedActions.append(st.attack(attack_target))
                    for se in self.units(SENTRY).idle:
                        self.combinedActions.append(se.attack(attack_target))
                    for zl in self.units(ZEALOT).idle:
                        self.combinedActions.append(zl.attack(attack_target))
                    for ad in self.units(ADEPT).idle:
                        self.combinedActions.append(ad.attack(attack_target))

        # seek & destroy
        if self.first_attack and not self.known_enemy_structures.exists and self.time > self.do_something_after:

            for se in self.units(SENTRY).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(se.attack(attack_target))
            for st in self.units(STALKER).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(st.attack(attack_target))
            for zl in self.units(ZEALOT).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(zl.attack(attack_target))
            for ad in self.units(ADEPT).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(ad.attack(attack_target))
            self.do_something_after = self.time + 5

        # execuite actions
        await self.do_actions(self.combinedActions)

# Specific Functions for One Base DT Build Order

    async def one_base_dt_buildings(self):
        # Build a Twilight Council
        if self.units(PYLON).exists:
            if self.units(NEXUS).exists:
                pylon = self.units(PYLON).ready.closest_to(self.units(NEXUS).random.position.towards(self.game_info.map_center, random.randrange(1, 8)))
                nexus = self.units(NEXUS).random
            else:
                pylon = self.units(PYLON).ready.random
                nexus = self.units(PYLON).ready.random
            if len(self.units(SHIELDBATTERY)) >= 1 and len(self.units(PHOTONCANNON)) >= 2 and self.units(DARKSHRINE).exists:
                self.MAX_GATES = 3
            else:
                self.MAX_GATES = 2
            # Build a Forge for Cannons
            if self.units(FORGE).amount < 1 and not self.already_pending(FORGE):
                if self.can_afford(FORGE):
                    await self.build(FORGE, near=self.units(PYLON).ready.random)
            if (len(self.units(GATEWAY)) + len(self.units(WARPGATE))) < self.MAX_GATES:
                if self.can_afford(GATEWAY):
                    await self.build(GATEWAY, near=self.units(PYLON).ready.random, max_distance=10, random_alternative=False, placement_step=5)
            elif len(self.units(SHIELDBATTERY)) < (len(self.units(ZEALOT)) + len(self.units(STALKER)) + len(self.units(ADEPT)))/2 and len(self.units(SHIELDBATTERY)) < 1:
                if self.can_afford(SHIELDBATTERY) and self.units(NEXUS).ready.exists:
                    position = await self.find_placement(SHIELDBATTERY, self.main_base_ramp.barracks_correct_placement.rounded, max_distance=10, random_alternative=True, placement_step=3)
                    await self.build(SHIELDBATTERY, near=position)
            # Build a Forge for Cannons
            elif self.units(FORGE).amount < 1 and not self.already_pending(FORGE):
                if self.can_afford(FORGE):
                    await self.build(FORGE, near=self.units(PYLON).ready.random)
            elif self.units(FORGE).ready.exists and len(self.units(PHOTONCANNON)) < 1 and not self.already_pending(PHOTONCANNON):
                if self.can_afford(PHOTONCANNON) and self.units(NEXUS).ready.exists:
                    position = await self.find_placement(PHOTONCANNON,
                                                         self.main_base_ramp.barracks_correct_placement.rounded,
                                                         max_distance=10, random_alternative=False, placement_step=3)
                    await self.build(PHOTONCANNON, near=position)
            # Always build a cannon in mineral line for defense
            elif self.units(FORGE).ready.exists:
                if self.units(PHOTONCANNON).closer_than(7, nexus).amount < 1:
                    if self.units(PYLON).ready.closer_than(7, nexus).amount < 1:
                        if self.can_afford(PYLON) and not self.already_pending(PYLON):
                            await self.build(PYLON, near=nexus.position.towards(self.game_info.map_center,
                                                                         random.randrange(-6, -1)), random_alternative=False, placement_step=1)
                    else:
                        if self.can_afford(PHOTONCANNON) and not self.already_pending(PHOTONCANNON):
                            await self.build(PHOTONCANNON,
                                             near=nexus.position.towards(self.game_info.map_center,
                                                                         random.randrange(-6, -1)), random_alternative=False, placement_step=1)

            if len(self.units(TWILIGHTCOUNCIL)) < 1 and not self.units(TWILIGHTCOUNCIL).exists and not self.already_pending(TWILIGHTCOUNCIL):
                if self.can_afford(TWILIGHTCOUNCIL) and self.units(CYBERNETICSCORE).ready.exists:
                    await self.build(TWILIGHTCOUNCIL, near=self.units(PYLON).ready.random, max_distance=10, random_alternative=False, placement_step=4)

            if len(self.units(DARKSHRINE)) < 1 and not self.units(DARKSHRINE).exists and not self.already_pending(DARKSHRINE):
                if self.can_afford(DARKSHRINE) and self.units(TWILIGHTCOUNCIL).ready.exists:
                    await self.build(DARKSHRINE, near=pylon)

            if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE):
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=self.units(PYLON).ready.closest_to(self.units(NEXUS).first))

            if len(self.units(DARKTEMPLAR)) >= 2 and self.units(TWILIGHTCOUNCIL).ready.exists and self.can_afford(RESEARCH_CHARGE) and not self.charge_started:
                twi = self.units(TWILIGHTCOUNCIL).ready.first
                await self.do(twi(RESEARCH_CHARGE))
                self.charge_started = self.time
                # print(self.charge_started)

    async def build_proxy_pylon_dt(self):
        if self.units(TWILIGHTCOUNCIL).ready.exists and not self.proxy_built and self.can_afford(PYLON):
            p = self.game_info.map_center.towards(self.enemy_start_locations[0], 17)
            await self.build(PYLON, near=p)
            self.proxy_built = True

    async def one_base_dt_offensive_force(self):

        for gw in self.units(GATEWAY).ready.noqueue:
            if self.units(CYBERNETICSCORE).ready.exists and len(self.units(ADEPT)) > 4 and not self.already_pending(TWILIGHTCOUNCIL) and not self.units(TWILIGHTCOUNCIL).ready.exists and self.minerals < 150:
                break
            elif self.units(TWILIGHTCOUNCIL).ready.exists and not self.already_pending(DARKSHRINE) and not self.units(DARKSHRINE).ready.exists and self.minerals < 150:
                break
            elif self.units(GATEWAY).ready.exists and self.units(DARKSHRINE).ready.exists and self.can_afford(DARKTEMPLAR) and self.supply_left > 1:
                await self.do(gw.train(DARKTEMPLAR))
            elif self.units(GATEWAY).ready.exists and self.units(CYBERNETICSCORE).ready.exists and len(self.units(STALKER)) == 0 and self.can_afford(STALKER) and self.supply_left > 1:
                await self.do(gw.train(STALKER))
            elif self.units(GATEWAY).ready.exists and self.units(CYBERNETICSCORE).ready.exists and len(self.units(STALKER)) > 0 and len(self.units(ADEPT))/len(self.units(STALKER)) > 3 and self.can_afford(STALKER) and self.supply_left > 1:
                await self.do(gw.train(STALKER))
            elif self.units(GATEWAY).ready.exists and self.units(CYBERNETICSCORE).ready.exists and self.can_afford(ADEPT) and self.supply_left > 1:
                await self.do(gw.train(ADEPT))
            elif self.units(GATEWAY).ready.exists and self.minerals > 250 and self.supply_left > 1:
                await self.do(gw.train(ZEALOT))

        for wg in self.units(WARPGATE).ready:
            abilities = await self.get_available_abilities(wg)
            if WARPGATETRAIN_ZEALOT in abilities:
                if self.units(SHIELDBATTERY).ready.exists:
                    pylon = self.units(PYLON).closest_to(self.units(SHIELDBATTERY).random)
                else:
                    pylon = self.units(PYLON).random
                pos = pylon.position.to2.random_on_distance(random.randrange(1, 6))
                warp_place = await self.find_placement(WARPGATETRAIN_ZEALOT, pos, placement_step=1)
                if self.units(TWILIGHTCOUNCIL).ready.exists and not self.already_pending(
                        DARKSHRINE) and not self.units(DARKSHRINE).ready.exists and self.minerals < 150:
                    break
                elif self.units(DARKSHRINE).ready.exists and self.minerals < 130 and self.vespene < 130:
                    break
                elif self.units(DARKSHRINE).ready.exists and self.can_afford(DARKTEMPLAR) and self.supply_left > 1:
                    proxy_pylon = self.units(PYLON).closest_to(self.enemy_start_locations[0])
                    pos = proxy_pylon.position.to2.random_on_distance(random.randrange(1, 6))
                    warp_place_dt = await self.find_placement(WARPGATETRAIN_ZEALOT, pos, placement_step=1)
                    await self.do(wg.warp_in(DARKTEMPLAR, warp_place_dt))
                elif len(self.units(DARKTEMPLAR)) >= 3 and self.can_afford(STALKER) and self.supply_left > 1:
                    await self.do(wg.warp_in(STALKER, warp_place))
                elif len(self.units(STALKER)) == 0 and self.can_afford(STALKER) and self.supply_left > 1:
                    await self.do(wg.warp_in(STALKER, warp_place))
                elif len(self.units(STALKER)) > 0 and len(self.units(ADEPT))/len(self.units(STALKER)) > 3 and self.can_afford(STALKER) and self.supply_left > 1:
                    await self.do(wg.warp_in(STALKER, warp_place))
                elif self.units(WARPGATE).ready.exists and self.vespene > 150 and self.minerals > 150 and self.can_afford(STALKER) and self.supply_left > 1:
                    await self.do(wg.warp_in(STALKER, warp_place))
                # elif self.units(WARPGATE).ready.exists and self.vespene > 150 and self.minerals > 150 and len(self.known_enemy_units.of_type(UnitTypeId.BANSHEE)) >= 1 and self.can_afford(STALKER) and self.supply_left > 1:
                #     await self.do(wg.warp_in(STALKER, warp_place))
                # elif self.units(WARPGATE).ready.exists and self.vespene > 150 and self.minerals > 150 and len(self.known_enemy_units.of_type(UnitTypeId.BANSHEE)) == 0 and self.can_afford(ADEPT) and self.supply_left > 1:
                #     await self.do(wg.warp_in(ADEPT, warp_place))
                elif self.units(WARPGATE).ready.exists and self.minerals > 250 and self.supply_left > 1:
                    await self.do(wg.warp_in(ZEALOT, warp_place))

    async def dt_unit_control(self):

        # defend nexus if there is no proxy pylon
        if not self.gathered:
            threats = []
            for structure_type in self.defend_around:
                for structure in self.units(structure_type):
                    threats += self.known_enemy_units.filter(lambda unit: unit.type_id not in self.units_to_ignore).closer_than(self.threat_proximity, structure.position)
                    if len(threats) > 0:
                        break
                if len(threats) > 0:
                    break
            if len(threats) > 0 and not self.defend:
                self.defend = True
                self.back_home = True
                defence_target = threats[0].position.random_on_distance(random.randrange(1, 3))
                for st in self.units(STALKER):
                    self.combinedActions.append(st.attack(defence_target))
                for ad in self.units(ADEPT):
                    self.combinedActions.append(ad.attack(defence_target))
                for zl in self.units(ZEALOT):
                    self.combinedActions.append(zl.attack(defence_target))
            elif len(threats) > 0 and self.defend:
                defence_target = threats[0].position.random_on_distance(random.randrange(1, 3))
                for st in self.units(STALKER).idle:
                    self.combinedActions.append(st.attack(defence_target))
                for ad in self.units(ADEPT).idle:
                    self.combinedActions.append(ad.attack(defence_target))
                for zl in self.units(ZEALOT).idle:
                    self.combinedActions.append(zl.attack(defence_target))
            elif len(threats) == 0 and self.back_home and self.units(NEXUS).exists:
                self.back_home = False
                self.defend = False
                defence_target = self.units(NEXUS).closest_to(self.game_info.map_center).position.towards(
                    self.game_info.map_center, random.randrange(1, 2))
                for st in self.units(STALKER):
                    self.combinedActions.append(st.attack(defence_target))
                for ad in self.units(ADEPT):
                    self.combinedActions.append(ad.attack(defence_target))
                for zl in self.units(ZEALOT):
                    self.combinedActions.append(zl.attack(defence_target))
            elif len(threats) == 0 and self.units(NEXUS).exists:
                self.back_home = False
                self.defend = False
                defence_target = self.units(NEXUS).closest_to(self.game_info.map_center).position.towards(
                    self.game_info.map_center, random.randrange(1, 2))
                for st in self.units(STALKER).idle:
                    self.combinedActions.append(st.attack(defence_target))
                for ad in self.units(ADEPT).idle:
                    self.combinedActions.append(ad.attack(defence_target))
                for zl in self.units(ZEALOT).idle:
                    self.combinedActions.append(zl.attack(defence_target))

        # attack_enemy_start
        if self.proxy_built and len(self.units(DARKTEMPLAR)) >= 1:
            if self.time > self.do_something_after and self.known_enemy_structures.exists:
                all_enemy_base = self.known_enemy_structures
                if all_enemy_base.exists and self.units(NEXUS).exists:
                    next_enemy_base = all_enemy_base.closest_to(self.units(NEXUS).first)
                    attack_target_exe = next_enemy_base.position.random_on_distance(random.randrange(1, 5))
                    attack_target_main = self.enemy_start_locations[0].random_on_distance(random.randrange(1, 5))
                else:
                    attack_target_main = self.known_enemy_structures.closest_to(self.units(DARKTEMPLAR).random.position).position.random_on_distance(random.randrange(1, 5))
                    attack_target_exe = attack_target_main
                if attack_target_main and not self.first_attack:
                    dt1 = self.units(DARKTEMPLAR)[0]
                    self.combinedActions.append(dt1(RALLY_UNITS, self.enemy_start_locations[0].towards(self.game_info.map_center, random.randrange(-5, -1))))
                    if len(self.units(DARKTEMPLAR)) == 2:
                        dt2 = self.units(DARKTEMPLAR)[1]
                        self.combinedActions.append(dt2(RALLY_UNITS, attack_target_exe))
                    if len(self.units(DARKTEMPLAR)) == 3:
                        dt3 = self.units(DARKTEMPLAR)[2]
                        self.combinedActions.append(dt3(RALLY_UNITS, attack_target_main))
                    if len(self.units(DARKTEMPLAR)) == 4:
                        dt4 = self.units(DARKTEMPLAR)[3]
                        self.combinedActions.append(dt4(RALLY_UNITS, attack_target_exe))
                    self.first_attack = True
                    print('--- First Attack started --- @: ', self.time, 'with Stalkers: ', len(self.units(STALKER)),
                          'and Dark-Templars: ', len(self.units(DARKTEMPLAR)), 'and Adepts: ', len(self.units(ADEPT)), 'and Zealots: ', len(self.units(ZEALOT)))
                if self.first_attack:
                    for dt in self.units(DARKTEMPLAR).idle:
                        if len(self.known_enemy_units.of_type(UnitTypeId.SPORECRAWLER)) > 0:
                            #print('Attacking Spore')
                            self.combinedActions.append(dt.attack(self.known_enemy_units.of_type(UnitTypeId.SPORECRAWLER).closest_to(dt.position)))
                        elif len(self.known_enemy_units.of_type({UnitTypeId.DRONE, UnitTypeId.PROBE, UnitTypeId.SCV})) > 0:
                            #print('Attacking Drone')
                            self.combinedActions.append(dt.attack(self.known_enemy_units.of_type({UnitTypeId.DRONE, UnitTypeId.PROBE, UnitTypeId.SCV}).closest_to(dt.position)))
                        else:
                            #print('Attacking Else')
                            self.combinedActions.append(dt.attack(attack_target_exe))

        if self.charge_started > 0 and self.time - self.charge_started >= 90:
            if self.time > self.do_something_after:
                all_enemy_base = self.known_enemy_structures
                if all_enemy_base.exists and self.units(NEXUS).exists:
                    next_enemy_base = all_enemy_base.closest_to(self.units(NEXUS).first)
                    attack_target = next_enemy_base.position.random_on_distance(random.randrange(1, 5))
                    gather_target = next_enemy_base.position.towards(self.units(NEXUS).first.position, 40)
                elif all_enemy_base.exists:
                    next_enemy_base = all_enemy_base.closest_to(self.game_info.map_center)
                    attack_target = next_enemy_base.position.random_on_distance(random.randrange(1, 5))
                    gather_target = self.game_info.map_center.towards(self.enemy_start_locations[0], 17)
                else:
                    attack_target = self.game_info.map_center.random_on_distance(random.randrange(12, 70 + int(self.time / 60)))
                    gather_target = self.game_info.map_center.towards(self.enemy_start_locations[0], 20)
                if self.gathered and not self.second_attack:
                    for st in self.units(STALKER):
                        self.combinedActions.append(st.attack(attack_target))
                    for ad in self.units(ADEPT):
                        self.combinedActions.append(ad.attack(attack_target))
                    for dt in self.units(DARKTEMPLAR):
                        self.combinedActions.append(dt.attack(attack_target))
                    for zl in self.units(ZEALOT):
                        self.combinedActions.append(zl.attack(attack_target))
                    self.second_attack = True
                    print('--- Second Attack started --- @: ', self.time, 'with Stalkers: ', len(self.units(STALKER)),
                          'and Darktemplars: ', len(self.units(DARKTEMPLAR)), 'and Adepts: ', len(self.units(ADEPT)), 'and Zealots: ', len(self.units(ZEALOT)))
                if gather_target and not self.second_attack and len(self.known_enemy_units.of_type(UnitTypeId.BANSHEE)) == 0:
                    for st in self.units(STALKER):
                        self.combinedActions.append(st.attack(gather_target))
                    for ad in self.units(ADEPT):
                        self.combinedActions.append(ad.attack(gather_target))
                    for dt in self.units(DARKTEMPLAR).idle:
                        self.combinedActions.append(dt.attack(gather_target))
                    for zl in self.units(ZEALOT):
                        self.combinedActions.append(zl.attack(gather_target))
                    wait = 30
                    self.do_something_after = self.time + wait
                    self.gathered = True
                if self.second_attack:
                    for st in self.units(STALKER).idle:
                        self.combinedActions.append(st.attack(attack_target))
                    for ad in self.units(ADEPT).idle:
                        self.combinedActions.append(ad.attack(attack_target))
                    for dt in self.units(DARKTEMPLAR).idle:
                        self.combinedActions.append(dt.attack(attack_target))
                    for zl in self.units(ZEALOT).idle:
                        self.combinedActions.append(zl.attack(attack_target))

        # if self.second_attack:
        #     threats = []
        #     for structure_type in self.defend_around:
        #         for structure in self.units(structure_type):
        #             threats += self.known_enemy_units.filter(lambda unit: unit.type_id not in self.units_to_ignore).closer_than(self.threat_proximity, structure.position)
        #             if len(threats) > 0:
        #                 break
        #         if len(threats) > 0:
        #             break
        #     if len(threats) > 0 and not self.defend:
        #         self.defend = True
        #         self.back_home = True
        #         defence_target = threats[0].position.random_on_distance(random.randrange(1, 3))
        #         for se in self.units(DARKTEMPLAR).idle:
        #             self.combinedActions.append(se.attack(defence_target))
        #     elif len(threats) > 0 and self.defend:
        #         defence_target = threats[0].position.random_on_distance(random.randrange(1, 3))
        #         for se in self.units(DARKTEMPLAR).idle:
        #             self.combinedActions.append(se.attack(defence_target))
        #     elif len(threats) == 0 and self.back_home and self.units(NEXUS).exists:
        #         self.back_home = False
        #         self.defend = False
        #         defence_target = self.units(NEXUS).closest_to(self.game_info.map_center).position.towards(
        #             self.game_info.map_center, random.randrange(8, 10))
        #         for se in self.units(DARKTEMPLAR).idle:
        #             self.combinedActions.append(se.attack(defence_target))
        #     elif len(threats) == 0 and self.units(NEXUS).exists:
        #         self.back_home = False
        #         self.defend = False
        #         defence_target = self.units(NEXUS).closest_to(self.game_info.map_center).position.towards(
        #             self.game_info.map_center, random.randrange(8, 10))
        #         for se in self.units(DARKTEMPLAR).idle:
        #             self.combinedActions.append(se.attack(defence_target))

        # Switch to Archons
        # if self.first_attack and len(self.units(DARKTEMPLAR)) < 2:
        # Currently not supported by the API =(
        #     return

        # seek & destroy
        if self.first_attack and not self.known_enemy_structures.exists and self.time > self.do_something_after:

            for se in self.units(DARKTEMPLAR).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(se.attack(attack_target))
            for st in self.units(STALKER).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(st.attack(attack_target))
            for zl in self.units(ZEALOT).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(zl.attack(attack_target))
            for ad in self.units(ADEPT).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(ad.attack(attack_target))
            self.do_something_after = self.time + 5

        # execuite actions
        await self.do_actions(self.combinedActions)

# Specific Functions for One Base VR Build Order

    async def one_base_vr_buildings(self):
        # Build a Twilight Council
        if self.units(PYLON).exists:
            if self.units(NEXUS).exists:
                pylon = self.units(PYLON).ready.closest_to(
                    self.units(NEXUS).random.position.towards(self.game_info.map_center, random.randrange(-8, 2)))
            else:
                pylon = self.units(PYLON).ready.random
            if (len(self.units(GATEWAY)) + len(self.units(WARPGATE))) < self.MAX_GATES:
                if self.can_afford(GATEWAY):
                    await self.build(GATEWAY, near=self.units(PYLON).ready.random, max_distance=10, random_alternative=False,
                                     placement_step=5)
            elif len(self.units(SHIELDBATTERY)) < (
                    len(self.units(ZEALOT)) + len(self.units(STALKER)) + len(self.units(ADEPT))) / 2 and len(
                    self.units(SHIELDBATTERY)) < 2 and not self.already_pending(SHIELDBATTERY):
                if self.can_afford(SHIELDBATTERY) and self.units(NEXUS).ready.exists:
                    position = await self.find_placement(SHIELDBATTERY,
                                                         self.main_base_ramp.barracks_correct_placement.rounded,
                                                         max_distance=10, random_alternative=False, placement_step=3)
                    await self.build(SHIELDBATTERY, near=position)
            elif self.units(CYBERNETICSCORE).ready.exists and self.can_afford(STARGATE) and not self.already_pending(STARGATE) and len(self.units(STARGATE).ready) < 1:
                await self.build(STARGATE, near=pylon, max_distance=10, random_alternative=False,
                                     placement_step=5)
            elif self.units(CYBERNETICSCORE).ready.exists and self.can_afford(STARGATE) and not self.already_pending(STARGATE) and len(self.units(STARGATE).ready) < 2 and self.vespene > 300:
                await self.build(STARGATE, near=pylon, max_distance=10, random_alternative=False,
                                     placement_step=5)

            if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE):
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE,
                                     near=self.units(PYLON).ready.closest_to(self.units(NEXUS).first))

            if len(self.units(VOIDRAY)) >= 2 and not self.units(TWILIGHTCOUNCIL).exists and not self.already_pending(TWILIGHTCOUNCIL):
                if self.can_afford(TWILIGHTCOUNCIL) and self.units(CYBERNETICSCORE).ready.exists:
                    await self.build(TWILIGHTCOUNCIL, near=self.units(PYLON).ready.random, max_distance=10, random_alternative=False,
                                     placement_step=5)

            if len(self.units(VOIDRAY)) >= 2 and self.units(TWILIGHTCOUNCIL).ready.exists and self.can_afford(
                    RESEARCH_CHARGE) and not self.charge_started:
                twi = self.units(TWILIGHTCOUNCIL).ready.first
                await self.do(twi(RESEARCH_CHARGE))
                self.charge_started = self.time
                # print(self.charge_started)

    async def one_base_vr_offensive_force(self):

        for sg in self.units(STARGATE).ready.noqueue:
            if self.can_afford(VOIDRAY) and self.supply_left > 3:
                await self.do(sg.train(VOIDRAY))

        for gw in self.units(GATEWAY).ready.noqueue:
            if self.units(CYBERNETICSCORE).ready.exists and len(self.units(ZEALOT)) + len(self.units(STALKER)) > 8 and not self.already_pending(
                    STARGATE) and len(self.units(STARGATE).ready) < 2 and self.minerals < 150:
                break
            elif self.units(STARGATE).ready.exists and not self.already_pending(
                    TWILIGHTCOUNCIL) and not self.units(TWILIGHTCOUNCIL).ready.exists and self.minerals < 150:
                break
            elif self.units(GATEWAY).ready.exists and self.units(STARGATE).ready.exists and self.minerals < 250:
                break
            elif self.units(GATEWAY).ready.exists and self.units(CYBERNETICSCORE).ready.exists and len(
                    self.units(STALKER)) <= 5 and self.can_afford(STALKER) and self.supply_left > 1:
                await self.do(gw.train(STALKER))
            elif self.units(GATEWAY).ready.exists and self.can_afford(
                    ZEALOT) and self.supply_left > 1:
                await self.do(gw.train(ZEALOT))
            elif self.units(GATEWAY).ready.exists and self.minerals > 250 and self.supply_left > 1:
                await self.do(gw.train(ZEALOT))

        for wg in self.units(WARPGATE).ready:
            abilities = await self.get_available_abilities(wg)
            if WARPGATETRAIN_ZEALOT in abilities:
                if self.units(SHIELDBATTERY).ready.exists:
                    pylon = self.units(PYLON).closest_to(self.units(SHIELDBATTERY).random)
                else:
                    pylon = self.units(PYLON).random
                pos = pylon.position.to2.random_on_distance(random.randrange(1, 6))
                warp_place = await self.find_placement(WARPGATETRAIN_ZEALOT, pos, placement_step=1)
                if self.units(STARGATE).ready.exists and not self.already_pending(
                    TWILIGHTCOUNCIL) and not self.units(TWILIGHTCOUNCIL).ready.exists and self.minerals < 150:
                    break
                elif self.units(STARGATE).ready.exists and self.minerals < 250 and self.vespene < 150:
                    break
                elif len(self.units(VOIDRAY)) >= 10 and self.can_afford(STALKER) and self.supply_left > 1:
                    await self.do(wg.warp_in(STALKER, warp_place))
                elif len(self.units(STALKER)) <= 5 and self.can_afford(STALKER) and self.supply_left > 1:
                    await self.do(wg.warp_in(STALKER, warp_place))
                if self.units(WARPGATE).ready.exists and self.minerals > 350 and self.supply_left > 1:
                    await self.do(wg.warp_in(ZEALOT, warp_place))

    async def vr_unit_control(self):

        # defend nexus if there is no proxy pylon
        if not self.gathered:
            threats = []
            for structure_type in self.defend_around:
                for structure in self.units(structure_type):
                    threats += self.known_enemy_units.filter(lambda unit: unit.type_id not in self.units_to_ignore).closer_than(self.threat_proximity, structure.position)
                    if len(threats) > 0:
                        break
                if len(threats) > 0:
                    break
            if len(threats) > 0 and not self.defend:
                self.defend = True
                self.back_home = True
                defence_target = threats[0].position.random_on_distance(random.randrange(1, 3))
                for vr in self.units(VOIDRAY):
                    self.combinedActions.append(vr.attack(defence_target))
                for st in self.units(STALKER):
                    self.combinedActions.append(st.attack(defence_target))
                for ad in self.units(ADEPT):
                    self.combinedActions.append(ad.attack(defence_target))
                for zl in self.units(ZEALOT):
                    self.combinedActions.append(zl.attack(defence_target))
            elif len(threats) > 0 and self.defend:
                defence_target = threats[0].position.random_on_distance(random.randrange(1, 3))
                for vr in self.units(VOIDRAY).idle:
                    self.combinedActions.append(vr.attack(defence_target))
                for st in self.units(STALKER).idle:
                    self.combinedActions.append(st.attack(defence_target))
                for ad in self.units(ADEPT).idle:
                    self.combinedActions.append(ad.attack(defence_target))
                for zl in self.units(ZEALOT).idle:
                    self.combinedActions.append(zl.attack(defence_target))
            elif len(threats) == 0 and self.back_home and self.units(NEXUS).exists:
                self.back_home = False
                self.defend = False
                defence_target = self.units(NEXUS).closest_to(self.game_info.map_center).position.towards(
                    self.game_info.map_center, random.randrange(8, 10))
                for vr in self.units(VOIDRAY):
                    self.combinedActions.append(vr.attack(defence_target))
                for st in self.units(STALKER):
                    self.combinedActions.append(st.attack(defence_target))
                for ad in self.units(ADEPT):
                    self.combinedActions.append(ad.attack(defence_target))
                for zl in self.units(ZEALOT):
                    self.combinedActions.append(zl.attack(defence_target))
            elif len(threats) == 0 and self.units(NEXUS).exists:
                self.back_home = False
                self.defend = False
                defence_target = self.units(NEXUS).closest_to(self.game_info.map_center).position.towards(
                    self.game_info.map_center, random.randrange(8, 10))
                for vr in self.units(VOIDRAY).idle:
                    self.combinedActions.append(vr.attack(defence_target))
                for st in self.units(STALKER).idle:
                    self.combinedActions.append(st.attack(defence_target))
                for ad in self.units(ADEPT).idle:
                    self.combinedActions.append(ad.attack(defence_target))
                for zl in self.units(ZEALOT).idle:
                    self.combinedActions.append(zl.attack(defence_target))

        # Attack!
        if self.charge_started > 0 and self.time - self.charge_started >= 90:
            if self.time > self.do_something_after:
                all_enemy_base = self.known_enemy_structures
                if all_enemy_base.exists and self.units(NEXUS).exists:
                    next_enemy_base = all_enemy_base.closest_to(self.units(NEXUS).first)
                    attack_target = next_enemy_base.position.random_on_distance(random.randrange(1, 5))
                    gather_target = next_enemy_base.position.towards(self.units(NEXUS).first.position, 40)
                elif all_enemy_base.exists:
                    next_enemy_base = all_enemy_base.closest_to(self.game_info.map_center)
                    attack_target = next_enemy_base.position.random_on_distance(random.randrange(1, 5))
                    gather_target = self.game_info.map_center.towards(self.enemy_start_locations[0], 17)
                else:
                    attack_target = self.game_info.map_center.random_on_distance(random.randrange(12, 70 + int(self.time / 60)))
                    gather_target = self.game_info.map_center.towards(self.enemy_start_locations[0], 20)
                if self.gathered and not self.first_attack:
                    for st in self.units(STALKER):
                        self.combinedActions.append(st.attack(attack_target))
                    for ad in self.units(ADEPT):
                        self.combinedActions.append(ad.attack(attack_target))
                    for vr in self.units(VOIDRAY):
                        self.combinedActions.append(vr.attack(attack_target))
                    for zl in self.units(ZEALOT):
                        self.combinedActions.append(zl.attack(attack_target))
                    self.first_attack = True
                    print('--- First Attack started --- @: ', self.time, 'with Stalkers: ',
                          len(self.units(STALKER)),
                          'and Voidrays: ', len(self.units(VOIDRAY)), 'and Adepts: ',
                          len(self.units(ADEPT)), 'and Zealots: ', len(self.units(ZEALOT)))
                if gather_target and not self.first_attack:
                    for st in self.units(STALKER):
                        self.combinedActions.append(st.attack(gather_target))
                    for ad in self.units(ADEPT):
                        self.combinedActions.append(ad.attack(gather_target))
                    for vr in self.units(VOIDRAY):
                        self.combinedActions.append(vr.attack(gather_target))
                    for zl in self.units(ZEALOT):
                        self.combinedActions.append(zl.attack(gather_target))
                    wait = 30
                    self.do_something_after = self.time + wait
                    self.gathered = True
                if self.first_attack:
                    for st in self.units(STALKER).idle:
                        self.combinedActions.append(st.attack(attack_target))
                    for ad in self.units(ADEPT).idle:
                        self.combinedActions.append(ad.attack(attack_target))
                    for vr in self.units(VOIDRAY).idle:
                        self.combinedActions.append(vr.attack(attack_target))
                    for zl in self.units(ZEALOT).idle:
                        self.combinedActions.append(zl.attack(attack_target))

        # seek & destroy
        if self.first_attack and not self.known_enemy_structures.exists and self.time > self.do_something_after:

            for vr in self.units(VOIDRAY).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(vr.attack(attack_target))
            for st in self.units(STALKER).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(st.attack(attack_target))
            for zl in self.units(ZEALOT).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(zl.attack(attack_target))
            for ad in self.units(ADEPT).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(ad.attack(attack_target))
            self.do_something_after = self.time + 5

        # execuite actions
        await self.do_actions(self.combinedActions)

    async def destroy_lifted_buildings(self):
        if (len(self.units(GATEWAY)) + len(self.units(WARPGATE))) < self.MAX_GATES:
            if self.can_afford(GATEWAY):
                await self.build(GATEWAY, near=self.units(PYLON).ready.random)

        if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE):
            if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                await self.build(CYBERNETICSCORE, near=self.units(PYLON).ready.random)

        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            if self.units(CYBERNETICSCORE).ready.exists and self.can_afford(STARGATE) and not self.already_pending(STARGATE) and len(self.units(STARGATE).ready) < 2:
                        await self.build(STARGATE, near=pylon)

        if self.supply_left > 3:
            for sg in self.units(STARGATE).ready.noqueue:
                if self.can_afford(VOIDRAY):
                    await self.do(sg.train(VOIDRAY))

        elif self.supply_used > 196:
            if self.units(ZEALOT).exists:
                target = self.units(ZEALOT).random
            elif self.units(ADEPT).exists:
                target = self.units(ADEPT).random
            elif self.units(IMMORTAL).exists:
                target = self.units(IMMORTAL).random
            elif self.units(COLOSSUS).exists:
                target = self.units(COLOSSUS).random
            elif self.units(SENTRY).exists:
                target = self.units(SENTRY).random
            elif self.units(STALKER).exists:
                target = self.units(STALKER).random
            else:
                target = self.game_info.map_center

            for st in self.units(STALKER):
                self.combinedActions.append(st.attack(target))
            for se in self.units(SENTRY):
                self.combinedActions.append(se.attack(target))
            for zl in self.units(ZEALOT):
                self.combinedActions.append(zl.attack(target))
            for ad in self.units(ADEPT):
                self.combinedActions.append(ad.attack(target))

        if self.supply_used > 70:
            if self.time > self.do_something_after:
                all_enemy_base = self.known_enemy_structures
                if all_enemy_base.exists and self.units(NEXUS).exists:
                    next_enemy_base = all_enemy_base.closest_to(self.units(NEXUS).first)
                    attack_target = next_enemy_base.position.random_on_distance(random.randrange(1, 5))
                    gather_target = next_enemy_base.position.towards(self.units(NEXUS).first.position, 40)
                else:
                    attack_target = self.enemy_start_locations[0].towards(self.game_info.map_center,
                                                                          random.randrange(15, 20))
                    gather_target = self.game_info.map_center.towards(self.enemy_start_locations[0], 20)
                if self.gathered and not self.final_attack:
                    for st in self.units(STALKER):
                        self.combinedActions.append(st.attack(attack_target))
                    for vr in self.units(VOIDRAY):
                        self.combinedActions.append(vr.attack(attack_target))
                    for se in self.units(SENTRY):
                        self.combinedActions.append(se.attack(attack_target))
                    for zl in self.units(ZEALOT):
                        self.combinedActions.append(zl.attack(attack_target))
                    for ad in self.units(ADEPT):
                        self.combinedActions.append(ad.attack(attack_target))
                    self.final_attack = True
                    print('--- Final Attack started --- @: ', self.time, 'with Stalkers: ', len(self.units(STALKER)), 'and Adepts: ', len(self.units(ADEPT)), 'and Voidrays: ', len(self.units(VOIDRAY)))
                if gather_target and not self.final_attack:
                    for st in self.units(STALKER):
                        self.combinedActions.append(st.attack(gather_target))
                    for vr in self.units(VOIDRAY):
                        self.combinedActions.append(vr.attack(gather_target))
                    for se in self.units(SENTRY):
                        self.combinedActions.append(se.attack(gather_target))
                    for zl in self.units(ZEALOT):
                        self.combinedActions.append(zl.attack(gather_target))
                    for ad in self.units(ADEPT):
                        self.combinedActions.append(ad.attack(gather_target))
                    wait = 30
                    self.do_something_after = self.time + wait
                    self.gathered = True
                if self.final_attack:
                    for st in self.units(STALKER).idle:
                        self.combinedActions.append(st.attack(attack_target))
                    for vr in self.units(VOIDRAY).idle:
                        self.combinedActions.append(vr.attack(attack_target))
                    for se in self.units(SENTRY).idle:
                        self.combinedActions.append(se.attack(attack_target))
                    for zl in self.units(ZEALOT).idle:
                        self.combinedActions.append(zl.attack(attack_target))
                    for ad in self.units(ADEPT).idle:
                        self.combinedActions.append(ad.attack(attack_target))

        # seek & destroy
        if self.final_attack and not self.known_enemy_structures.exists and self.time > self.do_something_after:

            for se in self.units(SENTRY).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(se.attack(attack_target))
            for vr in self.units(VOIDRAY).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(vr.attack(attack_target))
            for st in self.units(STALKER).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(st.attack(attack_target))
            for zl in self.units(ZEALOT).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(zl.attack(attack_target))
            for ad in self.units(ADEPT).idle:
                attack_target = self.game_info.map_center.random_on_distance(
                    random.randrange(12, 70 + int(self.time / 60)))
                self.combinedActions.append(ad.attack(attack_target))
            self.do_something_after = self.time + 5

        # execuite actions
        await self.do_actions(self.combinedActions)

    async def distribute_workers(self):
        """
        Distributes workers across all the bases taken.
        WARNING: This is quite slow when there are lots of workers or multiple bases.
        Customized
        """

        # TODO:
        # OPTIMIZE: Assign idle workers smarter
        # OPTIMIZE: Never use same worker mutltiple times

        expansion_locations = self.expansion_locations
        owned_expansions = self.owned_expansions
        worker_pool = []
        for idle_worker in self.workers.idle:
            if len(self.units(NEXUS).ready) == 1:
                mf = self.state.mineral_field.closest_to(self.units(NEXUS).first)
            else:
                mf = self.state.mineral_field.closest_to(idle_worker)
            self.combinedActions.append(idle_worker.gather(mf))


        for location, townhall in owned_expansions.items():
            workers = self.workers.closer_than(20, location)
            actual = townhall.assigned_harvesters
            ideal = townhall.ideal_harvesters
            excess = actual - ideal
            if actual > ideal:
                worker_pool.extend(workers.random_group_of(min(excess, len(workers))))
                continue
        for g in self.geysers:
            workers = self.workers.closer_than(5, g)
            actual = g.assigned_harvesters
            ideal = g.ideal_harvesters
            excess = actual - ideal
            if actual > ideal:
                worker_pool.extend(workers.random_group_of(min(excess, len(workers))))
                continue

        for g in self.geysers:
            actual = g.assigned_harvesters
            ideal = g.ideal_harvesters
            deficit = ideal - actual

            for x in range(0, deficit):
                if worker_pool:
                    w = worker_pool.pop()
                    if len(w.orders) == 1 and w.orders[0].ability.id in [HARVEST_RETURN]:
                        await self.do(w.move(g))
                        await self.do(w.return_resource(queue=True))
                    else:
                        await self.do(w.gather(g))

        for location, townhall in owned_expansions.items():
            actual = townhall.assigned_harvesters
            ideal = townhall.ideal_harvesters

            deficit = ideal - actual
            for x in range(0, deficit):
                if worker_pool:
                    w = worker_pool.pop()
                    mf = self.state.mineral_field.closest_to(townhall)
                    if len(w.orders) == 1 and w.orders[0].ability.id in [HARVEST_RETURN]:
                        await self.do(w.move(townhall))
                        await self.do(w.return_resource(queue=True))
                        await self.do(w.gather(mf, queue=True))
                    else:
                        await self.do(w.gather(mf))

        await self.do_actions(self.combinedActions)

    # Find enemy natural expansion location (Thanks @ CannonLover)
    async def find_enemy_natural(self):
        closest = None
        distance = math.inf
        for el in self.expansion_locations:
            def is_near_to_expansion(t):
                return t.position.distance_to(el) < self.EXPANSION_GAP_THRESHOLD

            if is_near_to_expansion(sc2.position.Point2(self.enemy_start_locations[0])):
                continue

            #if any(map(is_near_to_expansion, )):
                # already taken
            #    continue

            d = await self._client.query_pathing(self.enemy_start_locations[0], el)
            if d is None:
                continue

            if d < distance:
                distance = d
                closest = el

        return closest
