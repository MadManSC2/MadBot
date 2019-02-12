# MadBot
Starcraft 2 Bot written in Python using the python-sc2 libary
In order to run the Bot you will also need tensorflow and keras. Alternatively, you can run a specific build of the Bot by commenting out the neural network parts and comment in the static decision lines.

I started this project to learn both python and machine learning (ML) at the same time.
This means that not only my code might be messy or sub-optimal, but also my ML techniques might be pretty basic.
Yet, I tried to create a bot, that mostly relies on hard-coded build order, which are good in particular situations or against certain races.
This version was meant to randomly choose a build order and gather scouting data at approximately 2 minutes elapsed game time, which is stored together with the result of the game.
This training data will be used to train a neural network to choose the optimal build order based on the scouting information of the game instead of randomly choosing it right at the start.
Hopefully, this will increase win rates a lot (it certainly did against the build in computer opponent, but thats a different story).
The next step would be to identify more milestones within a typical game in which major choices had to be made and let a different network make these choices.
Possible candidates might be: Attack now or keep defending/expanding, alter unit composition (e.g. Colossi instead of Immortals), Commit to a attack or retreat, etc.
Thanks to sentdex, CreepyBot, Cannon-Lover and TapiocaBot for some inspiration
