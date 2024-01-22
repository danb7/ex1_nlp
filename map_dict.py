import numpy as np

#Topically related
run_topically_w2v_dict = {
    "runs" : 1, "running" : 1, "drive" : 0, "ran" : 1, "scamper" : 1, "tworun_double" : 0, "go" : 1,
    "twoout" : 0, "walk" : 1, "Mark_Grudzielanek_singled" : 1, "Batterymate_Miguel_Olivo" : 1, "homerun" : 1,
    "threerun" : 1, "Collin_Salzenstein" : 1, "basesloaded" : 0, "fielder's_choice_grounder" : 0,
    "Peter_Bourjos_tripled" : 1, "Casey_Kalenkosky" : 1, "Scutaro_singled" : 1, "clubbed_solo_homer" : 0
}
run_topically_gpt_dict = {
    "Sprint" : 1, "Jog" : 1, "Race" : 1, "Dash" : 1, "Rush" : 1, "Marathon" : 1, "Gallop" : 1,
    "Trot" : 1, "Pace" : 1, "Stride" : 1, "Jogger" : 1, "Sprinter" : 1, "Hurdle" : 1,
    "Bolt" : 1, "Scamper" : 1, "Hasten" : 1, "Hasten" : 1, "Hurry" : 1, "Propel" : 1, "Zoom" : 0
}
game_topically_w2v_dict = {
    "games" : 1, "play" : 1, "match" : 1, "matchup" : 1, "agame" : 1, "ballgame" : 1, 
    "thegame" : 1, "opener" : 1, "matches" : 1, "tournament" : 1, "playing" : 1,
    "league" : 1, "Game" : 1, "scrimmages" : 0, "fourgame" : 1, "scrimmage" : 0,
    "postseason" : 1, "playoffs" : 1, "gme" : 1, "season" : 1
}
game_topically_gpt_dict = {
    "Play" : 1, "Sport" : 1, "Competition" : 1, "Match" : 1, "Recreation" : 0, "Activity" : 0,
    "Challenge" : 0, "Gaming" : 1, "Contest" : 1, "Pastime" : 0, "Amusement" : 0, "Puzzle" : 1,
    "Strategy" : 0, "Tournament" : 1, "Leisure" : 0, "Entertainment" : 1, "Plaything" : 0,
    "Rivalry" : 0, "Recreation" : 0, "Contest" : 1
}
#Same semantic class
run_semantic_w2v_dict = {
    "runs" : 1, "running" : 1, "drive" : 0, "ran" : 1, "scamper" : 1, "tworun_double" : 1, "go" : 1,
    "twoout" : 0, "walk" : 1, "Mark_Grudzielanek_singled" : 0, "Batterymate_Miguel_Olivo" : 0, "homerun" : 1,
    "threerun" : 0, "Collin_Salzenstein" : 0, "basesloaded" : 0, "fielder's_choice_grounder" : 0,
    "Peter_Bourjos_tripled" : 0, "Casey_Kalenkosky" : 0, "Scutaro_singled" : 0, "clubbed_solo_homer" : 0
}
run_semantic_gpt_dict = {
    "Sprint" : 1, "Jog" : 1, "Race" : 0, "Dash" : 1, "Rush" : 1, "Marathon" : 0, "Gallop" : 0,
    "Trot" : 1, "Pace" : 1, "Stride" : 0, "Jogger" : 1, "Sprinter" : 1, "Hurdle" : 0,
    "Bolt" : 0, "Scamper" : 0, "Hasten" : 1, "Hasten" : 1, "Hurry" : 1, "Propel" : 0, "Zoom" : 0
}
game_semantic_w2v_dict = {
    "games" : 1, "play" : 1, "match" : 1, "matchup" : 1, "agame" : 1, "ballgame" : 1, 
    "thegame" : 1, "opener" : 0, "matches" : 1, "tournament" : 0, "playing" : 0, 
    "league" : 0, "Game" : 1, "scrimmages" : 0, "fourgame" : 1, "scrimmage" : 0, 
    "postseason" : 0, "playoffs" : 0, "gme" : 1, "season" : 0
}
game_semantic_gpt_dict = {
    "Play" : 1, "Sport" : 0, "Competition" : 0, "Match" : 1, "Recreation" : 0, "Activity" : 0,
    "Challenge" : 0, "Gaming" : 1, "Contest" : 0, "Pastime" : 0, "Amusement" : 0, "Puzzle" : 0,
    "Strategy" : 0, "Tournament" : 0, "Leisure" : 0, "Entertainment" : 0, "Plaything" : 0,
    "Rivalry" : 0, "Recreation" : 0, "Contest" : 0
}

topically_w2v_list = [np.array(list(run_topically_w2v_dict.values())), np.array(list(game_topically_w2v_dict.values()))]
topically_gpt_list = [np.array(list(run_topically_gpt_dict.values())), np.array(list(game_topically_gpt_dict.values()))]
semantic_w2v_list  = [np.array(list(run_semantic_w2v_dict.values())), np.array(list(game_semantic_w2v_dict.values()))]
semantic_gpt_list  = [np.array(list(run_semantic_gpt_dict.values())), np.array(list(game_semantic_gpt_dict.values()))]