import numpy as np

#Topically related
spy_topically_w2v_dict = {
    "spies" : 1, "spying" : 1, "espionage" : 1, "spymaster" : 1, "CIA" : 1, "Spy" : 1,
    "MI6" : 1, "spymasters" : 1, "intelligence" : 1, "CIA_operative" : 1, "eavesdropping" : 1,
    "covert" : 1, "persecute_dissidents" : 0, "counterspy" : 1, "counterintelligence" : 1,
    "supersecret" : 1, "counterspies" : 1, "KGB" : 1, "honeytrap" : 1, "superspy" : 1
}
spy_topically_gpt_dict = {
    "Agent" : 1, "Operative" : 1, "Infiltrator" : 1, "Mole" : 1, "Sleuth" : 1, "Spook" : 1,
    "Informant" : 1, "Saboteur" : 1, "Covert" : 1, "Espionage" : 1, "Surveillance" : 1,
    "Undercover" : 1, "Intelligence" : 1, "Detective" : 1, "Reconnaissance" : 1, "Secret" : 1,
    "Cryptanalyst" : 1, "Infiltration" : 1, "Stealth" : 1, "Clandestine" : 1
}
game_topically_w2v_dict = {
    "games" : 1, "play" : 1, "match" : 1, "matchup" : 1, "agame" : 1, "ballgame" : 1,
    "thegame" : 1, "opener" : 1, "matches" : 1, "tournament" : 1, "playing" : 1,
    "league" : 1,"Game" : 1, "scrimmages" : 0, "fourgame" : 1, "scrimmage" : 0,
    "postseason" : 1, "playoffs" : 1, "gme" : 1, "season" : 1
}
game_topically_gpt_dict = {
    "play" : 1, "match" : 1, "sport" : 1, "contest" : 1, "competition" : 1, "recreation" : 1,
    "amusement" : 1, "pastime" : 1, "entertainment" : 1, "activity" : 1, "diversion" : 1,
    "event" : 0, "challenge" : 1, "fun" : 1, "pursuit" : 0, "gaming" : 1, "exercise" : 1,
    "playtime" : 1, "frolic" : 1, "scrimmage" : 0
}
#Same semantic class
spy_semantic_w2v_dict = {
    "spies" : 1, "spying" : 1, "espionage" : 1, "spymaster" : 1, "CIA" : 0, "Spy" : 1,
    "MI6" : 0, "spymasters" : 1, "intelligence" : 0, "CIA_operative" : 0, "eavesdropping" : 0,
    "covert" : 1, "persecute_dissidents" : 0, "counterspy" : 1, "counterintelligence" : 0,
    "supersecret" : 0, "counterspies" : 1, "KGB" : 0, "honeytrap" : 0, "superspy" : 1
}
spy_semantic_gpt_dict = {
    "Agent" : 1, "Operative" : 1, "Infiltrator" : 1, "Mole" : 1, "Sleuth" : 1, "Spook" : 1,
    "Informant" : 0, "Saboteur" : 0, "Covert" : 0, "Espionage" : 1, "Surveillance" : 0,
    "Undercover" : 1, "Intelligence" : 0, "Detective" : 1, "Reconnaissance" : 0, "Secret" : 0,
    "Cryptanalyst" : 0, "Infiltration" : 0, "Stealth" : 0, "Clandestine" : 0
}
game_semantic_w2v_dict = {
    "games" : 1, "play" : 1, "match" : 1, "matchup" : 0, "agame" : 1, "ballgame" : 1,
    "thegame" : 1, "opener" : 0, "matches" : 1, "tournament" : 0, "playing" : 0,
    "league" : 0, "Game" : 1, "scrimmages" : 0, "fourgame" : 1, "scrimmage" : 0,
    "postseason" : 0, "playoffs" : 0, "gme" : 1, "season" : 0
}
game_semantic_gpt_dict = {
    "play" : 1, "match" : 1, "sport" : 0, "contest" : 0, "competition" : 0, "recreation" : 0,
    "amusement" : 0, "pastime" : 0, "entertainment" : 0, "activity" : 0, "diversion" : 0,
    "event" : 0, "challenge" : 0, "fun" : 0, "pursuit" : 0, "gaming" : 1, "exercise" : 0,
    "playtime" : 1, "frolic" : 0, "scrimmage" : 0
}

topically_w2v_list = [np.array(list(spy_topically_w2v_dict.values())), np.array(list(game_topically_w2v_dict.values()))]
topically_gpt_list = [np.array(list(spy_topically_gpt_dict.values())), np.array(list(game_topically_gpt_dict.values()))]
semantic_w2v_list  = [np.array(list(spy_semantic_w2v_dict.values())), np.array(list(game_semantic_w2v_dict.values()))]
semantic_gpt_list  = [np.array(list(spy_semantic_gpt_dict.values())), np.array(list(game_semantic_gpt_dict.values()))]