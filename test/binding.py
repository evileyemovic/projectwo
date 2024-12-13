import math
import numpy as np
from rlcard.games.base import Card


suit_list = ['S','H','D','C']
rank_list = ['A','1','2','3','4','5','6','7','8','9','10','J','Q','K']

matrix1 = [suit_list,rank_list]
cards = ''
cardr = ''

for s1 in matrix1[0]:
    cards = s1
    for r1 in matrix1[1]:
        cardr = r1
        cardm1 = Card(s1,r1)
        for s2 in matrix1[0]:
            cards = s2
            for r2 in matrix1[1]:
                cardr = r2
                cardm2 = Card(s2,r2)
                if cardm1 == cardm2:
                    continue
                else:
                    best_action = agent.eval_step()
                    



