from collections import defaultdict as dd
from parse import Parser
from agent import Agent
from reward import Rewarder
import numpy as np
from numpy.random import shuffle

# agent settings
HIDDEN = [500]

# game settings
PLAYERS = 2 # between 2 to 5
GAMEMODE = 0 # 0:og, 1:6suit
CLOCKS = 8
BOMBS = 3

# Moves / Operations
PLAY = 0
DISCARD = 1
INFORM = 2

# R, Y, G, W, B, M = 0, 1, 2, 3, 4, 5
if GAMEMODE == 0:
    COLORS = ['r', 'y', 'g', 'w', 'b']
elif GAMEMODE == 1:
    COLORS = ['r', 'y', 'g', 'w', 'b', 'm'] # 'm' is the multicolored suit

NCOLS = len(COLORS)
NUMBERS = ['1', '2', '3', '4', '5']
NNUMS = len(NUMBERS)
NNUMBERCARDS = {'1': 3, '2': 2, '3': 2, '4': 2, '5': 1}
COLNUMS = COLORS + NUMBERS

def generate_deck(gamemode):
    deck = []
    for i, n in NNUMBERCARDS.items():
        deck += [c+i for c in COLORS*n]

    return deck

class Game():
    def __init__(self, nPlayers, gamemode, clocks, bombs):
        self.nPlayers = nPlayers
        self.gamemode = gamemode
        self.clocks = clocks
        self.bombs = bombs
        self.nStartingCards = 5 if nPlayers-3<=0 else 4
        self.hands = [[] for i in range(nPlayers)] # cards of the player
        self.memories = [['xx']*self.nStartingCards for i in range(nPlayers)] # what the player knows (in the form of ['xx','xx',...] where 'x' means the player knows nothing)
        self.posInfos = [[] for i in range(nPlayers)] # possible information that can be given to this player 
        self.deck = generate_deck(gamemode)
        self.table = dict([(c, []) for c in COLORS]) # gamemode 0 has 5 suits, 1 has 6 suits
        self.parser = Parser(COLORS, nPlayers, self.nStartingCards)
        self.inputLen = NNUMS*NCOLS + (NNUMS+NCOLS)*self.nStartingCards*(nPlayers*2 - 1) + 5
        self.outputLen = self.nStartingCards*2 + (NNUMS+NCOLS)*(nPlayers-1)
        self.agents = [Agent(self.inputLen, HIDDEN, self.outputLen) for i in range(nPlayers)]
        self.totScore = 0
        self.suitScores = [0 for i in range(NCOLS)]
        self.actionsLookup = {'play': [i for i in range(self.nStartingCards)], 
                              'discard': [i for i in range(self.nStartingCards, self.nStartingCards*2)],
                              'inform': [i for i in range(self.nStartingCards*2, self.nStartingCards*2 + (nPlayers-1)*(NNUMS+NCOLS))]}
        self.gameOver = True
        self.rewarder = Rewarder(gamemode)
    
    def play(self):
        self.reset(CLOCKS, BOMBS)
        self.setup()
        rounds = -1
        while not self.gameOver:
            rounds += 1
            pi = 0
            while pi < self.nPlayers:
                # print("ROUND", rounds, "player", pi)
                agentActionCompleted = False
                agent = self.agents[pi]
                # check gameover
                if self.checkGameOver():
                    self.gameOver = True
                    break
                
                X = [self.getGameStateArray(pi)]
                action, y, p = agent.choice(X)

                # print(action)

                if action in self.actionsLookup['play']:
                    # player pi plays the action-th card in their hand
                    status, cardPlayed, mem = self.playCard(pi, action)
                    
                    self.rewarder.rewardCardPlayed(self.agents[pi], action, status, cardPlayed, mem)                    
                   
                    if status != -1: # status -1 when attempt to play a card out of index, need to resample action
                        agentActionCompleted = True
                
                elif action in self.actionsLookup['discard']:
                    # player pi discards the (discard.index(action))th card in their hand
                    ci = self.actionsLookup['discard'].index(action)
                    # print("DISCARDING", pi, ci)
                    # print("ACTION IS", action)
                    # print(self.actionsLookup['discard'])
                    agentActionCompleted, dmem, dcard, clocksGained = self.discard(pi, ci)

                    self.rewarder.rewardDiscard(self.agents[pi], action, agentActionCompleted, dmem, dcard, clocksGained)

                elif action in self.actionsLookup['inform']:
                    # player pi informs 
                    lookupI = self.actionsLookup['inform'].index(action)
                    reltpi = lookupI // (NNUMS+NCOLS)
                    tmp = [i for i in range(self.nPlayers)]
                    tmp.pop(pi)
                    tpi = tmp[reltpi]
                    info = COLNUMS[lookupI % (NNUMS+NCOLS)]
                    tpiPosInfo = [inf[0] for inf in self.posInfos[tpi]]
                    # print("INFORMING ACTION", action)
                    # print(self.actionsLookup['inform'])
                    # print("lookupI is", lookupI)
                    # print("tpi:", tpi, "pi", pi)
                    # print("info", info)
                    # print("tpiposinfo", tpiPosInfo)
                    tpiPosInfoTups = self.posInfos[tpi]
                    infoValue = 0.5

                    if info not in tpiPosInfo:
                        # TODO infoValue calculation
                        infoValue -= 100
                        # agent chose to inform something that cannot be informed
                        # print("error: agent chose to inform {} when tpi: {} info is".format(info, tpi), self.posInfos[tpi])
                        self.rewarder.rewardInfo(self.agents[pi], self.agents[tpi], action, infoValue)

                    else:
                        self.inform(pi, tpi, tpiPosInfo.index(info))
                        self.rewarder.rewardInfo(self.agents[pi], self.agents[tpi], action, infoValue)
                        agentActionCompleted = True
                
                # move on to the next player only if valid action is made, otherwise retry
                if agentActionCompleted:
                    pi += 1
                        

    def checkGameOver(self):
        if self.bombs == 0:
            # print("boom, bombs 0")
            return True
        elif self.totScore == 25:
            # print("score 25")
            return True
        elif self.gameIsStuck():
            # print("game is stuck")
            return True
        else:
            return False

    def gameIsStuck(self):
        return (self.noCardsPlayable() and len(self.deck)==0)

    def noCardsPlayable(self):
        stuck = True
        for hand in self.hands:
            for card in hand:
                if self.validPlay(card):
                    stuck = False
        return stuck

    def reset(self, clocks, bombs):
        self.clocks = clocks
        self.bombs = bombs
        self.hands = [[] for i in range(self.nPlayers)]
        self.memories = [['xx']*self.nStartingCards for i in range(self.nPlayers)]
        self.posInfos = [[] for i in range(self.nPlayers)]
        self.deck = generate_deck(self.gamemode)
        self.table = dict([(c, []) for c in COLORS])
        self.totScore = 0
        self.suitScores = [0 for i in range(NCOLS)]
        self.gameOver = True
        # self.table = [[] for i in range(self.gamemode + 5)]

    def setupDeal(self):
        for i in range(self.nStartingCards):
            for hand in self.hands:
                hand.append(self.deck.pop())

    def setup(self):
        self.gameOver = False
        shuffle(self.deck)
        self.setupDeal()
        for i in range(self.nPlayers): 
            self.updatePosInfo(i)

    def updateScore(self):
        suit_scores = []
        tot_score = 0
        for suit in self.table.values():
            suit_score = len(suit)
            suit_scores.append(suit_score)
            tot_score += suit_score
        
        self.suitScores = suit_scores
        self.totScore = tot_score
        return suit_scores, tot_score

    def updatePosInfo(self, i):
        # update player i's possible information
        self.posInfos[i] = self.evalInfo(self.hands[i])

    def evalInfo(self, hand):
        # information in the form of (indexes, info)
        # where info in ['r','y','g',...,'1','2',...'5']
        d = dd()
        for i, card in enumerate(hand):
            for j in range(2):
                if card[j] in d.keys():
                    d[card[j]] += [i]
                else:
                    d[card[j]] = [i]
        return list(d.items())

    def draw(self, pi):
        # player pi draws a card if the deck isn't empty
        if len(self.deck) > 0:
            self.hands[pi].append(self.deck.pop())
            self.memories[pi].append('xx')

    def cardIsAccessible(self, pi, ci):
        # penalise the agent if it is trying to index a card out of their hand
        if len(self.hands[pi]) <= ci:
            # ATTEMPT TO ACCESS AN INDEX THAT'S NOT IN THE HAND
            return False
            # todo add penalty
        else:
            return True

    def discard(self, pi, ci):
        # player pi discards their ci-th card
        # returns validAction, memory and card of the discard, and clocks gained
        clocksGained = False
        if not self.cardIsAccessible(pi, ci):
            # ATTEMPT TO DISCARD AN INDEX THAT'S NOT IN THE HAND
            return False, None, None, clocksGained
        dmem = self.memories[pi].pop(ci)
        dcard = self.hands[pi].pop(ci)

        # add the clock
        clocksGained = self.clocks<CLOCKS
        self.clocks += 1 if clocksGained else 0

        # draw a card and update their possible info
        self.draw(pi)
        self.updatePosInfo(pi)

        return True, dmem, dcard, clocksGained

    def validPlay(self, card):
        suitcards = len(self.table[card[0]])
        cardnum = int(card[1])
        # a playable card is always one higher than the highest card of that suit on the table
        return True if cardnum-suitcards==1 else False

    def playCard(self, pi, ci):
        # play player pi's ci-th card
        if not self.cardIsAccessible(pi, ci):
            # ATTEMPT TO PLAY AN INDEX THAT'S NOT IN THE HAND
            return -1, None, None
        card = self.hands[pi].pop(ci) # remove the card from the hand
        pmem = self.memories[pi].pop(ci) # remove memory of that card
        self.draw(pi) # add a card back
        self.updatePosInfo(pi) # update possible info
        playable = self.validPlay(card) # check if the card is playable
        if playable:
            self.table[card[0]].append(card) # add the card to corresponding suit on the table
            
            if card[1] == '5':
                # restore a clock for completing a set
                self.clocks += 1 if self.clocks<CLOCKS else 0

            self.updateScore() # update the score
            return 1, card, pmem

        else:
            # print("play error")
            self.bombs -= 1
            return 0, card, pmem

    def inform(self, fpi, tpi, infoi):
        # fpi gives tpi's infoi-th info

        # check if there are enough clocks
        if self.clocks == 0:
            # attempt to inform when there are no clock tokens left
            # todo penalise
            # print("attempt to inform when clocks are 0")
            return False

        self.clocks -= 1 if self.clocks>0 else 0
        info, info_indexes = self.posInfos[tpi][infoi]
        memI = 0 if info in COLORS else 1
        for index in info_indexes:
            if memI == 0:
                # color info
                self.memories[tpi][index] = info+self.memories[tpi][index][1:]
            else:
                # number info
                self.memories[tpi][index] = self.memories[tpi][index][:1]+info

    def getGameStateArray(self, pi):
        # converts game state to an input for agent pi
        concealed_hands = self.hands[:pi] + self.hands[pi+1:]
        X = self.parser.parse_all(self.table, concealed_hands, self.memories,
                                  self.nPlayers, self.clocks, self.bombs,
                                  self.gamemode, self.nStartingCards)
        return X

game = Game(PLAYERS, GAMEMODE, CLOCKS, BOMBS)
game.play()

for i in range(1000):
    if i%500 == 0:
        print(i)
    game.play()
# game.setup()
# print(game.hands)
# print(game.posInfos)
# game.inform(0,1,0)
# print(game.memories)
# 
# X = game.getGameStateArray(0)
# print(len(X))
# print(game.inputLen)
# o = game.agents[0].choice(X)
# print(game.posInfos)

