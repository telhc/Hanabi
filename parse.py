import numpy as np

NNUMS = 5

class Parser():
    def __init__(self, colorsList, nPlayers, startingCards):
        self.colorsList = colorsList
        self.nColors = len(colorsList)
        self.nPlayers = nPlayers
        self.startingCards = startingCards
        self.cardLookUp = dict(zip(colorsList+[str(i) for i in range(1,6)],[i for i in range(self.nColors+NNUMS)]))
        self.ARSize = self.nColors+NNUMS

    def parse_table(self, table):
        # accepts table as {'r': [], 'y': [], ...}
        X = np.zeros(self.nColors * NNUMS)
        for i, l in enumerate([len(v) for v in table.values()]):
            X[i*NNUMS:(i+1)*NNUMS] = np.hstack((np.ones(l), np.zeros(NNUMS-l)))
        return X

    def parse_card(self, card):
        # accepts card in 'cn'
        C = np.zeros(self.ARSize)
        for char in card:
            if char!='x':
                C[self.cardLookUp[char]] = 1
        return C

    def parse_hand(self, hand):
        # accepts hand in ['cn','cn'...]
        H = np.zeros(self.startingCards * self.ARSize)
        for i, card in enumerate(hand):
            H[i*self.ARSize:(i+1)*self.ARSize] = self.parse_card(card)
        return H

    def parse_all(self, table, hands, memories, nPlayers, clocks, bombs, gamemode, startingC):
        T = self.parse_table(table)
        Hs = np.array([self.parse_hand(hand) for hand in hands])
        Ms = np.array([self.parse_hand(mem) for mem in memories])
        G = np.array([nPlayers, clocks, bombs, gamemode, startingC])
        X = np.hstack((T, Hs.flatten(), Ms.flatten(), G))
        return X
        
#    def parse_info(self, infoList):
#        # accepts list of info in (info, [i1, i2...])
#        I = np.zeros(self.ARSize * self.startingCards)
#        for i, info, indexes in enumerate(infoList):

