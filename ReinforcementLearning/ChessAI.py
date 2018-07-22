import MCTS.Montecarlo_Tree_Search as Monte
from NeuralNetwork.Networks import Networks
import tensorflow as tf



class ChessAI :
    def __init__(self,path):
        self.networks = Networks(path)
        self.monte = Monte.MontecarloTreeSearch(self.networks)
        self.decision = None

    def ask(self, Board):
        turn = Board.turn
        self.monte.set_state(Board)
        self.analyze()
        return self.decision
    def refresh(self,move):
        self.monte.inherit(move)
    def analyze(self):
        self.decision = self.monte.predict()

    def get_MCTS(self, chessBoard):
        nextMove = self.monte.MCTS(chessBoard)

        return nextMove

    def getNetwork(self):
        return self.monte.getNetwork

