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

    def analyze(self):
        self.decision = self.monte.predict()

