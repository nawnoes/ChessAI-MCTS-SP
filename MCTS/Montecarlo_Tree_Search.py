import MCTS.Tree as TR
from NeuralNetwork.Networks import ValueNetwork as VN
from NeuralNetwork.Networks import Rollout as RO
from NeuralNetwork.Networks import PolicyNetwork as PN
import chess
import threading
import tensorflow as tf

class MontecarloTreeSearch():
    def __init__(self,path, searchRepeatNum=200, searchDepth = 10, expandPoint=1000):
        self.searchDepth = searchDepth
        self.expandPoint = expandPoint
        self.searchRepeatNum = searchRepeatNum
        self.evaluationQueue = []
        self.tree = TR.Tree(self.PolicyNetwork)


    def set_state(self,Board):
        self.tree.reset_board(Board)

    def MCTS(self,chessBoard):
        #몬테카를로 트리탐색을 통해 값을 얻기 전에
        #기존에 저장된 트리를 리셋해야함
        #추후에 트리 상속으로 개선
        self.tree.reset_board(chessBoard)
        print("몬테카를로 Search 시작")
        for i in range(self.searchRepeatNum):
            print("\r%d" % i , end="")
            self.search(chessBoard)
        nextMove = self.getNextMove()

        return nextMove

    def search(self,chessBoard):
        depth = 2
        self.tree.go_root(chessBoard)
        gameOver = self.tree.get_GameOver()
        job =[]
        selectionResult = False
        while not( gameOver or selectionResult):
            selectionResult = self.selection(depth)
            depth +=1
            gameOver = self.tree.get_GameOver()
        #selection이 끝난 후 트리가 가리키는 마지막 노드의 값을 Queue에 추가
        job.append(self.tree.get_CurrentNode())
        job.append(self.tree.get_currentBoard())
        self.evaluationQueue.append(job)
        print("잡 추가 ")
        if not gameOver:
            updateNode, rolloutResult , valueNetworkResult = self.evaluation()
            print("업데이트할 노드: ",updateNode," ",updateNode.get_W_rollout(), " ", updateNode.get_W_value())
            print("업데이트할 값  : ",rolloutResult,", ",valueNetworkResult)
            self.backpropagation(updateNode,rolloutResult , valueNetworkResult)
        else:
            #트리생성 중 게임이 종료되면 실제 결과를 적용
            realResult = self.tree.translatedResult()
            print("realResult: ",realResult)
            self.backpropagation(self.tree.get_CurrentNode(),realResult, realResult)


    def selection(self, depth):
        if depth > self.searchDepth:
           return True
        if depth == self.searchDepth and self.tree.currentNode.get_visit() >= self.expandPoint:
            #마지막 노드에서 확장 조건을 만족하면
            self.expansion()
            return True
        else:
            self.tree.makeNextChildByPolicyNetwork()
            # if True == self.tree.get_CurrentNode().get_Color():
            #     print("백")
            # else:
            #     print("흑")
            # self.tree.go_next()

        if depth == self.searchDepth:
            return True
        else:
            return False

    def evaluation(self):
        print("평가")
        #evaluationQueue에서 하나씩 평가 진행
        job = self.evaluationQueue.pop(0) # job[0]: 평가되어야할 노드, job[1]: 체스 보드
        updateNode = job[0]
        if updateNode.get_W_rollout() == 0 and updateNode.get_W_value() == 0:
            print("평가 새로 계산")
            valueNetworkResult = self.valueNetwork.get_ValueNetwork(job[1])
            rolloutResult = self.rolloutSimulation(job[1])
        else:
            print("평가 재사용")
            # 평가를 재사용하는 경우 값을 update 노드로 부터 받지 않아도 되지 않나 싶다
            valueNetworkResult= 0 #updateNode.get_W_value()
            rolloutResult= 0 #updateNode.get_W_rollout()

        return updateNode, rolloutResult, valueNetworkResult

    def expansion(self):
        #강화 학습된 정책망을 가지고
        #한단계 더 확장. 이때 가장 높은 정책망 값 선택
        self.tree.expand_RL_PolicyNetwork()

    def backpropagation(self,updateNode, rolloutResult, valueNetworkResult):
        parentNode = updateNode.get_Parent()
        if updateNode.is_root():
            return 0
        else:
            updateNode.renewForBackpropagation(rolloutResult, valueNetworkResult)
            return self.backpropagation(parentNode, rolloutResult, valueNetworkResult)


    def rolloutSimulation(self,chessBoard):
        simulationCount = 0
        tmpBoard = chessBoard.copy()
        while not tmpBoard.is_game_over():
            # print(simulationCount,end="")
            move = self.rollout.get_RolloutMove(tmpBoard)
            tmpBoard.push(chess.Move.from_uci(move))
            simulationCount +=1
            if simulationCount>10:
                print("롤아웃 결과: ", 0, " 시뮬레이션 수: ", simulationCount)
                return 0
        gameOutput = tmpBoard.result()
        #결과는 1-0, 1/2-1/2, 0-1로 나오므로 변환
        gameOutput = self.convertResult(gameOutput)
        print("롤아웃 결과: ", gameOutput," 시뮬레이션 수: ",simulationCount)
        return gameOutput
    def convertResult(self,result):
        rm = {'1-0': 1, '0-1': -1, '1/2-1/2': 0,'*': 0}
        # 게임의 끝, ( 백승 = 1, 흑승 = -1, 무승부, 0 )
        convertedResult = rm[result]
        return convertedResult
    def set_state(self, Board):
        self.tree.reset_board(Board)
    def getNextMove(self): #방문자 수가 가장 높은 다음 수 반환
        rootNode = self.tree.get_RootNode()
        index = rootNode.get_maxVisitedChildIndex()
        self.tree.root_Node.print_childInfo()
        return rootNode.child[index].command

if __name__ == "__main__":
    fens = ["r2q1rk1/pp1n2pp/2pbp1b1/3P1p2/2P1p2P/1P2P3/PB1NBPP1/2RQK2R b K - 0 14",
    "r2q1rk1/pp1n2pp/2pbp1b1/3P4/2P1pp1P/1P2P3/PB1NBPP1/2RQK2R w K - 0 15",
    "r2q1rk1/pp1n2pp/2pbp1b1/3P3P/2P1pp2/1P2P3/PB1NBPP1/2RQK2R b K - 0 15",
    "r2q1rk1/pp1n2pp/2pbp3/3P1b1P/2P1pp2/1P2P3/PB1NBPP1/2RQK2R w K - 1 16",
    "r2q1rk1/pp1n2pp/2pbp3/3P1b1P/2P1pP2/1P6/PB1NBPP1/2RQK2R b K - 0 16",
    "r2q1rk1/pp1n2pp/2p1p3/3P1b1P/2P1pb2/1P6/PB1NBPP1/2RQK2R w K - 0 17",
    "r2q1rk1/pp1n2pp/2p1p3/3P1b1P/2P1pb2/1P4P1/PB1NBP2/2RQK2R b K - 0 17"]

    mcts = MontecarloTreeSearch()

    for fen in fens:
        print("fen: ",fen)
        chessBoard = chess.Board(fen)
        nextMove = mcts.MCTS(chessBoard)
        print("몬테카를로 트리 탐색 결과 : ", nextMove)



