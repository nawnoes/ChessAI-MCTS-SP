import chess
import numpy as np
import tensorflow as tf
import os
import copy

from Support.Board2Array import Board2Array as B2A
from Support.OneHotEncoding import OneHotEncode as OHE

learning_rate=0.001

class Networks:
    def __init__(self,path):
        with tf.Graph().as_default():
            self.valueNetwork = ValueNetwork(path+'Value/')
            self.rollout = Rollout(path+'Rollout/')
            self.PolicyNetwork = PolicyNetwork(path+'Policy/')

class PolicyNetwork:
    def __init__(self,path):
        self.sess = tf.Session()
        self.batchSize=0

        policyNetworkName = "PN/"
        self.policyNetworkFilePath = path


        with tf.variable_scope("PN", reuse=False):
            self.X = tf.placeholder(tf.float32, [None, 8, 8, 36], name="X")  # 체스에서 8X8X10 이미지를 받기 위해 64
            self.K = tf.placeholder(tf.float32, [None], name="K")

            self.W1 = tf.get_variable("W1", shape=[5, 5, 36, 128], initializer=tf.contrib.layers.xavier_initializer())
            self.B1 = tf.get_variable("B1", initializer=tf.random_normal([128], stddev=0.01))
            self.L1 = tf.nn.relu(tf.nn.conv2d(self.X, self.W1, strides=[1, 1, 1, 1], padding='SAME') + self.B1)

            self.W2 = tf.get_variable("W2", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
            self.B2 = tf.get_variable("B2", initializer=tf.random_normal([128], stddev=0.01))
            self.L2 = tf.nn.relu(tf.nn.conv2d(self.L1, self.W2, strides=[1, 1, 1, 1], padding='SAME') + self.B2)

            self.W3 = tf.get_variable("W3", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
            self.B3 = tf.get_variable("B3", initializer=tf.random_normal([128], stddev=0.01))
            self.L3 = tf.nn.relu(tf.nn.conv2d(self.L2, self.W3, strides=[1, 1, 1, 1], padding='SAME') + self.B3)

            self.W4 = tf.get_variable("W4", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
            self.B4 = tf.get_variable("B4", initializer=tf.random_normal([128], stddev=0.01))
            self.L4 = tf.nn.relu(tf.nn.conv2d(self.L3, self.W4, strides=[1, 1, 1, 1], padding='SAME') + self.B4)

            self.W5 = tf.get_variable("W5", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
            self.B5 = tf.get_variable("B5", initializer=tf.random_normal([128], stddev=0.01))
            self.L5 = tf.nn.relu(tf.nn.conv2d(self.L4, self.W5, strides=[1, 1, 1, 1], padding='SAME') + self.B5)

            self.W6 = tf.get_variable("W6", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
            self.B6 = tf.get_variable("B6", initializer=tf.random_normal([128], stddev=0.01))
            self.L6 = tf.nn.relu(tf.nn.conv2d(self.L5, self.W6, strides=[1, 1, 1, 1], padding='SAME') + self.B6)

            self.W7 = tf.get_variable("W7", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
            self.B7 = tf.get_variable("B7", initializer=tf.random_normal([128], stddev=0.01))
            self.L7 = tf.nn.relu(tf.nn.conv2d(self.L6, self.W7, strides=[1, 1, 1, 1], padding='SAME') + self.B7)

            self.W8 = tf.get_variable("W8", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
            self.B8 = tf.get_variable("B8", initializer=tf.random_normal([128], stddev=0.01))
            self.L8 = tf.nn.relu(tf.nn.conv2d(self.L7, self.W8, strides=[1, 1, 1, 1], padding='SAME') + self.B8)

            self.W9 = tf.get_variable("W9", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
            self.B9 = tf.get_variable("B9", initializer=tf.random_normal([128], stddev=0.01))
            self.L9 = tf.nn.relu(tf.nn.conv2d(self.L8, self.W9, strides=[1, 1, 1, 1], padding='SAME') + self.B9)

            self.W10 = tf.get_variable("W10", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
            self.B10 = tf.get_variable("B10", initializer=tf.random_normal([128], stddev=0.01))
            self.L10 = tf.nn.relu(tf.nn.conv2d(self.L9, self.W10, strides=[1, 1, 1, 1], padding='SAME') + self.B10)

            self.W11 = tf.get_variable("W11", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
            self.B11 = tf.get_variable("B11", initializer=tf.random_normal([128], stddev=0.01))
            self.L11 = tf.nn.relu(tf.nn.conv2d(self.L10, self.W11, strides=[1, 1, 1, 1], padding='SAME') + self.B11)

            self.W12 = tf.get_variable("W12", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
            self.B12 = tf.get_variable("B12", initializer=tf.random_normal([128], stddev=0.01))
            self.L12 = tf.nn.relu(tf.nn.conv2d(self.L11, self.W12, strides=[1, 1, 1, 1], padding='SAME') + self.B12)

            self.W13 = tf.get_variable("W13", shape=[1, 1, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
            self.B13 = tf.get_variable("B13", initializer=tf.random_normal([128], stddev=0.01))
            self.L13 = tf.nn.relu(tf.nn.conv2d(self.L12, self.W13, strides=[1, 1, 1, 1], padding='SAME') + self.B13)

            self.FlatLayer = tf.reshape(self.L13, [-1, 8 * 8 * 128])
            self.Flat_W = tf.get_variable("Flat_W", shape=[8 * 8 * 128, 4096],initializer=tf.contrib.layers.xavier_initializer())
            self.Flat_B = tf.get_variable("Flat_B", initializer=tf.random_normal([4096], stddev=0.01))

            self.hypothesis = tf.matmul(self.FlatLayer, self.Flat_W) + self.Flat_B

            self.softmaxOfHypothesis = tf.nn.softmax(self.hypothesis)
            self.sotfmax = tf.nn.softmax(self.K)
            self.cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.hypothesis, labels=self.K))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
            # tf.get_variable_scope().reuse_variables() # 변수를 재사용하기 위한 방법

            correct_prediction = tf.equal(tf.argmax(self.hypothesis, 1), tf.argmax(self.K, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100

            tf.summary.scalar("Policy Cost", self.cost)
            accuracy_summary = tf.summary.scalar("Policy Accuracy", accuracy)

            # Summary
            self.summary = tf.summary.merge_all()
            self.sess.run(tf.global_variables_initializer())
        self.PN_saves = {policyNetworkName + "W1": self.W1, policyNetworkName + "B1": self.B1,
                         policyNetworkName + "W2": self.W2, policyNetworkName + "B2": self.B2,
                         policyNetworkName + "W3": self.W3, policyNetworkName + "B3": self.B3,
                         policyNetworkName + "W4": self.W4, policyNetworkName + "B4": self.B4,
                         policyNetworkName + "W5": self.W5, policyNetworkName + "B5": self.B5,
                         policyNetworkName + "W6": self.W6, policyNetworkName + "B6": self.B6,
                         policyNetworkName + "W7": self.W7, policyNetworkName + "B7": self.B7,
                         policyNetworkName + "W8": self.W8, policyNetworkName + "B8": self.B8,
                         policyNetworkName + "W9": self.W9, policyNetworkName + "B9": self.B9,
                         policyNetworkName + "W10": self.W10, policyNetworkName + "B10": self.B10,
                         policyNetworkName + "W11": self.W11, policyNetworkName + "B11": self.B11,
                         policyNetworkName + "W12": self.W12, policyNetworkName + "B12": self.B12,
                         policyNetworkName + "W13": self.W13, policyNetworkName + "B13": self.B13,
                         policyNetworkName + "Flat_W": self.Flat_W, policyNetworkName + "Flat_B": self.Flat_B,
                         }
        self.saver = tf.train.Saver(self.PN_saves)
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.policyNetworkFilePath))

        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            self.global_step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])
            print(ckpt.model_checkpoint_path)
        print("정책망 로딩 완료")

    def get_PolicyNetwork(self,chessBoard):
        input = self.make_Input(chessBoard)
        output = self.sess.run(self.softmaxOfHypothesis, feed_dict={self.X: input})
        return output
    def make_Input(self, chessBoard): #수정
        Input = []
        Input.append(B2A().board2array(chessBoard))
        self.batchSize = len(Input[0])
        return Input
    def learning(self, input, label):
        feed_dict = {self.X: input, self.K: label}
        s, c, _, = self.sess.run([self.summary, self.cost, self.optimizer], feed_dict=feed_dict)
    def saver(self):
        self.saver.save(self.sess, self.policyNetworkFilePath, global_step=self.global_step+self.batchSize, )
    def get_PolicyNetworkMove(self,chessBoard):
        softMax = self.get_PolicyNetwork(chessBoard)
        softMax = np.array(softMax[0])
        ArgMaxOfSoftmax = (-softMax).argsort()
        # 내림차순으로 분류한 것을 리스트로 반환 받는다
        # softMAxArgMax는 크기별로 Index만 저장 되어있다. 0~4095
        ohe = OHE()
        i = 0
        child = 0
        numOfLegalMoves = len(chessBoard.legal_moves)
        numOfChild = 1

        # 만드려고 하는 자식 개수보다 가능한 move 갯수가 적을때
        if numOfLegalMoves < numOfChild:
            numOfChild = numOfLegalMoves

        for j in range(4096):
            if child >= numOfChild:  # 만드려고 하는 자식 갯수보다 많으면 반환
                break
            try:
                tmpMove = ohe.indexToMove4096(ArgMaxOfSoftmax[i])
                strMove = copy.deepcopy(tmpMove)
                tmpMove = chess.Move.from_uci(tmpMove) # 주석처리: 선피쉬랑 붙기 위해 String 자체를 사용
            except:
                None
            if tmpMove in chessBoard.legal_moves:
                move=strMove # tmpMove가 legal이면 추가
                score = softMax[ArgMaxOfSoftmax[i]]
                print(i+1,"번째 선택된 점수 : ",score, " move: ",move)
                child += 1
            i += 1
        return move
    def getArraysOfPolicyNetwork(self,chessBoard):
        array4096 = self.get_PolicyNetwork(chessBoard)
        array4096 = np.array(array4096[0])
        ArgmaxOfSoftmax = (-array4096).argsort()
        #내림차순으로 분류한 것을 리스트로 반환 받는다
        #ArgMaxOfSoftmax 크기별로 Index만 저장 되어있다. 0~4095
        #계산된 softmax값과
        # 크기별로 정렬된 index가 들어 있는 ArgMaxOfSoftmax 반환
        return array4096, ArgmaxOfSoftmax


class ValueNetwork:
    def __init__(self,path):

        self.sess = tf.Session()
        self.batchSize=0

        valueNetworkName = "VN/"
        self.valueNetworkFilePath = path
        print("가치망 로딩")
        with tf.variable_scope("VN", reuse=False):
            self.X = tf.placeholder(tf.float32, [None, 8, 8, 36], name="X")  # 체스에서 8X8X10 이미지를 받기 위해 64
            self.K = tf.placeholder(tf.float32, [None], name="K")

            self.W1 = tf.get_variable("W1", shape=[5, 5, 36, 128], initializer=tf.contrib.layers.xavier_initializer())
            self.B1 = tf.get_variable("B1", initializer=tf.random_normal([128], stddev=0.01))
            self.L1 = tf.nn.relu(tf.nn.conv2d(self.X, self.W1, strides=[1, 1, 1, 1], padding='SAME') + self.B1)

            self.W2 = tf.get_variable("W2", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
            self.B2 = tf.get_variable("B2", initializer=tf.random_normal([128], stddev=0.01))
            self.L2 = tf.nn.relu(tf.nn.conv2d(self.L1, self.W2, strides=[1, 1, 1, 1], padding='SAME') + self.B2)

            self.W3 = tf.get_variable("W3", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
            self.B3 = tf.get_variable("B3", initializer=tf.random_normal([128], stddev=0.01))
            self.L3 = tf.nn.relu(tf.nn.conv2d(self.L2, self.W3, strides=[1, 1, 1, 1], padding='SAME') + self.B3)

            self.W4 = tf.get_variable("W4", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
            self.B4 = tf.get_variable("B4", initializer=tf.random_normal([128], stddev=0.01))
            self.L4 = tf.nn.relu(tf.nn.conv2d(self.L3, self.W4, strides=[1, 1, 1, 1], padding='SAME') + self.B4)

            self.W5 = tf.get_variable("W5", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
            self.B5 = tf.get_variable("B5", initializer=tf.random_normal([128], stddev=0.01))
            self.L5 = tf.nn.relu(tf.nn.conv2d(self.L4, self.W5, strides=[1, 1, 1, 1], padding='SAME') + self.B5)

            self.W6 = tf.get_variable("W6", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
            self.B6 = tf.get_variable("B6", initializer=tf.random_normal([128], stddev=0.01))
            self.L6 = tf.nn.relu(tf.nn.conv2d(self.L5, self.W6, strides=[1, 1, 1, 1], padding='SAME') + self.B6)

            self.W7 = tf.get_variable("W7", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
            self.B7 = tf.get_variable("B7", initializer=tf.random_normal([128], stddev=0.01))
            self.L7 = tf.nn.relu(tf.nn.conv2d(self.L6, self.W7, strides=[1, 1, 1, 1], padding='SAME') + self.B7)

            self.W8 = tf.get_variable("W8", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
            self.B8 = tf.get_variable("B8", initializer=tf.random_normal([128], stddev=0.01))
            self.L8 = tf.nn.relu(tf.nn.conv2d(self.L7, self.W8, strides=[1, 1, 1, 1], padding='SAME') + self.B8)

            self.W9 = tf.get_variable("W9", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
            self.B9 = tf.get_variable("B9", initializer=tf.random_normal([128], stddev=0.01))
            self.L9 = tf.nn.relu(tf.nn.conv2d(self.L8, self.W9, strides=[1, 1, 1, 1], padding='SAME') + self.B9)

            self.W10 = tf.get_variable("W10", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
            self.B10 = tf.get_variable("B10", initializer=tf.random_normal([128], stddev=0.01))
            self.L10 = tf.nn.relu(tf.nn.conv2d(self.L9, self.W10, strides=[1, 1, 1, 1], padding='SAME') + self.B10)

            self.W11 = tf.get_variable("W11", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
            self.B11 = tf.get_variable("B11", initializer=tf.random_normal([128], stddev=0.01))
            self.L11 = tf.nn.relu(tf.nn.conv2d(self.L10, self.W11, strides=[1, 1, 1, 1], padding='SAME') + self.B11)

            self.W12 = tf.get_variable("W12", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
            self.B12 = tf.get_variable("B12", initializer=tf.random_normal([128], stddev=0.01))
            self.L12 = tf.nn.relu(tf.nn.conv2d(self.L11, self.W12, strides=[1, 1, 1, 1], padding='SAME') + self.B12)

            self.W13 = tf.get_variable("W13", shape=[1, 1, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
            self.B13 = tf.get_variable("B13", initializer=tf.random_normal([128], stddev=0.01))
            self.L13 = tf.nn.relu(tf.nn.conv2d(self.L12, self.W13, strides=[1, 1, 1, 1], padding='SAME') + self.B13)

            self.FlatLayer = tf.reshape(self.L13, [-1, 8 * 8 * 128])
            self.Flat_W = tf.get_variable("Flat_W", shape=[8 * 8 * 128, 1],initializer=tf.contrib.layers.xavier_initializer())
            self.Flat_B = tf.get_variable("Flat_B", initializer=tf.random_normal([1], stddev=0.01))

            self.hypothesis = tf.matmul(self.FlatLayer, self.Flat_W) + self.Flat_B
            self.sigmoidOfHypothesis = tf.nn.sigmoid(self.hypothesis)
            self.cost = tf.reduce_mean(tf.square(self.hypothesis - self.K))
            # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.cost)
            # Accuracy
            correct_prediction = tf.equal(tf.argmax(self.hypothesis, 1), tf.argmax(self.K, 0))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100

            tf.summary.scalar("Cost", self.cost)
            accuracy_summary = tf.summary.scalar("Accuracy", accuracy)

            # Summary
            self.summary = tf.summary.merge_all()
            self.VN_saves = {valueNetworkName + "W1": self.W1, valueNetworkName + "B1": self.B1,
                             valueNetworkName + "W2": self.W2, valueNetworkName + "B2": self.B2,
                             valueNetworkName + "W3": self.W3, valueNetworkName + "B3": self.B3,
                             valueNetworkName + "W4": self.W4, valueNetworkName + "B4": self.B4,
                             valueNetworkName + "W5": self.W5, valueNetworkName + "B5": self.B5,
                             valueNetworkName + "W6": self.W6, valueNetworkName + "B6": self.B6,
                             valueNetworkName + "W7": self.W7, valueNetworkName + "B7": self.B7,
                             valueNetworkName + "W8": self.W8, valueNetworkName + "B8": self.B8,
                             valueNetworkName + "W9": self.W9, valueNetworkName + "B9": self.B9,
                             valueNetworkName + "W10": self.W10, valueNetworkName + "B10": self.B10,
                             valueNetworkName + "W11": self.W11, valueNetworkName + "B11": self.B11,
                             valueNetworkName + "W12": self.W12, valueNetworkName + "B12": self.B12,
                             valueNetworkName + "W13": self.W13, valueNetworkName + "B13": self.B13,
                             valueNetworkName + "Flat_W": self.Flat_W, valueNetworkName + "Flat_B": self.Flat_B,
                             }

            self.saver = tf.train.Saver(self.VN_saves)
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.valueNetworkFilePath))
            # print(ckpt.model_checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt.model_checkpoint_path)
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                self.global_step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])

    def get_ValueNetwork(self,chessBoard):
        input = self.make_Input(chessBoard)
        sigmoid = self.sess.run(self.sigmoidOfHypothesis, feed_dict = {self.X:input})
        tanh = 2*sigmoid -1
        tanh =  tanh[0][0] #tanh의 결과 값이 배열에 들어 있으므로
        return tanh
    def make_Input(self, chessBoard):
        input = []
        input.append(B2A().board2array(chessBoard))
        self.batchSize = len(input[0])
        return input
    def learning(self,input,label):
        feed_dict = {self.X: input, self.K: label}
        s, c, _, = self.sess.run([self.summary,self.cost, self.optimizer],feed_dict=feed_dict)
    def saver(self):
        self.saver.save(self.sess, self.valueNetworkFilePath, global_step=self.global_step+self.batchSize)
        # writer.add_summary(s, global_step=global_step)
class Rollout:
    def __init__(self,path):
        self.sess = tf.Session()

        rolloutName = "RO/"
        self.rolloutFilePath = path

        with tf.variable_scope("RO", reuse=False):
            self.X = tf.placeholder(tf.float32, [None, 8, 8, 16], name="X")  # 체스에서 8X8X10 이미지를 받기 위해 64

            self.W1 = tf.get_variable("W1", shape=[3, 3, 16, 128], initializer=tf.contrib.layers.xavier_initializer())
            self.B1 = tf.get_variable("B1", initializer=tf.random_normal([128], stddev=0.01))
            self.L1 = tf.nn.relu(tf.nn.conv2d(self.X, self.W1, strides=[1, 1, 1, 1], padding='SAME') + self.B1)

            self.W2 = tf.get_variable("W13", shape=[1, 1, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
            self.B2 = tf.get_variable("B13", initializer=tf.random_normal([128], stddev=0.01))
            self.L2 = tf.nn.relu(tf.nn.conv2d(self.L1, self.W2, strides=[1, 1, 1, 1], padding='SAME') + self.B2)

            self.FlatLayer = tf.reshape(self.L2, [-1, 8 * 8 * 128])
            self.Flat_W = tf.get_variable("Flat_W", shape=[8 * 8 * 128, 4096],
                                     initializer=tf.contrib.layers.xavier_initializer())
            self.Flat_B = tf.get_variable("Flat_B", initializer=tf.random_normal([4096], stddev=0.01))

            self.hypothesis = tf.matmul(self.FlatLayer, self.Flat_W) + self.Flat_B
            self.softmaxOfHypothesis = tf.nn.softmax(self.hypothesis)

            self.RO_saves = {rolloutName + "W1": self.W1, rolloutName + "B1": self.B1,
                             rolloutName + "W13": self.W2, rolloutName + "B13": self.B2,
                             rolloutName + "Flat_W": self.Flat_W, rolloutName + "Flat_B": self.Flat_B,
                             }

            saver = tf.train.Saver(self.RO_saves)
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.rolloutFilePath))
            # print(ckpt.model_checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                # print(ckpt.model_checkpoint_path)
                saver.restore(self.sess, ckpt.model_checkpoint_path)

    def get_Rollout(self, chessBoard):
        input = self.make_Input(chessBoard)
        rollout = self.sess.run(self.softmaxOfHypothesis, feed_dict={self.X: input})

        return rollout

    def make_Input(self, chessBoard):
        input = []
        input.append(B2A().board2arrayForRollout(chessBoard))
        return input

    def learning(self,input, label):
        pass

    def get_RolloutMove(self,chessBoard):
        softMax = self.get_Rollout(chessBoard)
        softMax = np.array(softMax[0])
        ArgMaxOfSoftmax = (-softMax).argsort()
        # 내림차순으로 분류한 것을 리스트로 반환 받는다
        # softMAxArgMax는 크기별로 Index만 저장 되어있다. 0~4095
        ohe = OHE()
        i = 0
        child = 0
        numOfLegalMoves = len(chessBoard.legal_moves)
        numOfChild = 1

        # 만드려고 하는 자식 개수보다 가능한 move 갯수가 적을때
        if numOfLegalMoves < numOfChild:
            numOfChild = numOfLegalMoves

        for j in range(4096):
            if child >= numOfChild:  # 만드려고 하는 자식 갯수보다 많으면 반환
                break
            try:
                tmpMove = ohe.indexToMove4096(ArgMaxOfSoftmax[i])
                strMove = copy.deepcopy(tmpMove)
                tmpMove = chess.Move.from_uci(tmpMove) # 주석처리: 선피쉬랑 붙기 위해 String 자체를 사용
            except:
                None
            if tmpMove in chessBoard.legal_moves:
                # print(tmpMove,end= "")
                move=strMove # tmpMove가 legal이면 추가
                score = softMax[ArgMaxOfSoftmax[i]]
                # print(i+1,"번째 선택된 점수 : ",score, " move: ",move)
                child += 1
            i += 1
        return move