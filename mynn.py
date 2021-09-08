
import numpy
import scipy.special
import pickle

# 신경망 클래스 정의
class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes    # 입력노드 개수
        self.hnodes = hiddennodes   # 은닉노드 개수
        self.onodes = outputnodes   # 출력노드 개수
        
        self.lr = learningrate      # 학습률

        # 가중치 행렬(wih, who)
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5),
                                       (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5),
                                       (self.onodes, self.hnodes))


        # self.activation_function = self.sigmoid
        # self.activation_function = self.logistic
        self.activation_function = lambda x : scipy.special.expit(x)
        pass

    def sigmoid(self, x):
        return scipy.special.expit(x)

    def logistic(self, x):
        return 1/(1+2.71828**(-x))

    def train(self, inputs_list, targets_list):
        "신경망 학습하는 함수"
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 순전파 과정을 통해 역전파 학습을 위한 데이터를 리턴받는다
        hidden_outputs, final_outputs = self.query(inputs_list)

        # 출력층 에러와 중간층 에러값을 얻는다
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        
        # 가중치 업데이트 수식을 적용한다(오류 역전파를 통해 가중치 증가/감소 : 학습)
        self.who += self.lr * numpy.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
            numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            numpy.transpose(inputs))

        # 나머지는 0.01, label은 0.99인 목표값 인덱스를 구한다
        squared_error = output_errors**2     # 제곱오차
        squared_error_sum = squared_error.sum() # 모든 노드의 오차를 더한 것
        target_idx = numpy.argmax(targets)   # 배열중 가장 큰 값의 인덱스
        predict_value = final_outputs[target_idx] # 정답 노드의 예측값

        return squared_error_sum, predict_value

    def query(self, inputs_list):
        "순전파/질의하기"

        # 입력리스트를 2차원 행렬로 변환
        inputs = numpy.array(inputs_list, ndmin=2).T
        # input->hidden weight 내적계산
        hidden_inputs = numpy.dot(self.wih, inputs)
        # sigmoid 함수를 통과시킨다
        hidden_outputs = self.activation_function(hidden_inputs)
        # hidden->output weight 내적계산
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # sigmoid 함수를 통과시킨다
        final_outputs = self.activation_function(final_inputs)

        return hidden_outputs, final_outputs
    
    def save(self, desc, wih_file, who_file):
        "가중치를 파일에 저장하는 함수"
        wih_list = wih_file.split('/')
        who_list = who_file.split('/')
        fname = "{}_{}_desc.txt".format(wih_list[-1], who_list[-1])
        with open(fname, "w") as f:   # w, wt
            f.write(desc)
        with open(wih_file, "wb") as f:
            pickle.dump(self.wih, f)
        with open(who_file, "wb") as f:
            pickle.dump(self.who, f)

    
    def load(self, wih_file, who_file):
        "가중치를 파일로부터 복원하는 함수"
        with open(wih_file, "rb") as f:
            self.wih = pickle.load(f)
        with open(who_file, "rb") as f:
            self.who = pickle.load(f)


if __name__ == '__main__':
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3
    learning_rate = 0.3

    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # tuple로 2개 배열이 묶여서 전달된다
    # foutputs = n.query([1.0, 0.5, -1.5])
    # print(foutputs)

    # ho, fo = n.query([1.0, 0.5, -1.5])
    # print("hidden_outputs : ", ho)
    # print("final_outputs : ", fo)

    # _를 쓴 것은 리턴값을 받지 않겠다는 뜻
    # _, fo = n.query([1.0, 0.5, -1.5])
    # print("final_outputs : ", fo)