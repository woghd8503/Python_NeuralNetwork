import numpy
import time
from mynn import *
import matplotlib.pyplot as plt


def main():
    input_nodes = 784           # 28x28=784 픽셀수
    hidden_nodes = 200          # 은닉계층 노드수(하이퍼파라미터)
    output_nodes = 10           # 0 ~9 까지 숫자 분류

    learning_rate = 0.1         # 가중치의 변경정도를 결정하는 보폭(하이퍼파라미터)

    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # 이미 학습된 가중치를 복원
    ans = input("학습데이터를 복원하시겠습니까? (yes) >> ")
    if ans == 'yes':
        n.load("./weights/wih_0.bin", "./weights/who_0.bin")

    # mnist 테스트 데이터를 읽어들이자
    test_data_file = open("./mnist_dataset/mnist_test.csv", "r")
    test_data_list = test_data_file.readlines() # 60000개 요소 리스트
    test_data_file.close()

    # 시험데이터 10000개중에 몇개 정답을 맞추나?
    scorecard = []
    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0]) # 첫번째 항목이 정답(해당 숫자)
        # 0~255 -> 0~1 ->  0.01<=input<=1
        inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99) + 0.01
        _, outputs = n.query(inputs)  # 순전파 과정을 통해 예측값을 출력
        label = numpy.argmax(outputs)   # 10개 노드중에 가장 큰 값이 정답
        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)

    scorecard_array = numpy.asarray(scorecard)
    print("perforcemance = ", scorecard_array.sum()/scorecard_array.size)





if __name__ == '__main__':
    main()