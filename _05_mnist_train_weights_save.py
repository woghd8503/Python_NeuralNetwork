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

    # mnist 훈련 데이터를 읽어들이자
    training_data_file = open("./mnist_dataset/mnist_train.csv", "r")
    training_data_list = training_data_file.readlines() # 60000개 요소 리스트
    training_data_file.close()

    LOG_REPEAT_NUM = 6000       # 화면에 로그를 보여주는 주기
    start_time = time.time()    # 현재시간 저장
    predict_sum = 0.0           # 6000번동안 누적한 예측값
    error_sum = 0.0             # 6000번동안 누적한 에러
    train_list_x = []           # 그래프의 x축(학습 횟수)
    error_list_y = []           # 그래프의 y축(각 구간별 누적에러율)
    predict_list_y = []         # 그래프의 y축(각 구간별 예측평균값)
    # 학습시작
    epoch = 10   # 전체 데이터셋을 가지고 몇번 반복 훈련할 것인지(하이퍼파라미터)
    for e in range(epoch):
        # print("epoch {}".format(e+1))
        for i, record in enumerate(training_data_list):
            # ------784노드에 입력할 입력데이터를 만들어주자------ start
            # ,를 구분자로 삼으므로 ,를 기준으로 리스트로 저장
            all_values = record.split(",")
            # print("len(all_values) = ", len(all_values))
            correct_ans = all_values[0]   # 첫번째 데이터가 정답
            input_vals = all_values[1:]   # 두번째부터 끝까지 저장(픽셀데이터)
            # 0을 안만들기 위해서, 0은 w를 무력화, 1은 w를 그대로 전달
            # 이미지 데이터의 픽셀은 0.01 <= i <= 1 사이의 값으로 정규화한다
            inputs = (numpy.asfarray(input_vals)/255.0*0.99) + 0.01
            # ------784노드에 입력할 입력데이터를 만들어주자------ end
            # ------10개 노드의 정답을 만들어주자------ start
            # 정답인 위치의 노드는 0.99로 하고, 정답이 아닌 노드는 0.01로 초기화
            # sigmoid함수는 0 < t < 1 범위이기 때문에 0또는 1을 결과로 주게 되면
            # 도달할 수 없는 값이 되므로 학습이 잘 되지 않는다
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(correct_ans)] = 0.99
            # ------10개 노드의 정답을 만들어주자------ start

            # 학습을 시키자
            squared_error_sum, predict_val = n.train(inputs, targets)
            error_sum += squared_error_sum      # 계속 누적한다
            predict_sum += predict_val
            if (i+1) % LOG_REPEAT_NUM == 0:
                print('-' * 30)
                print("train time : {}".format(time.time()-start_time))
                print("epoch {}/{}: data seq {}/{}".format(e+1, epoch, i+1, len(training_data_list)))
                print("error sum per {} : {}".format(LOG_REPEAT_NUM, error_sum))
                print("predict avg per {} : {}".format(LOG_REPEAT_NUM, predict_sum/LOG_REPEAT_NUM))
                train_list_x.append(e*len(training_data_list)+(i+1))
                error_list_y.append(error_sum)
                predict_list_y.append(predict_sum/LOG_REPEAT_NUM)
                error_sum = 0.0
                predict_sum = 0.0
                
    # 학습된 가중치값을 저장한다
    desc = """hidden_layer={}\nlearning_rate={}\nepoch={}""".format(
            hidden_nodes, learning_rate, epoch)
    n.save(desc,
           "./weights/wih_epoch_{}.bin".format(epoch),
           "./weights/who_epoch_{}.bin".format(epoch))

    plt.plot(train_list_x, error_list_y)
    plt.xlabel("train_num")
    plt.ylabel("error_sum")
    plt.show()

    # plt.plot(train_list_x, predict_list_y)
    # plt.xlabel("train_num")
    # plt.ylabel("predict_avg")
    # plt.show()





if __name__ == '__main__':
    main()