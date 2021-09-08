import numpy
import matplotlib.pyplot

# 1) 100개 훈련 파일
# data_file = open("./mnist_dataset/mnist_train_100.csv", "r")
# data_list = data_file.readlines()
# data_file.close()
#
# # print("len = ", len(data_list))
# print("len = " + str(len(data_list)))
# for data in data_list:
#     print(data)

# 2) 60000개 파일 읽기
data_file = open("./mnist_dataset/mnist_train.csv", "r")
data_list = data_file.readlines()
data_file.close()
print("len = " + str(len(data_list)))

# 3) 원하는 이미지 번호 선택
snum = input("0~59999중 번호 선택 : ")
num = int(snum)
print("숫자 : ", data_list[num][0])   # label
print("픽셀 : ", data_list[num][2:])  # 2번째 글자부터 끝까지

# 4) 시각화하기
all_values = data_list[num].split(',')
print("all_values = ", all_values)
image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
print("image_array = ", image_array)

# matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
matplotlib.pyplot.imshow(image_array, cmap='Greys_r', interpolation='None')
matplotlib.pyplot.show()
