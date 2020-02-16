import tensorflow as tf
import numpy as np
import PIL.Image as pilimg

"""
airplane 0726
cat 0884
dog 0701
motorbike 0787
person 0985
"""

def set_numlist():
    numlist = list(range(1000))
    for i in range(1000):
        numlist[i] = str(numlist[i])

    for j in range(10):
        numlist[j] = "00" + numlist[j]
    for k in range(90):
        numlist[k+10] = "0" + numlist[k+10]
    print(numlist)
    return numlist


def set_data(numlist):
    input_data, answer_data = [], []

    text = ['airplane', 'cat', 'dog', 'motorbike', 'person']
    length = [727, 885, 702, 788, 986]
    for i in range(len(text)):
        for j in range(length[i]):
            rgb_data = []
            data = pilimg.open("../data/" + text[i] + "/" + text[i] + "_0" + numlist[j] + ".jpg")
            data_np = np.array(data)
            data_np = normalize_image(data_np)
            rgb_data.append(get_onechannel(data_np, 0))
            rgb_data.append(get_onechannel(data_np, 1))
            rgb_data.append(get_onechannel(data_np, 2))
            input_data.append(rgb_data)
            answer_data.append(i)

    print(input_data[323][0], input_data[323][1], answer_data)
    return input_data, answer_data



def normalize_image(image):
    return image / 255.0


def get_onechannel(data, num):
   # print(len(data), len(data[0]))
    output = np.zeros((len(data), (len(data[0]))))
    for i in range(len(data)):
        for j in range(len(data[0])):
            output[i][j] = data[i][j][num]
    return output


if __name__=="__main__":

    #test = pilimg.open("../data/cat/cat_0000.jpg")
    #test_np = np.array(test)
    #print(len(test_np), len(test_np[0]), len(test_np[0][0]))

    #print(normalize_image(test_np[0]))
    #그냥이 행, test_np는 열

    set_data(set_numlist())