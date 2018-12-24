# coding:utf-8

from model import *

model = Model()
train_x, train_y, test_x, test_y = get_data("data/train.mat", "data/test.mat")
model.train(train_x, train_y, test_x, test_y, 0.0001, 3000, 40)
