# coding:utf-8

from model import Model
from PIL import Image
import requests
import numpy

table = [255 if i > 140 else i for i in range(256)]
url = "http://mis.teach.ustc.edu.cn/randomImage.do?date='" + str(numpy.random.randint(2147483647)) + "'"
req = requests.get(url)
try:
    with open("tmp.jpg", 'wb') as f:
        f.write(req.content)
        f.close()
except IOError:
    print("IOError")
finally:
    req.close()
img = Image.open("tmp.jpg").convert('L').point(table)
images = numpy.zeros([4, 20, 20])
images[0, :, :] = numpy.array(img.crop((00, 0, 20, 20)))
images[1, :, :] = numpy.array(img.crop((20, 0, 40, 20)))
images[2, :, :] = numpy.array(img.crop((40, 0, 60, 20)))
images[3, :, :] = numpy.array(img.crop((60, 0, 80, 20)))

labels = numpy.array(list("23456789ABCDEFGHJKLMNPQRSTUVWXYZ"))

model = Model()
prediction = model.test(images)
index = numpy.argmax(prediction, axis=1)
print("".join(labels[index]))
