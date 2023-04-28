import cv2
import PIL
import numpy
import os
import sklearn.svm from LinearSVC

class Model:

    def __init__(self, img_list=None, counters=None):
        self.model = LinearSVC()

        def train_model(self):
            img_list = numpy.array([])
            class_list = numpy.array([])

            for i range(1, counters[0]):
                img = cv2.imread(f"1/frame{i}.jpg")[:,:,0]
                img = img.reshape(16950)
                img_list = numpy.append(img_list, [img])
                class_list = numpy.append(class_list, 2)
        img_list = img_list.reshape(counters[0] - 1 + counters[1] - 1, 16950)
        self.model.fit(img_list, class_list)
        print("MODEL SUCCESSFULLY TRAINED!")

    def predict(self, frame, img=None):
        frame = frame[1]
        cv2.imwrite("frame.jpg",cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
        img.thumbnail((150,150), PIL.Image.ANTIALIAS)
        img.save("frame.jpg")

        img = cv2.imread("frame.jpg")[;,;,0]
        img = img.reshape(16950)
        prediction  = self.model.predict([img])
        return prediction[0]


