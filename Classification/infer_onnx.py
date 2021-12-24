import time
import cv2
import numpy as np
from sklearn.utils.extmath import softmax


class Predictor(object):
    def __init__(self, model_path="model.pth"):
        self.model = cv2.dnn.readNetFromONNX(model_path)
        self.classes = ["defect", "good"]

    def label_defective_region(self, img):
        img = img.astype(np.float32)
        input_blob = cv2.dnn.blobFromImage(
            image=img,
            scalefactor=1 / 255.0,
            size=(112, 112),
            mean=np.array([0.485, 0.456, 0.406]) * 255.0,
            swapRB=True,
            crop=False
        )
        input_blob[0] /= np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        self.model.setInput(input_blob)
        # OpenCV DNN inference
        outputs = self.model.forward()
        outputs = softmax(outputs)
        index = np.argmax(outputs)
        result = {"label": self.classes[index], "score": outputs[0][index]}
        return result


if __name__ == "__main__":
    img_path = "/home/zsv/PycharmProjects/training-cpp/Classification/data1/train/good/1640265077.6384468.png"
    # img = Image.open(img_path).convert('RGB')
    img = cv2.imread(img_path)
    predict = Predictor(model_path="label_defective_region.onnx")
    start = time.time()
    print(predict.label_defective_region(img))
    print(time.time() - start)