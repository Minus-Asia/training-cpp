import time
import cv2
import glob
import numpy as np
from sklearn.utils.extmath import softmax
import onnxruntime as ort


class Predictor(object):
    def __init__(self, model_path="model.pth"):
        providers = [('CUDAExecutionProvider', {'device_id': 0})]
        self.sess = ort.InferenceSession(model_path, providers=providers)
        # self.model = cv2.dnn.readNetFromONNX(model_path)
        self.classes = ["defect", "good"]

    def label_defective_region(self, img_list):
        input_blobs = cv2.dnn.blobFromImages(
            images=img_list,
            scalefactor=1 / 255.0,
            size=(112, 112),
            mean=np.array([0.485, 0.456, 0.406]) * 255.0,
            swapRB=True,
            crop=False
        )
        input_blobs /= np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        outputs = self.sess.run(None, {'input': input_blobs})
        outputs = softmax(outputs[0])
        indexes = np.argmax(outputs, axis=1)
        results = [{"label": self.classes[indexes[i]], "score": outputs[i][indexes[i]]} for i in range(len(img_list))]
        return results


if __name__ == "__main__":
    # img = Image.open(img_path).convert('RGB')
    predict = Predictor(model_path="label_defective_region.onnx")
    img_list = []
    for img_path in glob.glob("/mnt/Datasets/YB_2812/bad_defective_region1/*.png"):
        img = cv2.imread(img_path)
        img = img.astype(np.float32)
        img_list.append(img)

    for i in range(100):
        start = time.time()
        print(predict.label_defective_region(img_list))
        print(time.time() - start)
