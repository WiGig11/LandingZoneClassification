import os, sys,cv2
import numpy as np
import onnxruntime

def preprocess_image_opencv(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 确保图像是以彩色模式读入
    if image is None:
        raise FileNotFoundError("Image not found at the specified path.")
    image = cv2.resize(image, (28, 28))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=-1)
    image = (image / 255.0 - 0.5) / 0.5
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (0, 3, 1, 2))
    return image

# 模型加载
onnx_model_path = "model.onnx"
net_session = onnxruntime.InferenceSession(onnx_model_path)
inputs = {net_session.get_inputs()[0].name: preprocess_image_opencv('test.png')}
outs = net_session.run(None, inputs)[0]

print("onnx weights", outs)
print("onnx prediction", outs.argmax(axis=1)[0])
