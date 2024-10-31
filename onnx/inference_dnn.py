import cv2
import numpy as np


def img_demo():
    imgpath = '../dengchao.jpeg'
    # imgpath = '../human.jpg'
    # imgpath = '../portrait_heng.jpg'
    # imgpath = '../portrait_shu.jpg'
    frame = cv2.imread(imgpath)
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # 加载模型
    net = cv2.dnn.readNetFromONNX("../pretrained/modnet_photographic_portrait_matting.onnx")
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (512, 512), (127.5, 127.5, 127.5), swapRB=True)
    net.setInput(blob)
    score = net.forward()
    # print(score)
    mask = (np.squeeze(score[0]) * 255).astype('uint8')
    mask = cv2.resize(mask, dsize=(frameWidth, frameHeight), interpolation=cv2.INTER_AREA)

    # -----白色背景（可选）-----
    mask[mask < 230] = 0
    mask[mask >= 230] = 1
    mask[mask == 0] = 255
    mask[mask == 1] = 0
    # 改善图像边缘细节
    image = np.expand_dims(mask, axis=2)
    mask = np.concatenate((image, image, image), axis=-1)
    result = cv2.bitwise_or(frame, mask)
    cv2.imwrite('C:/Users/hldes/Desktop/MODNet/result.png', result)
    # ----------------------

    # # 改善图像边缘细节
    # mask[mask >= 200] = 255
    # mask[mask < 200] = 0
    #
    # # mask增加维度
    # image = np.expand_dims(mask, axis=2)
    # mask = np.concatenate((image, image, image), axis=-1)
    #
    # result = cv2.bitwise_and(frame, mask)
    #
    # cv2.imwrite('C:/Users/hldes/Desktop/MODNet/result2.png', result)

def cam_demo():

    cap = cv2.VideoCapture(0)
    # 加载模型
    net = cv2.dnn.readNetFromONNX("../pretrained/modnet_photographic_portrait_matting.onnx")
    while (cap.isOpened()):
        ret, frame = cap.read()
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (512, 512), (127.5, 127.5, 127.5), swapRB=True)
        net.setInput(blob)
        score = net.forward()

        mask = (np.squeeze(score[0]) * 255).astype('uint8')
        mask = cv2.resize(mask, dsize=(frameWidth, frameHeight), interpolation=cv2.INTER_AREA)

        # -----白色背景（可选）-----
        mask[mask < 230] = 0
        mask[mask >= 230] = 1
        mask[mask == 0] = 255
        mask[mask == 1] = 0
        # 改善图像边缘细节
        image = np.expand_dims(mask, axis=2)
        mask = np.concatenate((image, image, image), axis=-1)
        result = cv2.bitwise_or(frame, mask)
        cv2.imshow('result', result)
        q = cv2.waitKey(1)
        if q == ord('q'):
            cv2.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    img_demo()
    cam_demo()
