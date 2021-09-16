import os
import cv2
import numpy as np
import serial


class Port:
    def __init__(self):
        self.ser = self.__Open("COM3", 115200)

    def __Open(self, port, bps, timeout=None):
        try:
            ser = serial.Serial(port, bps, timeout=timeout)
            if ser.is_open:
                return ser
        except Exception as e:
            print("串口打开失败", e)
            # exit(-1)
        return None

    def send_msg(self, data: str):
        return self.ser.write(data.encode('utf-8'))


# 载入图像
def LoadImages(data):
    images = []
    names = []
    labels = []
    label = 0

    # 遍历所有文件夹
    for subdir in os.listdir(data):
        subpath = os.path.join(data, subdir)
        if os.path.isdir(subpath):
            # 在每一个文件夹中存放着一个人的许多照片
            names.append(subdir)
            # 遍历文件夹中的图片文件
            for filename in os.listdir(subpath):
                imgpath = os.path.join(subpath, filename)
                img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images.append(gray_img)
                labels.append(label)
            label += 1
    images = np.asarray(images)
    labels = np.array(labels)
    return images, labels, names


# 检验训练结果
def FaceRec(data):
    port = Port()
    # 加载训练的数据
    X, y, names = LoadImages(data)
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(X, y)
    model.save("model.xml")

    # 打开摄像头
    camera = cv2.VideoCapture(0)
    cv2.namedWindow('Face recognition')

    # 创建级联分类器
    face_casecade = cv2.CascadeClassifier('frontalface.xml')

    count_ok = 0  # 正确次数
    count_err = 0  # 错误次数
    last_label = -1  # 上一次识别标签
    while True:
        ret, frame = camera.read()
        frame = cv2.flip(frame, 180)
        # 判断图像是否读取成功
        if ret:
            # 转换为灰度图
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 利用级联分类器鉴别人脸
            faces = face_casecade.detectMultiScale(gray_img, 1.3, 5)
            # 遍历每一帧图像，画出矩形
            for (x, y, w, h) in faces:
                roi_gray = gray_img[y:y + h, x:x + w]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红框
                try:
                    roi_gray = cv2.resize(roi_gray, (184, 224), interpolation=cv2.INTER_LINEAR)
                    params = model.predict(roi_gray)  # 预测
                    label = params[0]  # 预测标签
                    confidence = params[1]  # 自信度
                    if confidence <= 45:
                        if last_label != label:
                            count_ok = 0
                        count_ok = (count_ok + 1) % 30
                        if count_ok >= 5:  # 连续五次以上成功视为真正的成功
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绿框
                            info = "name: {}".format(names[label])
                            cv2.putText(frame, info, (x + (w - len(info) * 12) // 2, y + h - 10),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 255, 1)
                        if count_ok == 6:
                            port.send_msg('1')  # 发送串口消息
                            print("识别成功")
                            count_err = 0
                        last_label = label
                    else:
                        count_err = (count_err + 1) % 30
                        if 5 <= count_err <= 25:  # 识别五次失败显示信息
                            info = 'failed'
                            cv2.putText(frame, info, (x + (w - len(info) * 12) // 2, y + h - 12),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                        if count_err == 6:
                            port.send_msg('0')  # 发送串口消息
                            print("识别失败")
                            count_ok = 0
                    # print('Label:%s,confidence:%.2f' % (params[0], params[1]))
                except:
                    continue

            cv2.imshow('Face recognition', frame)

            # 按下esc键退出
            if cv2.waitKey(100) & 0xff == 27:
                break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    FaceRec('./face')
