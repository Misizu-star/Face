import os
import shutil
import cv2

def generator(data):
    name = input('my name:')
    # 如果路径存在则删除路径
    path = os.path.join(data, name)
    if os.path.isdir(path):
        shutil.rmtree(path)
    # 创建文件夹
    os.mkdir(path)
    # 创建一个级联分类器
    face_casecade = cv2.CascadeClassifier('frontalface.xml')
    # 打开摄像头
    camera = cv2.VideoCapture(0)
    cv2.namedWindow('Dynamic')
    # 计数
    count = 0

    while True:
        # 读取一帧图像
        ret, frame = camera.read()
        frame = cv2.flip(frame, 180)
        if ret:
            # 转换为灰度图
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 人脸检测
            face = face_casecade.detectMultiScale(gray_img, 1.2, 3)
            for (x, y, w, h) in face:
                # 在原图上绘制矩形
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                info = "gathering"
                cv2.putText(frame, info, (x + (w - len(info) * 12) // 2, y + h - 12),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
                # 调整图像大小
                new_frame = cv2.resize(frame[y + 2:y + h - 2, x + 2:x + w - 2], (184, 224))
                # 保存人脸
                cv2.imwrite('%s/%s.png' % (path, str(count)), new_frame)
                count += 1
            cv2.imshow('Dynamic', frame)
            # 按下q键退出
            if count % 5 == 0:
                print(count)
            if cv2.waitKey(100) & 0xff == 27 or count >= 100:
                break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    generator('./face')