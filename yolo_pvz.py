import pygetwindow
from ultralytics import YOLOv10
import pyautogui as pag #获取鼠标
import pygetwindow as pgw #捕捉窗口
import numpy as np
import cv2 as cv
import torch
from PIL import ImageGrab

model = YOLOv10.from_pretrained('jameslahm/yolov10n')
model = YOLOv10("best.pt")
device = torch.device('cuda:0')#用显卡推理
model.to(device)
#定位的窗口名字
window_title = "Plants vs. Zombies"
#获取窗口
window = pgw.getWindowsWithTitle(window_title)[0]#getWindowsWithTitle()返回的是一个窗口对象列表，我们游戏只有一个窗口，所以只需要第一个元素

#捕获图像并处理q
while True:
    if window:#判断窗口是否正常显示
        #获取窗口坐标和尺寸
        x, y, w, h = window.left, window.top, window.width, window.height
        #用PIL抓取窗口获得截图
        screenshot = ImageGrab.grab([x, y, w+x, h+y])# x y定位窗口在屏幕上左上角坐标 w+x y+h定位窗口在屏幕右下角坐标
        #将截图由rgb转换成opencv可处理的bgr格式
        image_src = cv.cvtColor(np.array(screenshot),cv.COLOR_RGB2BGR)
        #获取尺寸，为后面的鼠标点击处理做准备。注意这里，图片的shape形状是【height，width】 所以x对应shape[1] y对应shape[0]
        size_x, size_y = image_src.shape[1], image_src.shape[0]
        # 转换成训练时的尺寸 方便模型检测
        image_rsp = cv.resize(image_src,(640,640))
        #用yolo模型进行检测
        results = model.predict(source=image_rsp, imgsz=640, conf=0.3, save=False)#conf置信度 越小检测精度越高 因为是游戏所以不需要太高
        #yolo检测返回的是一个列表 我们需要找到图片在窗口的中心坐标可以通过.boxes.xywhn找到
        boxes = results[0].boxes.xywhn #xywhn是一个二维矩阵 矩阵行数等于检测到的目标个数 每一行的格式【x，y，box-width，box-height】
        #优化打僵尸 优先从左边打
        boxes = sorted(boxes, key=lambda x:x[0])
        count = 0 #设置一个计数
        for box in boxes:
            #为了验证我们有没有检测到目标 我们在抓取到的截图中画出目标
            cv.rectangle(image_src, (int((box[0] - box[2]/2)*size_x), int((box[1] - box[3]/2)*size_y)),
                         (int((box[0] + box[2]/2)*size_x), int((box[1] + box[3]/2)*size_y)), color=(255,255,0), thickness=2) #因为返回的图片进行了归一化数据 所以我们要乘以原来的尺寸
            #设置点击鼠标操作
            pag.click(x=x + box[0] * size_x, y=y + box[1] * size_y)
            count +=1 #每次打僵尸直到打死
            if count == 1:
                break
         #画出框图验证
        cv.imshow('Detect',image_src)

        #按q退出
        if cv.waitKey(1) == ord('q'):
            break

        pass



