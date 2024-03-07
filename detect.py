# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.
Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import time
import platform
import sys
import numpy as np
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5x.pt',  # 使用yolo v5s
        source=ROOT / 'data/images',  # 檔案存取路徑
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(1080, 1080),  # 定義標示方框大小 (height, width)
        conf_thres=0.75,  # 辨識準確度
        iou_thres=0.45,  # 真實框與預測框的交集與合集
        max_det=1000,  # 每幅圖像的最大檢測數
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # 顯示結果
        save_txt=True,  # 儲存結果到*.txt
        save_conf=False,  # 儲存辨識準確度在*.txt標籤
        save_crop=False,  # 保存裁剪的預測框
        nosave=False,  # do not save images/videos
        classes=None,  # 依類別過濾: --class 0, or --class 0 2 3
        agnostic_nms=False,  # 一個對象應該只被標記為一個類，所以設為False
        augment=False,  # 關閉augment功能
        visualize=False,  # 關閉可視化功能
        update=False,  # 關閉更新模型的功能
        project=ROOT / 'runs/detect',  # 儲存結果路徑
        name='exp',  # 儲存結果之資料夾名稱
        exist_ok=False,  # 避免資料夾名稱重複
        line_thickness=1,  # 標示框之邊框像素
        hide_labels=False,  # 關閉標籤顯示
        hide_conf=False,  # 關閉辨識度顯示
        half=False,  # 關閉使用FP16半精確度推理
        dnn=False,  # 不將OpenCV DNN用於ONNX推理
        vid_stride=5,  # 降低輸入視頻的fps
        count=0,  #左上總和
):
    source = str(source) #把source從pathlib轉成str
    save_img = not nosave and not source.endswith('.txt')  # 儲存結果圖片
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS) # suffix是返回路徑中文件所有的最後一個元素
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))  # 判断source是否是url
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file) # isnumeric 判断是否只由數字组成
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # 新建保存文件目錄，如果不存在，則創建
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 增量運行
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 製作目錄

    # 載入模型
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # 確認圖像大小

    # 數據加載器
    bs = 1  # 初始化批次次數
    if webcam:
        view_img = check_imshow(warn=True) #回報錯誤
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride) #計算數據
        bs = len(dataset) #批次
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # 運行推理
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # 慢慢增加學習率，過高的學習率容易導致模型不穩定
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    for path, im, im0s, vid_cap, s in dataset: # path：圖片路徑  im:Padded resize後的圖片  im0s:原圖  vid_cap:影片相關參數  s:圖片訊息
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)  # torch.from_numpy()方法把數值轉換成張量，to(model.device)表示將所有最開始讀取數據的張量copy一份到device所指定的GPU上去，運算都在GPU上進行
            # im.half()是把數據類型轉換成float16
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 計算顏色0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                # 维度中使用None可以在所處緯度中多一维
                im = im[None]  # 擴展張量維度

        # 推理
        with dt[1]:
            # stem可以返回最後一項除了後綴以外的名字
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize) # 得到模型預測结果

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # 循環處理
        #如果檢測到邊界框，就把resize後的框大小調整尺度到原圖大小，再利用annotator.box_label把標籤（檢測结果和辨識準確度）和框畫到原圖上，最终保存圖片
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # 辨識框大小 >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg (p.name是獲取最後一項)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string

            cv2.line(im0, (175, 1250), (1650,1900), (0,0,255), 5)       #左下
            cv2.line(im0, (1700,1900), (2600,1200), (0, 0, 255), 5)     #右
            cv2.line(im0, (2400,1100), (1500, 850), (0, 0, 255), 5)     #上
            cv2.line(im0, (1550, 875), (400, 1200), (0, 0, 255), 5)     #左上
            #獲得對應圖片的長寬
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain im
            #根據圖片名稱設置txt文件的路徑
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # 將預測框的座標調整至基於其原本長寬的座標
                # 第一參數是resize後圖片的大小，第二個參數是邊框的大小，第三個參數是原圖的大小a, round()是四捨五入法求整數
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string，如果n>1，那麼就變成複數添加s

                # Write results
                for *xyxy, conf, cls in reversed(det): #逆序顯示圖片
                    # if cls != 2 or cls != 5 or cls != 7:
                    #     continue
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')


                    # 提取檢測框的坐標和置信度
                    left, top, right, bottom = map(int, xyxy)
                    center_x = (left + right) / 2
                    center_y = (top + bottom) / 2
                    #(1550, 875), (400, 1200)
                    #if ((center_x - 1550) * (1200 - 875) == (400 - 1550) * (center_y - 875)) and min(1550,400) <= center_x and center_x <= max(1550, 400) and min(875, 1200) <= center_y and center_y <= max(875, 1200):
                    #    count += 1
                    # 遍歷現有的目標，查找與檢測框最接近的目標
                    min_dist = float('inf')
                    targets = {}
                    target_id = None
                    for tid, target in targets.items():
                        # 計算目標的中心點坐標和檢測框的中心點坐標之間的歐氏距離
                        dist = np.sqrt((center_x - target['center_x']) ** 2 + (center_y - target['center_y']) ** 2)

                        # 如果距離小於閾值，則認為是同一個目標
                        if dist < min_dist:
                            min_dist = dist
                            target_id = tid
                    # 如果找到了最接近的目標，更新該目標的位置信息和框
                    if target_id is not None:
                        targets[target_id]['center_x'] = center_x
                        targets[target_id]['center_y'] = center_y
                        targets[target_id]['bbox'] = (left, top, right, bottom)


                    # 如果沒有找到最接近的目標，則創建一個新的目標
                    else:
                        tid = max(targets.keys(), default=0) + 1
                        targets[tid] = {'center_x': center_x, 'center_y': center_y, 'bbox': (left, top, right, bottom)}

                    if save_img or save_crop or view_img:  #添加 bbox 到圖像
                        if cls == 2 or cls == 5 or cls == 7:
                            label = 'ID:{}'.format(tid)
                            annotator.box_label(xyxy, label, (0, 255, 0))

                    if save_crop: #保存裁剪圖片
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result() #將 self.im 轉換為 numpy
            if view_img:
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow(str(p), im0.shape[1]//3, im0.shape[0]//3)

                cv2.putText(im0,str(count),(100, 90), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 3)
                cv2.imshow(str(p), im0)

                if cv2.waitKey(1) & 0xFF == 27: # 1 millisecond
                    im0.release()
                    cv2.destroyAllWindows()

            #保存結果（帶有檢測的圖像）
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()   # 釋放之前的視頻編寫器
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.MOV'))   #在結果視頻上強制 *.mp4 後綴
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)


        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")


    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    FPS=sum(t)
    LOGGER.info('{:.0f}'.format(FPS))

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  #更新模型（修復 SourceChangeWarning）


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5x.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1080], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', default= True, help='show results')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))

    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)