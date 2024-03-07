import customtkinter
from CTkMessagebox import CTkMessagebox
from tkinter.filedialog import askopenfile
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from untitled0 import get_points, remove_duplicates, is_in_list
import torch
from save import get_last_filename_in_directory


# Define variable :)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class main_window(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.FileName = ""
        self.IsOpenFile = False

        # load yolo model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

        # North
        self.count = 0  # 北上
        self.count2 = 0  # 南下

        self.e = 0 # 東
        self.s = 0 # 南
        self.w = 0 # 西
        self.n = 0 # 北
        self.dist_threshold = 150
        self.origin_n = {'x': 661, 'y': 213}
        self.target_n = {'x': 363, 'y': 270}
        self.list_n = remove_duplicates(get_points(self.origin_n, self.target_n))
        self.origin_w = {'x': 253, 'y': 317}
        self.target_w = {'x': 321, 'y': 591}
        self.list_w = remove_duplicates(get_points(self.origin_w, self.target_w))
        self.origin_e = {'x': 428, 'y': 645}
        self.target_e = {'x': 1055, 'y': 423}
        self.list_e = remove_duplicates(get_points(self.origin_e, self.target_e))
        self.origin_s = {'x': 753, 'y': 214}
        self.target_s = {'x': 1029, 'y': 425}
        self.list_s= remove_duplicates(get_points(self.origin_s, self.target_s))


        FontStyle = customtkinter.CTkFont(size=20, family="Arial")

        self.title("影像辨識與車流偵測")
        self.geometry(f"{1440}x{720}")
        #self.iconbitmap('icon.ico')
        # self.resizable(False, False)

        self.grid_columnconfigure((0), weight=3)
        self.grid_columnconfigure((1), weight=1)
        self.grid_rowconfigure((1), weight=2)
        #
        # Show Result Frame
        self.ResultFrame = customtkinter.CTkFrame(self)
        self.ResultFrame.grid(row=0, column=1, padx=10, pady=10, sticky="n")
        # Direction
        self.label_Direction = customtkinter.CTkLabel(self.ResultFrame, text="方向", font=FontStyle)
        self.label_Direction.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.label_Direction_E = customtkinter.CTkLabel(self.ResultFrame, text="East", font=FontStyle)
        self.label_Direction_E.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.label_Direction_S = customtkinter.CTkLabel(self.ResultFrame, text="South", font=FontStyle)
        self.label_Direction_S.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        self.label_Direction_W = customtkinter.CTkLabel(self.ResultFrame, text="West", font=FontStyle)
        self.label_Direction_W.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

        self.label_Direction_N = customtkinter.CTkLabel(self.ResultFrame, text="North", font=FontStyle)
        self.label_Direction_N.grid(row=4, column=0, padx=10, pady=10, sticky="nsew")

        # Quantity
        self.label_Quantity = customtkinter.CTkLabel(self.ResultFrame, text="數量", font=FontStyle)
        self.label_Quantity.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.label_Quantity_E = customtkinter.CTkLabel(self.ResultFrame, text="0", font=FontStyle)
        self.label_Quantity_E.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        self.label_Quantity_S = customtkinter.CTkLabel(self.ResultFrame, text="0", font=FontStyle)
        self.label_Quantity_S.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")

        self.label_Quantity_W = customtkinter.CTkLabel(self.ResultFrame, text="0", font=FontStyle)
        self.label_Quantity_W.grid(row=3, column=1, padx=10, pady=10, sticky="nsew")

        self.label_Quantity_N = customtkinter.CTkLabel(self.ResultFrame, text="0", font=FontStyle)
        self.label_Quantity_N.grid(row=4, column=1, padx=10, pady=10, sticky="nsew")

        # Average Speed
        # self.label_AverageSpeed = customtkinter.CTkLabel(self.ResultFrame, text="平均速度(km/h)", font=FontStyle)
        # self.label_AverageSpeed.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        #
        # self.label_AverageSpeed_E = customtkinter.CTkLabel(self.ResultFrame, text="0", font=FontStyle)
        # self.label_AverageSpeed_E.grid(row=1, column=2, padx=10, pady=10, sticky="nsew")
        #
        # self.label_AverageSpeed_S = customtkinter.CTkLabel(self.ResultFrame, text="0", font=FontStyle)
        # self.label_AverageSpeed_S.grid(row=2, column=2, padx=10, pady=10, sticky="nsew")
        #
        # self.label_AverageSpeed_W = customtkinter.CTkLabel(self.ResultFrame, text="0", font=FontStyle)
        # self.label_AverageSpeed_W.grid(row=3, column=2, padx=10, pady=10, sticky="nsew")
        #
        # self.label_AverageSpeed_N = customtkinter.CTkLabel(self.ResultFrame, text="0", font=FontStyle)
        # self.label_AverageSpeed_N.grid(row=4, column=2, padx=10, pady=10, sticky="nsew")

        # Congestion
        self.label_Congestion = customtkinter.CTkLabel(self.ResultFrame, text="擁擠度", font=FontStyle)
        self.label_Congestion.grid(row=0, column=3, padx=10, pady=10, sticky="nsew")

        self.label_Congestion_E = customtkinter.CTkLabel(self.ResultFrame, text="低", font=FontStyle)
        self.label_Congestion_E.grid(row=1, column=3, padx=10, pady=10, sticky="nsew")

        self.label_Congestion_S = customtkinter.CTkLabel(self.ResultFrame, text="低", font=FontStyle)
        self.label_Congestion_S.grid(row=2, column=3, padx=10, pady=10, sticky="nsew")

        self.label_Congestion_W = customtkinter.CTkLabel(self.ResultFrame, text="低", font=FontStyle)
        self.label_Congestion_W.grid(row=3, column=3, padx=10, pady=10, sticky="nsew")

        self.label_Congestion_N = customtkinter.CTkLabel(self.ResultFrame, text="低", font=FontStyle)
        self.label_Congestion_N.grid(row=4, column=3, padx=10, pady=10, sticky="nsew")

        # Play Video Frame
        self.PlayVideoFrame = customtkinter.CTkFrame(self)
        self.PlayVideoFrame.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nswe")

        self.label_Video = customtkinter.CTkLabel(self.PlayVideoFrame, text="請選擇影片", font=customtkinter.CTkFont(size=20, family="Arial"))
        self.label_Video.place(relx=0.5, rely=0.5, anchor="center")

        # Result Graph Frame
        self.ResultGraphFrame = customtkinter.CTkFrame(self)
        self.ResultGraphFrame.grid(row=1, column=1, rowspan=2, padx=10, pady=10, sticky="nswe")

        self.label_GraphTitle = customtkinter.CTkLabel(self.ResultGraphFrame, text="各方向車流量", font=customtkinter.CTkFont(size=20, family="Arial", weight="bold"))
        self.label_GraphTitle.place(relx=0.5, rely=0.06, anchor='center')
        self.canvas = customtkinter.CTkCanvas(self.ResultGraphFrame, width=350, height=340)
        self.canvas.place(relx=0.5, rely=0.55, anchor="center")

        # Button Frame
        self.ButtonFrame = customtkinter.CTkFrame(self)
        self.ButtonFrame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        self.OpenFileBtn = customtkinter.CTkButton(self.ButtonFrame, text='選擇影片',
                                                   font=customtkinter.CTkFont(size=20, family="Arial"),
                                                   command=lambda:self.open_file())
        self.OpenFileBtn.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        # self.OpenLiveBtn = customtkinter.CTkButton(self.ButtonFrame, text='選擇直播',
        #                                       font=customtkinter.CTkFont(size=20, family="Arial"))
        # self.OpenLiveBtn.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        self.StartBtn = customtkinter.CTkButton(self.ButtonFrame, text='雙向道',
                                               font=customtkinter.CTkFont(size=20, family="Arial"),
                                               command=lambda:self.start())
        self.StartBtn.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        self.Start1Btn = customtkinter.CTkButton(self.ButtonFrame, text='十字路口',
                                                font=customtkinter.CTkFont(size=20, family="Arial"),
                                                command=lambda: self.start1())
        self.Start1Btn.grid(row=0, column=2, padx=10, pady=10, sticky="n")

        self.OpenPathBtn = customtkinter.CTkButton(self.ButtonFrame, text='開啟資料夾',
                                                    font=customtkinter.CTkFont(size=20, family="Arial"),
                                                    command=lambda:self.OpenFilePath())
        self.OpenPathBtn.grid(row=0, column=3, padx=10, pady=10, sticky="n")

        # self.DrawResult()

    def open_file(self):
        file = askopenfile(mode='r', filetypes=[('Video Files', ["*.mp4"])])
        if file is not None:
            self.FileName = file.name
            print(self.FileName)
            self.IsOpenFile = True

    def start1(self):
        if self.IsOpenFile:
            self.CapVideo1(True)
        else:
           CTkMessagebox(title="Error", message="未選擇檔案！", icon="cancel")

    def CapVideo1(self, IsOpenVideo):
        if IsOpenVideo is True:
            self.cap = cv2.VideoCapture(self.FileName)
            self.label_Video.configure(text="")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            # 設定儲存位置
            directory_path = "runs/detect"

            # 取得路徑中最後一個檔案
            last_filename = get_last_filename_in_directory(directory_path)

            # 避免檔案重複
            if last_filename is not None:
                file_number = int(last_filename.split('.')[0].split('_')[-1])
                new_file_number = file_number + 1
                new_filename = f"result_{new_file_number}.mp4"
            else:
                new_filename = "result_1.mp4"
            self.output_file = os.path.join(directory_path, new_filename)
            self.out = cv2.VideoWriter(self.output_file, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (1240, 810))
            self.update1()

    def DrawResult1(self):
        data = [self.e, self.s, self.w, self.n]
        dir = ["E", "S", "W", "N"]
        fig = plt.figure(figsize=(3.5, 3.5))
        plt.subplot()
        plt.pie(data,
            radius=1,
            labels=dir,
            textprops={'weight':'bold', 'size':12},
            autopct='%.1f%%',
            wedgeprops={'linewidth':3, 'edgecolor':'w'})
        # plt.show()
        self.canvas1 = FigureCanvasTkAgg(fig, self.canvas)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().place(relx=0.5, rely=0.5, anchor="center")

    def update1(self):
        # Disabled Button
        self.OpenFileBtn.configure(state='disabled')
        # self.OpenLiveBtn.configure(state='disabled')
        self.Start1Btn.configure(state='disabled')
        self.OpenPathBtn.configure(state='disabled')

        ret, frame = self.cap.read()
        if ret == True:

            frame = cv2.resize(frame, (1240, 810))  # (1240, 810)
            results = self.model(frame)
            detections = results.xyxy[0]  # 獲取偵測到的物件和其邊界框座標

            for detection in detections:
                class_idx = int(detection[-1].item())  # 獲取物件的類別標籤
                if class_idx in [2, 7]:  # 只取汽車和卡車
                    xmin, ymin, xmax, ymax = map(int, detection[:4].tolist())  # 解析邊界框座標
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)  # 繪製邊界框

                    # 計算中心點座標
                    center_x = int((xmin + xmax) / 2)
                    center_y = int((ymin + ymax) / 2)
                    center = (center_x, center_y)
                    # 計數車輛並繪製辨識框
                    if (is_in_list(self.list_s, center)):
                        cv2.circle(frame, (center_x, center_y), 8, (0, 255, 255), -1)
                        self.s += 1

                    if (is_in_list(self.list_n, center)):
                        cv2.circle(frame, (center_x, center_y), 8, (0, 255, 255), -1)
                        self.n += 1

                    if (is_in_list(self.list_e, center)):
                        cv2.circle(frame, (center_x, center_y), 8, (0, 255, 255), -1)
                        self.e += 1

                    if (is_in_list(self.list_w, center)):
                        cv2.circle(frame, (center_x, center_y), 8, (0, 255, 255), -1)
                        self.w += 1

            # 標記FPS及數量

            cv2.putText(frame, f'FPS: {self.fps}',
                        (500, 40), cv2.FONT_HERSHEY_PLAIN, 3, WHITE, 3)
            cv2.putText(frame, f'South: {self.s}',
                        (50, 775), cv2.FONT_HERSHEY_PLAIN, 3, WHITE, 3)
            cv2.putText(frame, f'East: {self.e}',
                        (975, 775), cv2.FONT_HERSHEY_PLAIN, 3, WHITE, 3)
            cv2.putText(frame, f'North: {self.n}',
                        (975, 40), cv2.FONT_HERSHEY_PLAIN, 3, WHITE, 3)
            cv2.putText(frame, f'West: {self.w}',
                        (50, 40), cv2.FONT_HERSHEY_PLAIN, 3, WHITE, 3)

            self.out.write(frame)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

            img = Image.fromarray(cv2image).resize((1080, 600))
            imgtks = ImageTk.PhotoImage(image=img)

            self.label_Video.imgtk = imgtks
            self.label_Video.configure(image=imgtks)
            self.label_Video.after(2, self.update1)

            # update label
            self.label_Quantity_N.configure(text=str(self.n))
            self.label_Quantity_S.configure(text=str(self.s))
            self.label_Quantity_E.configure(text=str(self.e))
            self.label_Quantity_W.configure(text=str(self.w))
        else:
            self.cap.release()
            self.out.release()
            self.label_Video.configure(image="", text="請選擇影片", font=customtkinter.CTkFont(size=20, family="Arial"))
            self.DrawResult1()
            self.IsOpenFile = False
            self.OpenFileBtn.configure(state='normal')
            # self.OpenLiveBtn.configure(state='normal')
            self.StartBtn.configure(state='normal')
            self.Start1Btn.configure(state='normal')
            self.OpenPathBtn.configure(state='normal')
        #self.UpdateLabel()

    def start(self):
        if self.IsOpenFile:
            self.CapVideo(True)

        else:
           CTkMessagebox(title="Error", message="未選擇檔案！", icon="cancel")

    def CapVideo(self, IsOpenVideo):
        if IsOpenVideo is True:
            self.cap = cv2.VideoCapture(self.FileName)
            self.label_Video.configure(text="")
            # 標記FPS及數量
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)

            # 設定儲存位置
            directory_path = "runs/detect"

            # 取得路徑中最後一個檔案
            last_filename = get_last_filename_in_directory(directory_path)

            # 避免檔案重複
            if last_filename is not None:
                file_number = int(last_filename.split('.')[0].split('_')[-1])
                new_file_number = file_number + 1
                new_filename = f"result_{new_file_number}.mp4"
            else:
                new_filename = "result_1.mp4"
            self.output_file  = os.path.join(directory_path, new_filename)
            self.out = cv2.VideoWriter(self.output_file, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (1240, 810))
            self.update()
    def update(self):
        # Disabled Button
        self.OpenFileBtn.configure(state='disabled')
        # self.OpenLiveBtn.configure(state='disabled')
        self.StartBtn.configure(state='disabled')
        self.OpenPathBtn.configure(state='disabled')
        ret, frame = self.cap.read()
        if ret == True:

            frame = cv2.resize(frame, (1240, 810))  # (1240, 810)
            results = self.model(frame)
            detections = results.xyxy[0]  # 獲取偵測到的物件和其邊界框座標

            for detection in detections:
                class_idx = int(detection[-1].item())  # 獲取物件的類別標籤
                if class_idx in [2, 7]:  # 只取汽車和卡車
                    xmin, ymin, xmax, ymax = map(int, detection[:4].tolist())  # 解析邊界框座標
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)  # 繪製邊界框

                    # 計算中心點座標
                    center_x = int((xmin + xmax) / 2)
                    center_y = int((ymin + ymax) / 2)
                    # 計數車輛並繪製辨識框
                    if (550 < center_y < 555):
                        cv2.circle(frame, (center_x, center_y), 3, RED, -1)
                        if (center_x > 600):  # 北上數量+1
                            self.count += 1
                            self.label_Quantity_N.configure(text=str(self.count))
                        else:  # 南下數量+1
                            self.count2 += 1
                            self.label_Quantity_S.configure(text=str(self.count2))
                    else:
                        cv2.circle(frame, (center_x, center_y), 3, WHITE, -1)


            cv2.putText(frame, f'FPS: {self.fps}',
                        (500, 40), cv2.FONT_HERSHEY_PLAIN, 3, WHITE, 3)
            cv2.putText(frame, f'North: {self.count}',
                        (975, 40), cv2.FONT_HERSHEY_PLAIN, 3, WHITE, 3)
            cv2.putText(frame, f'South: {self.count2}',
                        (50, 40), cv2.FONT_HERSHEY_PLAIN, 3, WHITE, 3)

            self.out.write(frame)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

            img = Image.fromarray(cv2image).resize((1080, 600))
            imgtks = ImageTk.PhotoImage(image=img)
            self.label_Video.imgtk = imgtks
            self.label_Video.configure(image=imgtks)
            self.label_Video.after(2, self.update)


        else:
            self.cap.release()
            self.out.release()
            self.label_Video.configure(image="", text="請選擇影片", font=customtkinter.CTkFont(size=20, family="Arial"))
            self.DrawResult()
            self.IsOpenFile = False
            self.OpenFileBtn.configure(state='normal')
            self.StartBtn.configure(state='normal')
            self.OpenPathBtn.configure(state='normal')



    def DrawResult(self):
        data = [0, self.count2, 0, self.count]
        dir = ["E", "S", "W", "N"]
        fig = plt.figure(figsize=(3.5, 3.5))
        plt.subplot()
        plt.pie(data,
            radius=1,
            labels=dir,
            textprops={'weight':'bold', 'size':12},
            autopct='%.1f%%',
            wedgeprops={'linewidth':3, 'edgecolor':'w'})
        # plt.show()
        self.canvas1 = FigureCanvasTkAgg(fig, self.canvas)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().place(relx=0.5, rely=0.5, anchor="center")

    def UpdateLabel(self, label, num):
        # This is change label text method
        # self.label_Direction.configure(text=str)
        pass

    def OpenFilePath(self):
        self.path = 'runs\detect'
        os.startfile(self.path)


if __name__ == "__main__":
    app = main_window()
    app.mainloop()
