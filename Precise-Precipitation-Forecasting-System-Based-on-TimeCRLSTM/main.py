import sys
import threading
import time
import csv
from typing import List, Optional
import PyQt5.QtWidgets as qw
from PyQt5.QtCore import QThread, QTimer, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QTextCursor, QColor
from PyQt5.QtSerialPort import QSerialPortInfo
import serial_ui
from serial_thread import Serial_Qthread_function
import matplotlib
matplotlib.use("Qt5Agg")
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
matplotlib.rcParams['axes.unicode_minus'] = False     # 解决负号显示
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

class PredictionModelInterface:
    """外部模型只要实现该接口即可被注入"""
    def predict(self, series: List[float]) -> List[float]:
        raise NotImplementedError


class TrendModel(PredictionModelInterface):
    """默认模型（无训练依赖）：线性趋势外推 + 均值兜底"""
    def __init__(self, window: int = 30, horizon: int = 50):
        self.window = max(5, window)
        self.horizon = max(1, horizon)

    def predict(self, series: List[float]) -> List[float]:
        if not series:
            return [0.0] * self.horizon
        tail = np.asarray(series[-self.window:], dtype=float)
        x = np.arange(len(tail))
        try:
            a, b = np.polyfit(x, tail, deg=1)
            x_future = np.arange(len(tail), len(tail) + self.horizon)
            y_future = a * x_future + b
            return y_future.tolist()
        except Exception:
            base = float(np.mean(tail))
            return [base] * self.horizon


class TimeKANWrapper(PredictionModelInterface):
    """
    可选：当你以后有 TimeKAN 训练好的权重时再启用。
    现在无权重，本包装类不会被自动加载，不影响使用。
    """
    def __init__(self, weights_path: str, device: str = "cpu",
                 seq_len: int = 96, pred_len: int = 24):
        self.weights_path = weights_path
        self.device = device
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model = None  # 延迟构建

    def create_model(self):
        from types import SimpleNamespace
        from models.TimeKAN import Model as TimeKANModel
        import torch
        cfg = SimpleNamespace(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            d_model=64,
            embed="timeF",
            freq="h",
            dropout=0.1,
            output_attention=False,
            use_norm=True,
            class_strategy="projection",
            R=1,
        )
        m = TimeKANModel(cfg)
        state = torch.load(self.weights_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        m.load_state_dict(state, strict=False)
        m.eval()
        return m

    def _ensure_model(self):
        if self.model is None:
            import torch
            self.model = self.create_model()
            self.model.to(torch.device(self.device))

    def predict(self, series: List[float]) -> List[float]:
        import torch
        self._ensure_model()
        arr = np.asarray(series, dtype=np.float32)
        if len(arr) < self.seq_len:
            pad_v = arr[0] if len(arr) > 0 else 0.0
            arr = np.pad(arr, (self.seq_len - len(arr), 0), mode="constant", constant_values=pad_v)
        else:
            arr = arr[-self.seq_len:]
        x = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(-1).to(torch.device(self.device))
        with torch.no_grad():
            y = self.model(x)  # 确保 forward 返回 [B, L] 或 [B, L, 1]
        y = y.squeeze().detach().cpu().numpy().reshape(-1)
        if len(y) > self.pred_len:
            y = y[:self.pred_len]
        elif len(y) < self.pred_len:
            y = np.pad(y, (0, self.pred_len - len(y)), mode="edge")
        return y.tolist()


# ========== 预测线程 ==========
class PredictionWorker(QObject):
    signal_prediction_ready = pyqtSignal(list)    # 发回预测结果
    signal_log = pyqtSignal(str)

    def __init__(self, model: Optional[PredictionModelInterface] = None):
        super().__init__()
        # 没有训练好的模型时，默认使用趋势外推（不会是一条直线）
        self._model: PredictionModelInterface = model or TrendModel()
        self._active = False

    @pyqtSlot(object)
    def set_model(self, model_obj):
        if isinstance(model_obj, PredictionModelInterface):
            self._model = model_obj
            self.signal_log.emit("已注入外部模型接口")
        else:
            self.signal_log.emit("注入模型失败：对象未实现 PredictionModelInterface")

    @pyqtSlot(list)
    def run_predict(self, series: List[float]):
        """单次预测：用于实时数据到来或 CSV 上传后触发"""
        try:
            y_pred = self._model.predict(series)
            self.signal_prediction_ready.emit(y_pred)
        except Exception as e:
            self.signal_log.emit(f"预测异常：{e}")

    @pyqtSlot(bool)
    def set_active(self, flag: bool):
        self._active = flag
        self.signal_log.emit(f"预测状态：{'开启' if flag else '停止'}")

    @pyqtSlot(str)
    def predict_from_csv(self, file_path: str):
        """读取 CSV 的第一列为数值序列并预测（UTF-8/GBK 兼容）"""
        self.signal_log.emit(f"开始读取 CSV：{file_path}")
        series: List[float] = []
        last_err = None
        for enc in ('utf-8', 'gbk'):
            try:
                with open(file_path, 'r', newline='', encoding=enc) as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if not row:
                            continue
                        try:
                            # 若你的数据在第2列，把 row[0] 改成 row[1]
                            series.append(float(row[0]))
                        except ValueError:
                            continue
                if series:
                    self.signal_log.emit(f"CSV({enc}) 加载完成：{len(series)} 条样本")
                    self.run_predict(series)
                    return
            except Exception as e:
                last_err = e
                continue
        self.signal_log.emit(f"CSV 读取失败或无数值列，最后错误：{last_err}")


# 用于将 GUI 发给 PredictionWorker 的信号容器
class PredictBridge(QObject):
    signal_do_predict = pyqtSignal(list)
    signal_set_active = pyqtSignal(bool)
    signal_set_model = pyqtSignal(object)
    signal_predict_csv = pyqtSignal(str)


class SerialFrom(qw.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = serial_ui.Ui_Serial()
        self.ui.setupUi(self)

        print("主线程id:", threading.current_thread().ident)

        # ===== 串口初始化 =====
        self.Interface_Init()
        self.UI_Init()
        self.serial_thread = QThread()
        self.serial_thread_function = Serial_Qthread_function()
        self.serial_thread_function.moveToThread(self.serial_thread)
        self.serial_thread.start()
        self.serial_thread_function.signal_Srialstart_function.connect(
            self.serial_thread_function.SerialInit_function
        )
        self.serial_thread_function.signal_Srialstart_function.emit()
        self.serial_thread_function.signal_pushButton_Open.connect(
            self.serial_thread_function.slot_pushButton_Open
        )
        self.serial_thread_function.signal_pushButton_Open_flage.connect(
            self.slot_pushButton_Open_flage
        )
        self.serial_thread_function.signal_Read_Data.connect(self.slot_ReadData)
        self.serial_thread_function.signal_DTR.connect(
            self.serial_thread_function.slot_DTR
        )
        self.serial_thread_function.signal_RTX.connect(
            self.serial_thread_function.slot_RTX
        )
        self.serial_thread_function.signal_Send_data.connect(
            self.serial_thread_function.slot_Send_data
        )
        self.serial_thread_function.signal_Send_data_lenth.connect(
            self.slot_Send_data_lenth
        )

        # ===== 串口扫描与发送定时器 =====
        self.port_Name = []
        self.time_scan = QTimer()
        self.time_scan.timeout.connect(self.TimeOut_Scan)
        self.time_scan.start(1000)

        self.time_send = QTimer()
        self.time_send.timeout.connect(self.TimeOut_Send)

        self.Receivelenth = 0
        self.Sendlenth = 0

        # ===== 实时渲染画布 =====
        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.grid(True)
        self.ui.horizontalLayout_9.insertWidget(0, self.canvas)

        self.rendering = False
        self.data: List[int] = []

        self.ui.pushButton_StartRender.clicked.connect(self.start_render)
        self.ui.pushButton_StopRender.clicked.connect(self.stop_render)

        # ===== 预测可视化画布 =====
        self.figure_pred = plt.Figure()
        self.canvas_pred = FigureCanvas(self.figure_pred)
        self.ax_pred = self.figure_pred.add_subplot(111)
        self.ax_pred.grid(True)
        if hasattr(self.ui, "predictPlotLayout"):
            self.ui.predictPlotLayout.addWidget(self.canvas_pred)
        else:
            self.ui.gridLayout_2.addWidget(self.canvas_pred, 4, 0, 1, 2)

        # ===== 预测线程与桥接信号 =====
        self.pred_thread = QThread()
        self.pred_worker = PredictionWorker()  # 默认趋势模型
        self.pred_worker.moveToThread(self.pred_thread)
        self.pred_thread.start()

        self.pred_bridge = PredictBridge()
        self.pred_bridge.signal_do_predict.connect(self.pred_worker.run_predict)
        self.pred_bridge.signal_set_active.connect(self.pred_worker.set_active)
        self.pred_bridge.signal_set_model.connect(self.pred_worker.set_model)
        self.pred_bridge.signal_predict_csv.connect(self.pred_worker.predict_from_csv)

        self.pred_worker.signal_prediction_ready.connect(self.on_prediction_ready)
        self.pred_worker.signal_log.connect(self.on_predict_log)

        self.predicting = False     # 是否开启实时预测
        self.last_pred: List[float] = []

        # 预测区按钮
        if hasattr(self.ui, "pushButton_StartPredict"):
            self.ui.pushButton_StartPredict.clicked.connect(self.on_start_predict_clicked)
        if hasattr(self.ui, "pushButton_StopPredict"):
            self.ui.pushButton_StopPredict.clicked.connect(self.on_stop_predict_clicked)
        if hasattr(self.ui, "pushButton_UploadCSV"):
            self.ui.pushButton_UploadCSV.clicked.connect(self.on_upload_csv_clicked)

        # 不自动加载 TimeKAN（你目前没有训练权重），使用默认趋势模型即可
        self.on_predict_log("已启用默认【趋势外推】模型，无需训练即可预测")

    # ===== 实时渲染 =====
    def start_render(self):
        if self.serial_thread_function.state == 1:
            self.rendering = True
            send_data = {'data': 'START_DATA', 'End': 0, 'Hex': 0}
            self.serial_thread_function.signal_Send_data.emit(send_data)

    def stop_render(self):
        if self.rendering:
            self.rendering = False
            send_data = {'data': 'STOP_DATA', 'End': 0, 'Hex': 0}
            self.serial_thread_function.signal_Send_data.emit(send_data)

    # ===== 预测按钮回调 =====
    def on_start_predict_clicked(self):
        self.predicting = True
        self.pred_bridge.signal_set_active.emit(True)
        self.on_predict_log("开始预测按钮已按下")
        # 立即对当前已有数据做一次预测（若有）
        if self.data:
            tail = self.data[-1000:]  # 限制长度，避免过长
            self.pred_bridge.signal_do_predict.emit([float(x) for x in tail])
        else:
            self.on_predict_log("当前没有历史数据，等待串口数据或上传 CSV")

    def on_stop_predict_clicked(self):
        self.predicting = False
        self.pred_bridge.signal_set_active.emit(False)
        self.on_predict_log("已停止预测")

    def on_upload_csv_clicked(self):
        path, _ = qw.QFileDialog.getOpenFileName(
            self, "选择 CSV 文件", "", "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self.on_predict_log(f"选择了 CSV：{path}")
            self.pred_bridge.signal_predict_csv.emit(path)
        else:
            self.on_predict_log("取消选择 CSV")

    # ===== 串口定时与事件 =====
    def TimeOut_Scan(self):
        availablePort = QSerialPortInfo.availablePorts()
        new_port = [p.portName() for p in availablePort]
        if len(self.port_Name) != len(new_port):
            self.port_Name = new_port
            self.ui.comboBox_Com.clear()
            self.ui.comboBox_Com.addItems(self.port_Name)

    def TimeOut_Send(self):
        self.slot_pushButton_Send()

    def Interface_Init(self):
        # 串口参数下拉项
        self.Baud = ('9600', '14400', '19200', '38400', '43000', '57600', '76800', '115200', '128000', '230400', '256000', '460800', '921600', '1843200', '3686400')
        self.Stop = ('1', '1.5', '2')
        self.Data = ('5', '6', '7', '8')
        self.Check = ('None', 'Even', 'Odd')
        self.ui.comboBox_Baud.addItems(self.Baud)
        self.ui.comboBox_Stop.addItems(self.Stop)
        self.ui.comboBox_Data.addItems(self.Data)
        self.ui.comboBox_Check.addItems(self.Check)
        self.ui.checkBox_RTX.stateChanged.connect(self.slot_checkBox_RTX)
        self.ui.checkBox_DTR.stateChanged.connect(self.slot_checkBox_DTR)
        self.ui.checkBox_HexSend.stateChanged.connect(self.slot_checkBox_HexSend)
        self.ui.pushButton_Send.clicked.connect(self.slot_pushButton_Send)
        self.ui.checkBox_TimeSend.stateChanged.connect(self.slot_checkBox_TimeSend)
        self.ui.lineEdit_IntervalTime.setText('1000')
        self.ui.pushButton_ReceiveClean.clicked.connect(self.slot_pushButton_ReceiveClean)
        self.ui.pushButton_SendClean.clicked.connect(self.slot_pushButton_SendClean)

    def UI_Init(self):
        self.ui.pushButton_Open.clicked.connect(self.pushButton_Open)

    def pushButton_Open(self):
        self.set_parameter = {
            'comboBox_Com': self.ui.comboBox_Com.currentText(),
            'comboBox_Baud': self.ui.comboBox_Baud.currentText(),
            'comboBox_Stop': self.ui.comboBox_Stop.currentText(),
            'comboBox_Data': self.ui.comboBox_Data.currentText(),
            'comboBox_Check': self.ui.comboBox_Check.currentText(),
        }
        self.serial_thread_function.signal_pushButton_Open.emit(self.set_parameter)

    def slot_pushButton_Open_flage(self, state):
        print("串口打开状态：", state)
        if state == 0:
            qw.QMessageBox.warning(self, "错误信息", "串口已被占用，打开失败！")
        elif state == 1:
            self.ui.pushButton_Open.setStyleSheet("color: red")
            self.ui.pushButton_Open.setText("关闭串口")
            self.time_scan.stop()
        else:
            self.ui.pushButton_Open.setStyleSheet("color: black")
            self.ui.pushButton_Open.setText("打开串口")
            self.time_scan.start(1000)

    def slot_ReadData(self, data):
        self.Receivelenth += len(data)
        self.ui.label_review.setText("接收:" + str(self.Receivelenth))

        if self.ui.checkBox_TimeView.checkState():
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\\r\\n'
            self.ui.textEdit_Receive.setTextColor(QColor(255, 100, 100))
            self.ui.textEdit_Receive.insertPlainText(time_str)
            self.ui.textEdit_Receive.setTextColor(QColor(0, 0, 0))

        Byte_data = bytes(data)
        filtered_data = bytes([b for b in Byte_data if b not in {0x0D, 0x0A}])

        # 文本显示
        if self.ui.checkBox_HexView.checkState():
            hex_str = ' '.join(f"{b:02x}" for b in filtered_data)
            self.ui.textEdit_Receive.insertPlainText(hex_str + ' ')
        else:
            decoded_data = None
            for enc in ('utf-8', 'gbk', 'latin-1'):
                try:
                    decoded_data = filtered_data.decode(enc)
                    break
                except Exception:
                    continue
            if decoded_data is None:
                decoded_data = filtered_data.decode('utf-8', errors='ignore')
            self.ui.textEdit_Receive.insertPlainText(decoded_data)

        # 始终解析并缓存数据（预测不依赖渲染开关）
        try:
            value = int.from_bytes(filtered_data, byteorder='big')
            self.data.append(value)
        except ValueError:
            pass

        # 实时渲染图（仅在开启渲染时）
        if self.rendering:
            self.ax.clear()
            self.ax.grid(True)
            self.ax.plot(self.data)
            self.canvas.draw()

        # 实时预测（在开始预测后，每次新数据都触发一次）
        if self.predicting and self.data:
            tail = self.data[-1000:]
            self.pred_bridge.signal_do_predict.emit([float(x) for x in tail])

    def slot_checkBox_RTX(self, state):
        self.serial_thread_function.signal_RTX.emit(state)

    def slot_checkBox_DTR(self, state):
        self.serial_thread_function.signal_DTR.emit(state)

    def slot_checkBox_HexSend(self, state):
        if state == 2:
            send_text = self.ui.textEdit_Send.toPlainText()
            Byte_text = str.encode(send_text)
            View_data = ''
            for i in range(len(send_text)):
                View_data += '{:02x}'.format(Byte_text[i]) + ' '
            self.ui.textEdit_Send.setText(View_data)
        else:
            send_list = []
            send_text = self.ui.textEdit_Send.toPlainText()
            while send_text != '':
                try:
                    num = int(send_text[:2], 16)
                except:
                    qw.QMessageBox.warning(self, "错误信息", "输入的不是十六进制数！")
                    return
                send_text = send_text[2:].strip()
                send_list.append(num)
            input_s = bytes(send_list)
            self.ui.textEdit_Send.setText(input_s.decode(errors='ignore'))

    def slot_pushButton_Send(self):
        send_data = {}
        send_data['data'] = self.ui.textEdit_Send.toPlainText()
        send_data['End'] = self.ui.checkBox_End.checkState()
        send_data['Hex'] = self.ui.checkBox_HexSend.checkState()
        self.serial_thread_function.signal_Send_data.emit(send_data)

    def slot_checkBox_TimeSend(self, state):
        if state == 2:
            time_data = self.ui.lineEdit_IntervalTime.text()
            self.time_send.start(int(time_data))
        else:
            self.time_send.stop()

    def slot_pushButton_ReceiveClean(self):
        self.Receivelenth = 0
        self.ui.label_review.setText("接收:0")
        self.ui.textEdit_Receive.clear()

    def slot_pushButton_SendClean(self):
        self.Sendenth = 0
        self.ui.label_send.setText("发送:0")
        self.ui.textEdit_Send.clear()

    def slot_Send_data_lenth(self, lenth):
        self.Sendlenth += lenth
        self.ui.label_send.setText("发送: " + str(self.Sendlenth))

    # ===== 收到预测结果：画到预测页（连续折线） =====
    @pyqtSlot(list)
    def on_prediction_ready(self, y_pred: List[float]):
        self.on_predict_log(f"收到预测结果，长度={len(y_pred)}")
        self.last_pred = y_pred
        self.ax_pred.clear()
        self.ax_pred.grid(True)
        # 历史尾部（不要太长）
        tail_len = max(len(y_pred), 200)
        tail = self.data[-tail_len:] if self.data else []
        if tail:
            x_hist = list(range(len(tail)))
            self.ax_pred.plot(x_hist, tail, label="历史(尾部)")
            # 让预测从历史尾后继续
            start = len(tail)
            x_pred = list(range(start, start + len(y_pred)))
            self.ax_pred.plot(x_pred, y_pred, label="预测")
        else:
            self.ax_pred.plot(list(range(len(y_pred))), y_pred, label="预测")
        self.ax_pred.legend(loc='best')
        self.canvas_pred.draw()

    @pyqtSlot(str)
    def on_predict_log(self, msg: str):
        # 在“接收区”里显示预测日志
        self.ui.textEdit_Receive.insertPlainText(f"[PREDICT] {msg}\n")
        self.ui.textEdit_Receive.moveCursor(QTextCursor.End)


if __name__ == "__main__":
    app = qw.QApplication(sys.argv)
    w = SerialFrom()
    w.show()
    sys.exit(app.exec_())
