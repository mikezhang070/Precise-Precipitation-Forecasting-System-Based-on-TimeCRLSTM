# serial_thread.py
import sys
import PyQt5.QtWidgets as qw
from PyQt5.QtCore import QObject, pyqtSignal, QThread
import threading
from PyQt5.QtSerialPort import QSerialPort

class Serial_Qthread_function(QObject):

    signal_Srialstart_function = pyqtSignal()
    signal_pushButton_Open = pyqtSignal(object)
    signal_pushButton_Open_flage = pyqtSignal(object)
    signal_Read_Data = pyqtSignal(object)
    signal_DTR = pyqtSignal(object)
    signal_RTX = pyqtSignal(object)
    signal_Send_data = pyqtSignal(object) 
    signal_Send_data_lenth = pyqtSignal(object) 

    def __init__(self,parent=None):
        super(Serial_Qthread_function,self).__init__(parent)
        #开始调用网络的信号
        print("初始化线程id:",threading.current_thread().ident)
        self.state = 0  # 串口状态 0:未打开 1:串口已打开 2:串口已关闭
    
    def slot_DTR(self,state):
        print("DTR:",state)
        if state == 2:
            self.Serial.setDataTerminalReady(True)
        else:
            self.Serial.setDataTerminalReady(False)


    def slot_RTX(self,state):  # 接收发送框
        print("RTX:",state)
        if state == 2:
            self.Serial.setRequestToSend(True)
        else:
            self.Serial.setRequestToSend(False)

    def slot_pushButton_Open(self,parameter):  # 打开串口
        if self.state == 0:  # 未打开
            print(parameter)
            self.Serial.setPortName(parameter['comboBox_Com'])  # 设置串口名称
            self.Serial.setBaudRate(int(parameter['comboBox_Baud']))  # 设置波特率

            if parameter['comboBox_Stop'] == '1.5':  # 设置停止位
                self.Serial.setStopBits(3)
            elif parameter['comboBox_Stop'] == '2':
                self.Serial.setStopBits(int(parameter['comboBox_Stop']))

            self.Serial.setDataBits(int(parameter['comboBox_Data']))  # 设置数据位
            setpaity = 0
            if parameter['comboBox_Check'] == 'None':  # 设置校验位
                setpaity = 0
            elif parameter['comboBox_Check'] == 'Odd':
                setpaity = 3
            elif parameter['comboBox_Check'] == 'Even':
                setpaity = 2
            self.Serial.setParity(setpaity)
            if self.Serial.open(QSerialPort.ReadWrite):  # 打开串口
                print("串口打开成功")
                self.state = 1  # 串口已打开
                self.signal_pushButton_Open_flage.emit(self.state)  # 发送信号
            else:
                print("串口打开失败")  # 串口打开失败
                self.signal_pushButton_Open_flage.emit(0)  # 发送信号
        else:
            print("关闭串口")
            self.state = 0  # 串口已关闭
            self.Serial.close()  # 关闭串口
            self.signal_pushButton_Open_flage.emit(2)  # 发送信号

    def slot_Send_data(self, send_data):  # 发送数据
        if self.state != 1:
            return
        send_buff = ''
        print("发送数据", send_data.get('Hex'), send_data.get('data'))
        if send_data.get('Hex') == 2:
            send_list = []
            send_text = send_data.get('data', '')
            while send_text != '':
                try:
                    num = int(send_text[0:2], 16)
                except:
                    return
                send_text = send_text[2:].strip()
                send_list.append(num)
            input_s = bytes(send_list).decode()
            if send_data.get('End') == 2:
                send_buff = input_s + '\r\n'
            else:
                send_buff = input_s
        else:
            if send_data.get('End') == 2:
                send_buff = send_data.get('data', '') + '\r\n'
            else:
                send_buff = send_data.get('data', '')

        Byte_data = str.encode(send_buff)  # 字符串转字节数组
        self.Serial.write(Byte_data)  # 发送数据
        self.signal_Send_data_lenth.emit(len(Byte_data))

    def Serial_receive_data(self):  # 串口接收数据
        # print("接收线程数据id:",threading.current_thread().ident)
        # print(self.Serial.readAll().data())
        self.signal_Read_Data.emit(self.Serial.readAll())  # 发送信号

    def SerialInit_function(self):
        print("串口线程id:",threading.current_thread().ident)
        self.Serial = QSerialPort()  # 串口初始化
        self.Serial.readyRead.connect(self.Serial_receive_data)  # 接收信号连接槽函数
        self.Serial.setBaudRate(9600)  # 设置波特率
        self.Serial.setDataBits(8)  # 设置数据位