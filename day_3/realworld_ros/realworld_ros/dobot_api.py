import socket
import threading
from tkinter import Text, END
import datetime
import numpy as np
import os
import json

alarmControllerFile="files/alarm_controller.json"
alarmServoFile="files/alarm_servo.json"


MyType=np.dtype([('len', np.int16, ),
                ('Reserve', np.int16, (3,) ),
                ('digital_input_bits', np.int64, ),
                ('digital_outputs', np.int64, ),
                ('robot_mode', np.int64, ),
                ('controller_timer', np.int64, ),
                ('run_time', np.int64, ),
                ('test_value', np.int64, ),
                ('safety_mode', np.float64, ),
                ('speed_scaling', np.float64, ),
                ('linear_momentum_norm', np.float64, ),
                ('v_main', np.float64, ),
                ('v_robot', np.float64, ),
                ('i_robot', np.float64, ),
                ('program_state', np.float64, ),
                ('safety_status', np.float64, ),
                ('tool_accelerometer_values', np.float64, (3,)),
                ('elbow_position', np.float64, (3,)),
                ('elbow_velocity', np.float64, (3,)),
                ('q_target', np.float64, (6,)),
                ('qd_target', np.float64,(6,)),
                ('qdd_target', np.float64, (6,)),
                ('i_target', np.float64,(6,)),
                ('m_target', np.float64, (6,)),
                ('q_actual', np.float64, (6,)),
                ('qd_actual', np.float64, (6,)),
                ('i_actual', np.float64, (6,)),
                ('i_control', np.float64, (6,)),
                ('tool_vector_actual', np.float64, (6,)),
                ('TCP_speed_actual', np.float64, (6,)),
                ('TCP_force', np.float64, (6,)),
                ('Tool_vector_target', np.float64, (6,)),
                ('TCP_speed_target', np.float64, (6,)),
                ('motor_temperatures', np.float64, (6,)),
                ('joint_modes', np.float64, (6,)),
                ('v_actual', np.float64, (6,)),
                ('handtype', np.int8, (4,)),
                ('userCoordinate', np.int8, (1,)),
                ('toolCoordinate', np.int8, (1,)),
                ('isRunQueuedCmd', np.int8, (1,)),
                ('isPauseCmdFlag', np.int8, (1,)),
                ('velocityRatio', np.int8, (1,)),
                ('accelerationRatio', np.int8, (1,)),
                ('jerkRatio', np.int8, (1,)),
                ('xyzVelocityRatio', np.int8, (1,)),
                ('rVelocityRatio', np.int8, (1,)),
                ('xyzAccelerationRatio', np.int8, (1,)),
                ('rAccelerationRatio', np.int8, (1,)),
                ('xyzJerkRatio', np.int8, (1,)),
                ('rJerkRatio', np.int8, (1,)),
                ('BrakeStatus', np.int8, (1,)),
                ('EnableStatus', np.int8, (1,)),
                ('DragStatus', np.int8, (1,)),
                ('RunningStatus', np.int8, (1,)),
                ('ErrorStatus', np.int8, (1,)),
                ('JogStatus', np.int8, (1,)),
                ('RobotType', np.int8, (1,)),
                ('DragButtonSignal', np.int8, (1,)),
                ('EnableButtonSignal', np.int8, (1,)),
                ('RecordButtonSignal', np.int8, (1,)),
                ('ReappearButtonSignal', np.int8, (1,)),
                ('JawButtonSignal', np.int8, (1,)),
                ('SixForceOnline', np.int8, (1,)),
                ('Reserve2', np.int8, (82,)),
                ('m_actual[6]', np.float64, (6,)),
                ('load', np.float64, (1,)),
                ('centerX', np.float64, (1,)),
                ('centerY', np.float64, (1,)),
                ('centerZ', np.float64, (1,)),
                ('user', np.float64, (6,)),
                ('tool', np.float64, (6,)),
                ('traceIndex', np.int64,),
                ('SixForceValue', np.int64, (6,)),
                ('TargetQuaternion', np.float64, (4,)),
                ('ActualQuaternion', np.float64, (4,)),
                ('Reserve3', np.int8, (24,)),
                ])


def alarmAlarmJsonFile():
    currrntDirectory=os.path.dirname(__file__)
    jsonContrellorPath=os.path.join(currrntDirectory,alarmControllerFile)
    jsonServoPath=os.path.join(currrntDirectory,alarmServoFile)

    with open(jsonContrellorPath,encoding='utf-8') as f:
        dataController=json.load(f)
    with open(jsonServoPath,encoding='utf-8') as f:
        dataServo=json.load(f)
    return dataController,dataServo


class DobotApi:
    def __init__(self, ip, port, *args):
        self.ip = ip
        self.port = port
        self.socket_dobot = 0
        self.__globalLock = threading.Lock()
        self.text_log: Text = None
        if args:
            self.text_log = args[0]

        if self.port == 29999 or self.port == 30003 or self.port == 30004 or self.port == 30005 or self.port == 30006:
            try:
                self.socket_dobot = socket.socket()
                self.socket_dobot.connect((self.ip, self.port))
            except socket.error:
                print(socket.error)
                raise Exception(
                    f"Unable to set socket connection use port {self.port} !", socket.error)
        else:
            raise Exception(
                f"Connect to dashboard server need use port {self.port} !")

    def log(self, text):
        if self.text_log:
            date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S ")
            self.text_log.insert(END, date+text+"\n")
        else:
            print(text)

    def send_data(self, string):
        try:
            self.log(f"Send to {self.ip}:{self.port}: {string}")
            self.socket_dobot.send(str.encode(string, 'utf-8'))
        except Exception as e:
            print(e)

    def wait_reply(self):
        data = ""
        try:
            data = self.socket_dobot.recv(1024)
        except Exception as e:
            print(e)

        finally:
            if len(data) == 0:
                data_str = data
            else:
                data_str = str(data, encoding="utf-8")
                self.log(f'Receive from {self.ip}:{self.port}: {data_str}')
            return data_str

    def close(self):
        if (self.socket_dobot != 0):
            self.socket_dobot.close()

    def sendRecvMsg(self, string):
        with self.__globalLock:
            self.send_data(string)
            recvData = self.wait_reply()
            return recvData

    def __del__(self):
        self.close()


class DobotApiDashboard(DobotApi):

    def EnableRobot(self,*dynParams):
        string = "EnableRobot("
        for i in range(len(dynParams)):
         if i == len(dynParams)-1:
            string = string + str(dynParams[i])
         else:
            string = string + str(dynParams[i]) + ","
        string = string + ")"
        return self.sendRecvMsg(string)

    def DisableRobot(self):
        string = "DisableRobot()"
        return self.sendRecvMsg(string)

    def ClearError(self):
        string = "ClearError()"
        return self.sendRecvMsg(string)

    def ResetRobot(self):
        string = "ResetRobot()"
        return self.sendRecvMsg(string)

    def SpeedFactor(self, speed):
        string = "SpeedFactor({:d})".format(speed)
        return self.sendRecvMsg(string)

    def User(self, index):
        string = "User({:d})".format(index)
        return self.sendRecvMsg(string)

    def Tool(self, index):
        string = "Tool({:d})".format(index)
        return self.sendRecvMsg(string)

    def RobotMode(self):
        string = "RobotMode()"
        return self.sendRecvMsg(string)

    def PayLoad(self, weight, inertia):
        string = "PayLoad({:f},{:f})".format(weight, inertia)
        return self.sendRecvMsg(string)

    def DO(self, index, status):
        string = "DO({:d},{:d})".format(index, status)
        return self.sendRecvMsg(string)

    def AccJ(self, speed):
        string = "AccJ({:d})".format(speed)
        return self.sendRecvMsg(string)

    def AccL(self, speed):
        string = "AccL({:d})".format(speed)
        return self.sendRecvMsg(string)

    def SpeedJ(self, speed):
        string = "SpeedJ({:d})".format(speed)
        return self.sendRecvMsg(string)

    def SpeedL(self, speed):
        string = "SpeedL({:d})".format(speed)
        return self.sendRecvMsg(string)

    def Arch(self, index):
        string = "Arch({:d})".format(index)
        return self.sendRecvMsg(string)

    def CP(self, ratio):
        string = "CP({:d})".format(ratio)
        return self.sendRecvMsg(string)

    def LimZ(self, value):
        string = "LimZ({:d})".format(value)
        return self.sendRecvMsg(string)

    def RunScript(self, project_name):
        string = "RunScript({:s})".format(project_name)
        return self.sendRecvMsg(string)

    def StopScript(self):
        string = "StopScript()"
        return self.sendRecvMsg(string)

    def PauseScript(self):
        string = "PauseScript()"
        return self.sendRecvMsg(string)

    def ContinueScript(self):
        string = "ContinueScript()"
        return self.sendRecvMsg(string)

    def GetHoldRegs(self, id, addr, count, type=None):
        if type is not None:
          string = "GetHoldRegs({:d},{:d},{:d},{:s})".format(
            id, addr, count, type)
        else:
          string = "GetHoldRegs({:d},{:d},{:d})".format(
            id, addr, count)
        return self.sendRecvMsg(string)

    def SetHoldRegs(self, id, addr, count, table, type=None):
        if type is not None:
         string = "SetHoldRegs({:d},{:d},{:d},{:d})".format(
            id, addr, count, table)
        else:
         string = "SetHoldRegs({:d},{:d},{:d},{:d},{:s})".format(
            id, addr, count, table, type)
        return self.sendRecvMsg(string)

    def GetErrorID(self):
        string = "GetErrorID()"
        return self.sendRecvMsg(string)


    def DOExecute(self,offset1,offset2):
        string = "DOExecute({:d},{:d}".format(offset1,offset2)+")"
        return self.sendRecvMsg(string)

    def ToolDO(self,offset1,offset2):
        string = "ToolDO({:d},{:d}".format(offset1,offset2)+")"
        return self.sendRecvMsg(string)

    def ToolDOExecute(self,offset1,offset2):
        string = "ToolDOExecute({:d},{:d}".format(offset1,offset2)+")"
        return self.sendRecvMsg(string)

    def  SetArmOrientation(self,offset1):
        string = "SetArmOrientation({:d}".format(offset1)+")"
        return self.sendRecvMsg(string)

    def SetPayload(self, offset1, *dynParams):
        string = "SetPayload({:f}".format(
            offset1)
        for params in dynParams:
          string = string +","+ str(params)+","
        string = string + ")"
        return self.sendRecvMsg(string)

    def PositiveSolution(self,offset1,offset2,offset3,offset4,user,tool):
        string = "PositiveSolution({:f},{:f},{:f},{:f},{:d},{:d}".format(offset1,offset2,offset3,offset4,user,tool)+")"
        return self.sendRecvMsg(string)

    def InverseSolution(self,offset1,offset2,offset3,offset4,user,tool,*dynParams):
        string = "InverseSolution({:f},{:f},{:f},{:f},{:d},{:d}".format(offset1,offset2,offset3,offset4,user,tool)
        for params in dynParams:
            print(type(params), params)
            string = string + repr(params)
        string = string + ")"
        return self.sendRecvMsg(string)

    def SetCollisionLevel(self,offset1):
        string = "SetCollisionLevel({:d}".format(offset1)+")"
        return self.sendRecvMsg(string)

    def  GetAngle(self):
        string = "GetAngle()"
        return self.sendRecvMsg(string)

    def  GetPose(self):
        string = "GetPose()"
        return self.sendRecvMsg(string)

    def EmergencyStop(self):
        string = "EmergencyStop()"
        return self.sendRecvMsg(string)


    def ModbusCreate(self,ip,port,slave_id,isRTU):
        string ="ModbusCreate({:s},{:d},{:d},{:d}".format(ip,port,slave_id,isRTU)+")"
        return self.sendRecvMsg(string)

    def ModbusClose(self,offset1):
        string = "ModbusClose({:d}".format(offset1)+")"
        return self.sendRecvMsg(string)

    def GetInBits(self,offset1,offset2,offset3):
        string = "GetInBits({:d},{:d},{:d}".format(offset1,offset2,offset3)+")"
        return self.sendRecvMsg(string)

    def GetInRegs(self,offset1,offset2,offset3,*dynParams):
        string = "GetInRegs({:d},{:d},{:d}".format(offset1,offset2,offset3)
        for params in dynParams:
            print(type(params), params)
            string = string + params[0]
        string = string + ")"
        return self.sendRecvMsg(string)

    def GetCoils(self,offset1,offset2,offset3):
        string = "GetCoils({:d},{:d},{:d}".format(offset1,offset2,offset3)+")"
        return self.sendRecvMsg(string)

    def SetCoils(self,offset1,offset2,offset3,offset4):
        string = "SetCoils({:d},{:d},{:d}".format(offset1,offset2,offset3)+","+ repr(offset4)+")"
        print(str(offset4))
        return self.sendRecvMsg(string)

    def DI(self,offset1):
        string = "DI({:d}".format(offset1)+")"
        return self.sendRecvMsg(string)

    def ToolDI(self,offset1):
        string = "DI({:d}".format(offset1)+")"
        return self.sendRecvMsg(string)

    def DOGroup(self,*dynParams):
        string = "DOGroup("
        for params in dynParams:
            string = string + str(params)+","
        string =string+ ")"
        return self.wait_reply()

    def BrakeControl(self,offset1,offset2):
        string = "BrakeControl({:d},{:d}".format(offset1,offset2)+")"
        return self.sendRecvMsg(string)

    def StartDrag(self):
        string = "StartDrag()"
        return self.sendRecvMsg(string)

    def StopDrag(self):
        string = "StopDrag()"
        return self.sendRecvMsg(string)

    def LoadSwitch(self,offset1):
        string = "LoadSwitch({:d}".format(offset1)+")"
        return self.sendRecvMsg(string)

    def wait(self):
        string = "wait()"
        return self.sendRecvMsg(string)

    def pause(self):
        string = "pause()"
        return self.sendRecvMsg(string)

    def Continue(self):
        string = "continue()"
        return self.sendRecvMsg(string)

class DobotApiMove(DobotApi):

    def MovJ(self, x, y, z, r,*dynParams):
        string = "MovJ({:f},{:f},{:f},{:f}".format(
            x, y, z, r)
        for params in dynParams:
             string =string+ ","+ str(params)
        string =string+ ")"
        print(string)
        return self.sendRecvMsg(string)

    def MovL(self, x, y, z, r,*dynParams):
        string = "MovL({:f},{:f},{:f},{:f}".format(
            x, y, z, r)
        for params in dynParams:
             string =string+ ","+ str(params)
        string =string+ ")"
        print(string)
        return self.sendRecvMsg(string)

    def JointMovJ(self, j1, j2, j3, j4,*dynParams):
        string = "JointMovJ({:f},{:f},{:f},{:f}".format(
            j1, j2, j3, j4)
        for params in dynParams:
            string =string+ ","+ str(params)
        string =string+ ")"
        print(string)
        return self.sendRecvMsg(string)

    def Jump(self):
        print("待定")

    def RelMovJ(self, x, y, z, r,*dynParams):
        string = "RelMovJ({:f},{:f},{:f},{:f}".format(
            x, y, z, r)
        for params in dynParams:
            string =string+ ","+ str(params)
        string =string+ ")"
        return self.sendRecvMsg(string)

    def RelMovL(self, offsetX, offsetY, offsetZ,offsetR,*dynParams):
        string = "RelMovL({:f},{:f},{:f},{:f}".format(offsetX, offsetY, offsetZ,offsetR)
        for params in dynParams:
            string =string+ ","+ str(params)
        string =string+ ")"
        return self.sendRecvMsg(string)

    def MovLIO(self, x, y, z, r, *dynParams):

        string = "MovLIO({:f},{:f},{:f},{:f}".format(
            x, y, z, r)
        for params in dynParams:
            string =string+ ","+ str(params)
        string =string+ ")"
        return self.sendRecvMsg(string)

    def MovJIO(self, x, y, z, r, *dynParams):

        string = "MovJIO({:f},{:f},{:f},{:f}".format(
            x, y, z, r)
        self.log("Send to 192.168.1.6:29999:" + string)
        for params in dynParams:
            string =string+ ","+ str(params)
        string =string+ ")"
        print(string)
        return self.sendRecvMsg(string)

    def Arc(self, x1, y1, z1, r1, x2, y2, z2, r2,*dynParams):
        string = "Arc({:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f}".format(
            x1, y1, z1, r1, x2, y2, z2, r2)
        for params in dynParams:
            string =string+ ","+ str(params)
        string =string+ ")"
        print(string)
        return self.sendRecvMsg(string)

    def Circle(self, x1, y1, z1, r1, x2, y2, z2, r2,count,*dynParams):
        string = "Circle({:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:d}".format(
             x1, y1, z1, r1, x2, y2, z2, r2, count)
        for params in dynParams:
            string = string + ","+ str(params)
        string = string + ")"
        return self.sendRecvMsg(string)

    def MoveJog(self, axis_id=None, *dynParams):
        if axis_id is not None:
          string = "MoveJog({:s}".format(axis_id)
        else:
          string = "MoveJog("
        for params in dynParams:
            string = string + ","+ str(params)
        string = string + ")"
        return self.sendRecvMsg(string)


    def Sync(self):
        string = "Sync()"
        return self.sendRecvMsg(string)

    def RelMovJUser(self, offset_x, offset_y, offset_z, offset_r, user, *dynParams):
        string = "RelMovJUser({:f},{:f},{:f},{:f}, {:d}".format(
            offset_x, offset_y, offset_z, offset_r, user)
        for params in dynParams:
            string = string + ","+ str(params)
        string = string + ")"
        return self.sendRecvMsg(string)

    def RelMovLUser(self, offset_x, offset_y, offset_z, offset_r, user, *dynParams):
        string = "RelMovLUser({:f},{:f},{:f},{:f}, {:d}".format(
            offset_x, offset_y, offset_z, offset_r, user)
        for params in dynParams:
            string = string + ","+ str(params)
        string = string + ")"
        return self.sendRecvMsg(string)

    def RelJointMovJ(self, offset1, offset2, offset3, offset4, *dynParams):
        string = "RelJointMovJ({:f},{:f},{:f},{:f}".format(
            offset1, offset2, offset3, offset4)
        for params in dynParams:
           string = string + ","+ str(params)
        string = string + ")"
        return self.sendRecvMsg(string)

    def MovJExt(self, offset1, *dynParams):
        string = "MovJExt({:f}".format(
            offset1)
        for params in dynParams:
           string = string + ","+ str(params)
        string = string + ")"
        return self.sendRecvMsg(string)

    def SyncAll(self):
        string = "SyncAll()"
        return self.sendRecvMsg(string)


