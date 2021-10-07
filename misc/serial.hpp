/*
 * 这是 TJU Robomasters 上位机源码，未经管理层允许严禁传播给其他人（包括队内以及队外）
 *
 * SerialPort 为串口通信类，实现了对串口底层的基本操作：例如监听发到指定串口的数据、发送指定数据到串口
 * SerialManager 为针对通信协议封装的串口类，管理了接受和发送的数据，数据的解包和装包
 * ElectronicControlParams 保存电控数据的静态类
 */

#pragma once

#define UINT unsigned int

#include<string>
#include <unistd.h>     /*Unix标准函数定义*/
#include <fcntl.h>      /*文件控制定义*/
#include <termios.h>   /*PPSIX终端控制定义*/
#include <memory.h>
#include <vector>
#include "configurations.hpp"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
//g++ -o main main.cpp -lpthread
/** 串口通信类
*
*  本类实现了对串口的基本操作
*  例如监听发到指定串口的数据、发送指定数据到串口
*/
class SerialPort {

public:

    SerialPort(int portNo = 1) {
#ifdef DEVICE_TX 
        const char* Dev[4] = {"/dev/ttyTHS1","/dev/ttyTHS2", "/dev/ttyTHS3",""};
#elif defined DEVICE_MANIFOLD
        const char* Dev[4] = {"/dev/ttyTHS2","/dev/ttyS0","",""};
#elif defined DEVICE_PC
        const char* Dev[4] = {"/dev/ttyUSB0","/dev/ttyUSB1","",""};
#endif
        fd = open(Dev[portNo], O_RDWR | O_NOCTTY | O_NDELAY);//OPEN THE SERIAL 0
        if (fd == -1) {
            perror("Can't Open Serial Port");
        } else printf("Open Serial Port Successful\n");

        /*恢复串口为阻塞状态*/
        if (fcntl(fd, F_SETFL, 0) < 0)
            printf("fcntl failed!\n");
        else
            printf("fcntl=%d\n", fcntl(fd, F_SETFL, 0));

        /*测试是否为终端设备*/
        if (isatty(STDIN_FILENO) == 0)
            printf("standard input is not a terminal device\n");
        else
            printf("isatty success!\n");
        printf("fd-open=%d\n", fd);

    }

    ~SerialPort(void) {
        if (!close(fd)) printf("Close Serial Port Successful\n"); /*关闭串口*/
    }

    /** 初始化串口函数
    *
    *  @param:    portNo 串口编号,默认值为1,即COM1,注意,尽量不要大于9
    *  @param:  UINT baud   波特率,默认为9600 可以设置为115200
    *  @param:  char parity 是否进行奇偶校验,'Y'表示需要奇偶校验,'N'表示不需要奇偶校验
    *  @param:  UINT databits 数据位的个数,默认值为8个数据位
    *  @param:  UINT stopsbits 停止位使用格式,默认值为1
    *  @param:  DWORD dwCommEvents 默认为EV_RXCHAR,即只要收发任意一个字符,则产生一个事件
    *  @return: bool  初始化是否成功
    *  @note:   在使用其他本类提供的函数前,请先调用本函数进行串口的初始化
    *　　　　　   \n本函数提供了一些常用的串口参数设置,若需要自行设置详细的DCB参数,可使用重载函数
    *           \n本串口类析构时会自动关闭串口,无需额外执行关闭串口
    *  @see:
    */
    bool InitPort(UINT baud = 9600, char parity = 'N', UINT databits = 8, UINT stopsbits = 1) {
        struct termios newtio, oldtio;//新老终端接口
        /*保存测试现有串口参数设置，在这里如果串口号等出错，会有相关的出错信息*/
        if (
                tcgetattr(fd, &oldtio
                ) != 0) {
            perror("SetupSerial 1");
            printf("tcgetattr( fd,&oldtio) -> %d\n",
                   tcgetattr(fd, &oldtio
                   ));
            return 0;
        }
        bzero(&newtio, sizeof(newtio));
        /*步骤一，设置字符大小 设置终端接口的控制形式*/
        newtio.c_cflag |= CLOCAL | CREAD; //打开接收端
        newtio.c_cflag &= ~CSIZE; //字符长度掩码
        /*设置停止位 以及 设置数据位*/
        switch (databits) {
            case 7:
                newtio.c_cflag |= CS7;
                break;
            case 8:
                newtio.c_cflag |= CS8;
                break;
        }
        /*设置奇偶校验位*/
        switch (parity) {
            case 'Y': //奇验位
                newtio.c_cflag |= PARENB;
                newtio.c_cflag |= PARODD;
                newtio.c_iflag |= (INPCK | ISTRIP);
                break;
//            case 'E': //偶数
//                newtio.c_iflag |= (INPCK | ISTRIP);
//                newtio.c_cflag |= PARENB;
//                newtio.c_cflag &= ~PARODD;1号是地，最外边。4号是Rx,5号是Tx。


//                break;
            case 'N':  //无奇偶校验位
                newtio.c_cflag &= ~PARENB;
                break;
            default:
                break;
        }
        /*设置波特率*/
        int speed_arr[] = {B921600, B460800, B230400, B115200, B57600, B38400, B19200, B9600, B4800, B2400, B1200, B300,
                           B38400, B19200, B9600, B4800, B2400, B1200, B300,};
        int name_arr[] = {921600, 460800, 230400, 115200, 57600, 38400, 19200, 9600, 4800, 2400, 1200, 300, 38400,
                          19200, 9600, 4800, 2400, 1200, 300,};
        int i;
        for (i = 0; i < sizeof(speed_arr) / sizeof(int); i++) {
            if (baud == name_arr[i]) {
                tcflush(fd, TCIOFLUSH);//fd是打开的串口描述符 清空输入输出缓存
                cfsetispeed(&newtio, speed_arr[i]);//设置输入波特率
                cfsetospeed(&newtio, speed_arr[i]);//设置输出波特率
            }
            tcflush(fd, TCIOFLUSH);
        }
        /*设置停止位*/
        if (stopsbits == 1)
            newtio.c_cflag &= ~CSTOPB; //设置停止位为1位
        else if (stopsbits == 2)
            newtio.c_cflag |= CSTOPB; //停止位为2位
        /*设置等待时间和最小接收字符*/
        newtio.c_cc[VTIME] = 0;
        newtio.c_cc[VMIN] = 0;
        /*处理未接收字符*/
        tcflush(fd, TCIFLUSH);
        /*激活新配置*/
        if ((tcsetattr(fd, TCSANOW, &newtio)) != 0) {
            perror("com set error");
            return 0;
        }
        printf("set done!\n");
        return 1;
    }


/** 向串口写数据
*
*  将缓冲区中的数据写入到串口
*  @param:  unsigned char * pData 指向需要写入串口的数据缓冲区
*  @param:  unsigned int length 需要写入的数据长度
*  @return: bool  操作是否成功
*  @note:   length不要大于pData所指向缓冲区的大小
*  @see:
*/
    bool WriteData(unsigned char *pData, UINT length) {

        if (write(fd, pData, length) == -1) {
            printf("write failed.");
            return 0;
        }
        return 1;

    }

/** 读取串口接收缓冲区中所有数据，并清除当前缓冲区
*
*
*  @param:  char & cRecved 存放读取数据的字符变量
*  @return: int  读出的字节数
*  @note:
*  @see:
*/
    int ReadBuffer(unsigned char cRecved[],int bufflen = 512) {
        return read(fd, cRecved, bufflen);
    }

    int fd;
};


/*
 * 定义指令ＩＤ常量
 * 指令ＩＤ保证大于２００，参数字节小于２００，数据末尾为２００
 * ２０１：
 */
const uchar SEND_ORDER_PTZ_DELTA_ANGLE = 201;
const uchar SEND_ORDER_RECOMMEND_SHOOT_SPEED = 202;
const uchar SEND_ORDER_SHOOT = 203;
const uchar SEND_ORDER_DSF_DONE = 204;
const uchar SEND_ORDER_REQUEST_DATA = 205;
const uchar SEND_ORDER_PTZ_ABSOLUTE_ANGLE = 206;
const uchar SEND_ORDER_MOTION_CONTROL = 207;
const uchar SET_PAN_TILT_CONTROL = 208;

const uchar RECEIVE_ORDER_SELECT_MODULE = 201;
const uchar RECEIVE_ORDER_ENEMY_COLOR = 202;
const uchar RECEIVE_ORDER_CURRENT_SHOOT_SPEED = 203;
const uchar RECEIVE_ORDER_PTZ_SPEED = 204;
const uchar RECEIVE_ORDER_DSF_NEXT_HIT = 205;
const uchar RECEIVE_ORDER_PTZ_ANGLE = 206;
const uchar RECEIVE_ORDER_WORLD_POSITION = 207;
const uchar RECEIVE_MOVING_SPEED = 209;

class ElectronicControlParams
{
public:
    static int shotSpeed;
    static Point2f PTZSpeed,PTZAngle;
    static ArmorColorType teamInfo;
    static bool isSmallEnergy;
    static Point2f worldPosition;
};

int ElectronicControlParams::shotSpeed = 20;
Point2f ElectronicControlParams::PTZSpeed(0,0),ElectronicControlParams::PTZAngle(0,0);
ArmorColorType ElectronicControlParams::teamInfo = ARMOR_BLUE;
bool ElectronicControlParams::isSmallEnergy = true;
Point2f ElectronicControlParams::worldPosition(0,0);

double movingSpeed;

class SerialManager
{
public:

    vector<ModuleBase*> module_list;

    bool DSFContinueFire = true;

    SerialManager(SerialPort *serial)
    {
        UINT Baud = ConfigurationVariables::GetInt("Baud", 115200);
        UINT Parity = ConfigurationVariables::GetInt("Parity", 0);
        UINT Databits = ConfigurationVariables::GetInt("Databits", 8);
        isWorking = serial->InitPort(Baud,Parity ? 'Y':'N',Databits);
        this->serial = serial;
    }

    bool IsWorking()
    {
        return isWorking;
    }

    void RegisterModule(ModuleBase* module)
    {
        module_list.push_back(module);
    }

    void SendData(unsigned char data[], int count, int offset = 0)
    {
        if (isWorking)
        {
            // find existing command
            for (int i=0;i<writeCount;i++)
                if (writeBuff[i] == data[offset])
                {
                    for (int j = 1; j < count; j++)
                        writeBuff[i + j] = data[offset + j];
                    return;
                }
            // no existing command
            for (int i = 0; i < count; i++)
                writeBuff[writeCount++] = data[offset + i];
        }
    }

    void SendPTZDeltaAngle(float dx,float dy)
    {
        uchar ar[5] = {SEND_ORDER_PTZ_DELTA_ANGLE, 0,0,0,0};
        MapToParams(ar[1],ar[2],-60,60,dx);
        MapToParams(ar[3],ar[4],-30,30,dy);
        SendData(ar, 5);
    }

    void SendPTZAbsoluteAngle(float x,float y)
    {
        uchar ar[5] = {SEND_ORDER_PTZ_ABSOLUTE_ANGLE, 0,0,0,0};
#if defined ROBOT_SENTINEL
        MapToParams(ar[1],ar[2],-200.0,200.0,x);
        MapToParams(ar[3],ar[4],-47,23,y);
#elif defined ROBOT_INFANCY
        MapToParams(ar[1],ar[2],-200,200,x);
        MapToParams(ar[3],ar[4],-50,50,y);
#elif defined ROBOT_HERO
        MapToParams(ar[1],ar[2],-200,200,x);
        MapToParams(ar[3],ar[4],-50,50,y);        
#endif
        SendData(ar, 5);
    }

    void SendRecommandedFireSpeed(int speed)
    {
        speed = speed > 25 ? 25 : speed;
        //uchar ar[2] = { 202,speed };
        uchar ar[2] = { SEND_ORDER_RECOMMEND_SHOOT_SPEED,speed };
        SendData(ar, 2);
    }

    //fashezidan
    void SendFiringOrder(bool fire,bool openWheel)
    {
        uchar c = 1;
        if (openWheel && fire) c = 4;
        else if (openWheel) c = 2;
        uchar ar[2] = {SEND_ORDER_SHOOT,c};
        SendData(ar,2);
    }

    void SendContralMode(bool foundTarget){
        uchar c = 0;
        if(foundTarget){
            c = 1;
        }
        uchar ar[2] = {SET_PAN_TILT_CONTROL, c};
        SendData(ar, 2);
    }

    void SendSingleOrder(uchar order)
    {
        uchar ar[1] = { order };
        SendData(ar, 1);
    }

    void SendRequestInitInfo()
    {
        SendSingleOrder(SEND_ORDER_REQUEST_DATA);
    }

    void SendDSFDone()
    {
        SendSingleOrder(SEND_ORDER_DSF_DONE);
    }

    void SendMotionControl(float x,float y,float r)
    {
        uchar ar[7] = {SEND_ORDER_MOTION_CONTROL};
        MapToParams(ar[1],ar[2],-1,1,x);
        MapToParams(ar[3],ar[4],-1,1,y);
        MapToParams(ar[5],ar[6],-1,1,r);
    }

    void FlushData()
    {
//        if (writeCount) {
//            FOREACH(i, writeCount) cout << (0xff & writeBuff[i]) << " ";
//            cout << endl;
//        }
        if (writeCount)
            serial->WriteData(writeBuff, writeCount);
        writeCount = 0;
    }

    void UpdateReadData()
    {
        int count = serial->ReadBuffer(dataBuffer);//读取到的字节数
       // if (count)
        //{
            //cout << "read " << count << endl;
            //FOREACH(i,count)
            //    cout << (dataBuffer[i] & 0xff) << " ";
            //cout << endl;
        //}
        int commandsList[100][2],lstCount = 0; // list structure code,index
        FOREACH(i,count)
        {
            if (dataBuffer[i] > 200) //符合指令条件>200
            {
                bool flag = true; //接收指令标志位设置为1
                FOREACH(j,lstCount) 
                if (commandsList[j][0] == dataBuffer[i])
                {
                    commandsList[j][1] = i;
                    flag = false;
                    break;
                }
                // add new code
                if (flag)  
                {
                    commandsList[lstCount][0] = dataBuffer[i]; //储存参数
                    commandsList[lstCount++][1] = i; //储存第几条指令
                }
            }
        }
        FOREACH(i,lstCount) {
            DealCommand(commandsList[i][0],dataBuffer + commandsList[i][1]);
        }
        //cout << "c" << endl;

    }

    void UpdateCurModule(ImageData frame,float dtTime)
    {
        // if (curModule > -1)
            // curModule = 0;
            //cout << "this setep question" << endl;
            module_list[curModule]->Update(frame,dtTime);
    }

    void EnableModule(int moduleID)//此处为调用modulebase的构造函数，需要指明ID，ID为128表示不调用任何模块
    {
        //cout << "set module : " << (0xff & data[1]) << endl;
        int tgtModule = -2;
        if (moduleID == 128) tgtModule = -1;
        else FOREACH(i, module_list.size())//module_list在图像处理线程中已经被压入相应的模块
        if (module_list[i]->moduleID == moduleID)
        {
            tgtModule = i;//接受到所需的id对应的索引
            break;
        }
        if (tgtModule != -2 && tgtModule != curModule)
        {
            if (curModule > -1)
                module_list[curModule]->DisableModule();    //一次只能运行一个module，所以需要关闭上一个module
            if (tgtModule > -1)
                module_list[tgtModule]->EnableModule();
            curModule = tgtModule;
        }
    }
private:
    int curDealer = -1, paramCount = 0,writeCount = 0 ,curModule = -1;
    unsigned char dataBuffer[1000],paramBuff[100],writeBuff[100];
    bool isWorking = false;
    SerialPort *serial;

    inline int clamp(int t, int min, int max)
    {
        return t < min ? min : (t > max ? max : t);
    }

    void DealCommand(int command,unsigned char* buf)
    {
        switch(command)
        {
            case RECEIVE_ORDER_SELECT_MODULE:
                EnableModule(buf);
                break;
            case RECEIVE_ORDER_ENEMY_COLOR:
                SetTeamInfo(buf);//get the enemy armor color
                break;
            case RECEIVE_ORDER_CURRENT_SHOOT_SPEED:
                SetShootSpeed(buf);
                break;
            case RECEIVE_ORDER_PTZ_SPEED:
                SetPTZSpeed(buf);
                break;
            case RECEIVE_ORDER_DSF_NEXT_HIT:
                ReceivedContinueHit();//大神符持续击打
                break;
            case RECEIVE_ORDER_PTZ_ANGLE:
                SetPTZPosition(buf);//return the current gimbal angle 
                break;
            case RECEIVE_ORDER_WORLD_POSITION:
                SetWorldPosition(buf);//return the predicted angle of the gimbal range from -30 to 30 
                break;
            case RECEIVE_MOVING_SPEED:
                SetMovingSpeed(buf);
                break;
            default:
                break;
        }
    }

    void EnableModule( unsigned char* data)
    {
        EnableModule(data[1] & 0xff);//此处是为了获取模块的ID进行的与操作
    }

    void SetTeamInfo(unsigned char* data)
    {
        if (data[1] == 2)
            ElectronicControlParams::teamInfo = ARMOR_RED;
        else if (data[1] == 0)
            ElectronicControlParams::teamInfo = ARMOR_BLUE;
    }

    void SetMovingSpeed(unsigned char* data){
        movingSpeed = data[1];
    }

    void SetShootSpeed(unsigned char* data)
    {
        ElectronicControlParams::shotSpeed = data[1];

        //cout << "shoot speed :" << (data[1] && 0xff) << endl;
    }

    void SetPTZSpeed(unsigned char* data)
    {
        const float doublePI = 6.2831853f;
        ElectronicControlParams::PTZSpeed = Point2f(doublePI*((200 * data[1] + data[2]) / 20000.0 - 1),
                                                    doublePI *((200 * data[3] + data[4]) / 20000.0 - 1));
//        for (int i=1;i<5;i++)
//            cout << (0xff&data[i]) << " ";
//        cout << endl;
//        cout << ElectronicControlParams::PTZSpeed << endl;
    }
    void ReceivedContinueHit()
    {
        DSFContinueFire = true;
    }

    void SetPTZPosition(unsigned char* data)
    {
#if defined  ROBOT_SENTINEL
        ElectronicControlParams::PTZAngle = Point2f(MapToFloat(data[1],data[2],-200,200),
                                                    MapToFloat(data[3],data[4],-47,23));
#elif defined ROBOT_INFANCY
        ElectronicControlParams::PTZAngle = Point2f(MapToFloat(data[1],data[2],-80,80),
                                                    MapToFloat(data[3],data[4],-50,50));//分别为pitch and yaw轴的角度
#elif defined ROBOT_HERO
        ElectronicControlParams::PTZAngle = Point2f(MapToFloat(data[1],data[2],-80,80),
                                                    MapToFloat(data[3],data[4],-50,50));//分别为pitch and yaw轴的角度
#endif
        //cout << "get position : " << ElectronicControlParams::PTZAngle << endl;
    }

    void SetWorldPosition(unsigned char* data)
    {
        ElectronicControlParams::worldPosition= Point2f(
                MapToFloat(data[1],data[2],-30,30),
                MapToFloat(data[3],data[4],-30,30)
        );
    }

    inline float MapToFloat(int paramA,int paramB,float min,float max)
    {
        return (paramA * 200 + paramB) / 40000.0f * (max - min) + min;
    }

    inline void MapToParams(uchar &paramA,uchar &paramB,float minv,float maxv,float val)
    {
        val = min(maxv,max(minv,val));
        int p = (int)((val-minv)/(maxv-minv)*40000.0f);
        paramA = p / 200; paramB = p % 200;
    }
};
