/***
 * When I wrote this, only God and I understood what I was doing
 * Now, God only knows
 *                    _ooOoo_
 *                   o8888888o
 *                   88" . "88
 *                   (| -_- |)
 *                    O\ = /O
 *                ____/`---'\____
 *              .   ' \\| |// `.
 *               / \\||| : |||// \
 *             / _||||| -:- |||||- \
 *               | | \\\ - /// | |
 *             | \_| ''\---/'' | |
 *              \ .-\__ `-` ___/-. /
 *           ___`. .' /--.--\ `. . __
 *        ."" '< `.___\_<|>_/___.' >'"".
 *       | | : `- \`.;`\ _ /`;.`/ - ` : | |
 *         \ \ `-. \_ __\ /__ _/ .-` / /
 * ======`-.____`-.___\_____/___.-`____.-'======
 *                    `=---='
 *.............................................
 *          佛祖保佑             永无BUG
 *
 ***/


#include "custom_definations.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include "armorDetector.hpp"
#include "serial.hpp"
#include <thread>
#include "camview.hpp"         
#include <unistd.h>
#include "armorTracker.hpp"
#include "armorHiter.hpp"
#include "dnnManager.hpp"
#include "DfcDetector.hpp"
#include "DfcHiter.hpp"
using namespace cv;
using namespace std;


void ProcessFullFunction(ImageData &frame);
void ProcessAlgorithmFunction(ImageData &frame);
void ArmorDetectDebug(ImageData &frame);
void DfcDetectDebug(ImageData &frame);
double lastt = getTickCount(), curt = lastt,checkpoint_time = lastt,dtTime = 0;
int fps  = 0,frames = 0;

// returns delta time
void UpdateFPS()
{
    frames++;
    curt = (double)getTickCount();
    //getTickFrequency 返回时钟频率
    //如果当前时钟减去上次输出时钟计数器为运行频率，则统计当前frames
    if (curt - checkpoint_time >= getTickFrequency()) 
    {
        checkpoint_time = curt;
        fps = frames;
        frames = 0;
        cout <<"fps:"<< fps << endl;
    }
    dtTime = (curt - lastt) / getTickFrequency();
    lastt = curt;  
}

void DisplayFPS(Mat &m)
{
    char fpsstr[30];
    sprintf(fpsstr,"fps:%d",fps);
    putText(m,fpsstr,Point(20,20),FONT_HERSHEY_SIMPLEX,1.0,Scalar(0,0,0),2);

}

void capture_init(VideoCapture &capture)
{
    //设置帧缓存,capture.set返回1则设置成功，返回0则设置失败
    cout<<"帧缓存设置："<<capture.set(CV_CAP_PROP_BUFFERSIZE,2)<<endl; 
    //设置分辨率
    switch (ConfigurationVariables::GetInt("resolutionType",0))
    {
    case 0:
        capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
        capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
        break;
    case 1 :
        capture.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
        capture.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
    case 2:
        capture.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
        capture.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
    default:
        break;
    }  
    cout<<"图片压缩格式设置："<<capture.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'))<<endl; 
    cout<<"手动曝光设置:"<<capture.set(CV_CAP_PROP_AUTO_EXPOSURE,0.25)<<endl;
    cout<<"曝光度设置："<<capture.set(CV_CAP_PROP_EXPOSURE, ConfigurationVariables::GetInt("exposure",30)*0.0002)<<endl;
    cout<<"曝光度："<<capture.get(CV_CAP_PROP_EXPOSURE)<<endl;
    //0-1 映射到qv4l2的1-5000
}

#if defined DEVICE_TX
    VideoCapture rmcap("/dev/video0");
    VideoWriter rmvw;
#elif defined DEVICE_MANIFOLD
    VideoCapture rmcap("/dev/video0");
    VideoWriter rmvw;
#elif defined DEVICE_PC
    VideoCapture rmcap("/home/alexliu/lxc/TJURobomaster/TJU-TEST-VERSION/kcf/2021-01-23 13:09:30.avi");
    VideoWriter rmvw;
#endif

bool threadContinueFlag = true;
char waitedKey = 0;
bool onpause = false;
Lock<ImageData> frameData;//the framedata includes the data of every image
Lock<ImageData> writeData; //videowrite data 
// modules definations
SerialManager *serial_ptr = NULL;
CameraView *camview_ptr = NULL;
ArmorBaseDetector *armor_detector_ptr = NULL;
ArmorTrackerBase *armor_tracker_ptr = NULL;
ArmorHiter *armor_hiter_ptr = NULL;
DnnManager *dnn_manager_ptr = NULL;

EllipseDfcDetector *dfc_detector_ptr = NULL;
DfcHiter * dfc_hiter_ptr = NULL;

ArmorTrackerBase* anti_top_ptr=NULL;

Point2f res;

// 图像收集线程
void ImageCollectThread()
{
    SerialPort serialPort(ConfigurationVariables::GetInt("Port",1));
    SerialManager serialManager(&serialPort);
    serial_ptr = &serialManager;

    int frameIndex = 0;
    int writeIndex = 0;
    Mat img;
    while(threadContinueFlag)
    {
        try{
            // Get Image
            rmcap >> img;
            if (ConfigurationVariables::resolutionType == 0)
                resize(img,img,Size(960,720));
            // 更新当前帧的电控参数
            serialManager.UpdateReadData();
            frameData.Lock();   //frameData 加锁，防止其他线程在更改变量时调用 将串口下位机发送的指令赋值给framedata
            //{
                frameData.variable.ptzSpeed = ElectronicControlParams::PTZSpeed;
                frameData.variable.ptzAngle = ElectronicControlParams::PTZAngle;
                frameData.variable.worldPosition = ElectronicControlParams::worldPosition;
                frameData.variable.shootSpeed = ElectronicControlParams::shotSpeed;
                cout << "shootSpeed: " << ElectronicControlParams::shotSpeed << endl;
                // frameData.variable.shootSpeed = 16;

                // 告知处理线程，图像准备完成
                frameData.variable.image = img;
                frameData.variable.index = frameIndex++;
                // DEBUG_DISPLAY(frameData.variable.image);
            //}
            frameData.Unlock(); //数据更改完成，解锁

            writeData.Lock();
            {
            writeData.variable.image = img;
            writeData.variable.index = writeIndex++;
            }
            writeData.Unlock();

        }
        catch(...)
        {
            cout << "Error in Collect." << endl;
        }
    }
}
//图像处理线程
void ImageProcessThread()
{
    // 初始化各模块
    char camparams[100] ;
    // //format output to camparams
    //注意如果更改了源码保存位置一定修改FILEDIR路径
    sprintf(camparams,FILEDIR(camparams_%d.xml),ConfigurationVariables::resolutionType);    //根据分辨率修改内参矩阵
    CameraView camview(camparams);// load the parameters of the camera
    camview_ptr = &camview;

    PolyMatchArmorDetector armor_detector;
    armor_detector_ptr = (ArmorBaseDetector*)(&armor_detector);
    armor_detector_ptr->SetTargetArmor(2 - ElectronicControlParams::teamInfo);


    // KFCPredictor linear_predictor;
    KalmanPredictor angel_predictor; 

    DnnManager dnnManager(FILEDIR(model/armor.cfg), FILEDIR(model/armor.weights)); // file names
    dnn_manager_ptr = &dnnManager;

    //PredictKFCArmorTracker predict_pid_tracker(camview_ptr,armor_detector_ptr,dnn_manager_ptr,&linear_predictor);
    PredictAngelArmorTracker predict_angel_tracker(camview_ptr,armor_detector_ptr,dnn_manager_ptr,&angel_predictor);
    armor_tracker_ptr = &predict_angel_tracker;
    // anti_top apply_anti_top;
    // anti_topArmorTracker anti_top_pridector(camview_ptr,armor_detector_ptr,dnn_manager_ptr,&apply_anti_top);
    // anti_top_ptr=&anti_top_pridector;



    // 等待串口类被初始化
    while(!serial_ptr);
    
#if defined ROBOT_SENTINEL
    SentinelArmorHiter sentinelHiter(serial_ptr,armor_tracker_ptr);
    armor_hiter_ptr = &sentinelHiter;
#elif defined ROBOT_INFANCY
    InfancyArmorHiter infancyHiter(serial_ptr,armor_tracker_ptr);
    armor_hiter_ptr = &infancyHiter;

    EllipseDfcDetector dfcDetector;
    dfc_detector_ptr = &dfcDetector;
    dfc_detector_ptr->SetTargetArmor(2 - ElectronicControlParams::teamInfo); //设置敌方装甲板颜色

    InfancyDfcHiter infancy_dfc_Hiter(serial_ptr,dfc_detector_ptr);
    dfc_hiter_ptr = &infancy_dfc_Hiter;

 #elif defined ROBOT_HERO
    HeroArmorHiter heroHiter(serial_ptr,armor_tracker_ptr);
    armor_hiter_ptr = &heroHiter;
#endif

    //初始化serialManager中的模块列表
    //压入2个模块 装甲识别 2； 大风车 4；
    serial_ptr->RegisterModule(armor_hiter_ptr);    //将模块push进串口管理器
#ifdef ROBOT_INFANCY
    serial_ptr->RegisterModule(dfc_hiter_ptr);
#endif

    int lastIndex = 0;
    ImageData processFrame;
    while(threadContinueFlag)
    {
        if (waitedKey == 'p')
        {
            waitedKey = 0;
            onpause = !onpause;
        }
        if (onpause) continue;
        try
        {
            // 当前图像获取线程还不能提供最新图像
            if (lastIndex >= frameData.variable.index) continue;
//LOG(A)
            // 拷贝获取线程得到的图像，防止内存错误
            frameData.Lock();
                frameData.variable.copyTo(processFrame);
            frameData.Unlock();

//LOG(B)    
            lastIndex = processFrame.index;
            if (processFrame.image.empty()) break;
            UpdateFPS();

            // 处理该帧数据
            if (ConfigurationVariables::MainEntry < 4)
            {
                ProcessFullFunction(processFrame);
            }
            else 
                ProcessAlgorithmFunction(processFrame);

//LOG(C)
            //DisplayFPS(processFrame.image);
            //cout << "final step" << endl;
            waitedKey = 0;

        }
        catch(...)
        {
            cout << "Error in process." << endl;
        }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        // 串口发送当前帧数据
        serial_ptr->FlushData();
//LOG(D)
        // 每秒更新一次configuration
        if(frames == 0 && ConfigurationVariables::KeepUpdateConfiguration)
        {
            // reloading configuration variables
            #ifdef ROBOT_INFANCY
                ConfigurationVariables::ReadConfiguration(true,FILEDIR(config/config_infantry.ini));
            #elif defined ROBOT_HERO
                ConfigurationVariables::ReadConfiguration(true,FILEDIR(config/config_hero.ini));
            #elif defined ROBOT_SENTINEL
                ConfigurationVariables::ReadConfiguration(true,FILEDIR(config/config_sentinel.ini));
            #endif
            if (!ConfigurationVariables::Loaded) cout << "Error when update configuration variables." << endl;
        }
    }
}
//图像显示线程
void ImageDisplayThread()
{
    while(threadContinueFlag)
    {
        if (ConfigurationVariables::DebugMode)
        {
            try{
                if (!onpause || DebugDisplayManager::imgs.size())
                    DebugDisplayManager::DisplayAll();
                int c = waitKey(1);
                if (c != -1) waitedKey = (char)c;
                if (waitedKey == 'x') threadContinueFlag = false;
            }catch(...){
                cout << "Error in Display." << endl;
            }
        }
	else
	    sleep(1);
    }
}
//视频录制线程
void VideoWriteThread()
{
    /**********************rmvw init************************/
    int write_fps = rmcap.get(CV_CAP_PROP_FPS);  //获取摄像机帧率
    String write_name = getCurrentTimeStr() + ".avi";   //video名称
	if (write_fps <= 0) write_fps = 25;
    Size write_size;                                //video size
    if (ConfigurationVariables::GetInt("resolutionType",0) == 0)
        write_size = Size(960,720); //480p的分辨率经过了resize，所以需要重新处理下
    else
        write_size = Size(rmcap.get(CAP_PROP_FRAME_WIDTH),rmcap.get(CAP_PROP_FRAME_HEIGHT));
    
	//创建视频文件
	rmvw.open(write_name,                          //路径
		VideoWriter::fourcc('M', 'J', 'P', 'G'), //编码格式
		write_fps,                                  //帧率
		write_size                                  //尺寸
		);
	if (!rmvw.isOpened())
	{
		cout << "VideoWriter open failed!" << endl;
		getchar();

	}
	cout << "VideoWriter open success!" << endl;
    /**********************Video write************************/
    ImageData recordFrame;
    int lastIndex = 0;
    while(threadContinueFlag)    
	{
        try
        {
            if (lastIndex >= writeData.variable.index) continue;
            writeData.Lock();
                writeData.variable.copyTo(recordFrame);
            writeData.Unlock();
            lastIndex = recordFrame.index;
            if (recordFrame.image.empty()) 
            {
                break;
            }
            //写入视频文件
            rmvw.write(recordFrame.image);
            }
        catch(...)
        {
            cout << "Error in Write." << endl;
        }
        

	}
}

int main() {
    /*********************************Load configration***************************************/
    #ifdef ROBOT_INFANCY
        ConfigurationVariables::ReadConfiguration(true,FILEDIR(config/config_infantry.ini));
    #elif defined ROBOT_HERO
        ConfigurationVariables::ReadConfiguration(true,FILEDIR(config/config_hero.ini));
    #elif defined ROBOT_SENTINEL
        ConfigurationVariables::ReadConfiguration(true,FILEDIR(config/config_sentinel.ini));
    #endif
    if (!ConfigurationVariables::Loaded)
        cout << "Load Configuration Failed. Using default values." << endl;
    else
        cout << "Configuration Loaded Successfully." << endl;
    ElectronicControlParams::teamInfo = ConfigurationVariables::GetInt("StartArmorType",2);

    /*********************************Init videocapture***************************************/
    capture_init(rmcap);

    /*********************************camera calibration***************************************/
    if (ConfigurationVariables::MainEntry == 10)//相机标定程序
    {
        char camparams[100] ;
        sprintf(camparams,FILEDIR(camparams_%d.xml),ConfigurationVariables::resolutionType);
        CameraView::CalibrateCameraProcess(camparams,rmcap); // write the parameters of the camera into the camparas.xml
        return 0;
    }

     thread proc_thread(ImageProcessThread);
     thread display_thread(ImageDisplayThread);
     thread collect_thread(ImageCollectThread);
     // thread write_thread(VideoWriteThread);

    collect_thread.join();
    proc_thread.join();
    display_thread.join();
    // write_thread.join();

    return 0;
}



void ProcessFullFunction(ImageData &frame)
{
    // 调试模块的模式下 强制打开模块
    switch(ConfigurationVariables::MainEntry)
    {
    case 0: serial_ptr->EnableModule(4);serial_ptr->EnableModule(2);break;  //默认打开自瞄模式
    case 1: serial_ptr->EnableModule(2);break;
    case 2: serial_ptr->EnableModule(4);break;
    }

    serial_ptr->UpdateCurModule(frame,dtTime);
    // for(int i=0;i<(serial_ptr->module_list.size());i++)
    // {
    //     cout<<serial_ptr->module_list[i]->moduleID<<" "<<serial_ptr->module_list[i]->is_enabled<<endl;
    // }
}

void ProcessAlgorithmFunction(ImageData &frame)
{
    switch(ConfigurationVariables::MainEntry){
        case 4:  // armor detect
            ArmorDetectDebug(frame);
            break;
        case 5: //dfc detect
            DfcDetectDebug(frame);
            break;
    }
}

void ArmorDetectDebug(ImageData &frame)
{
    res = armor_tracker_ptr->UpdateFrame(frame,dtTime)*0.3;
}

void DfcDetectDebug(ImageData &frame)
{
    dfc_hiter_ptr->Update(frame,dtTime);
}
