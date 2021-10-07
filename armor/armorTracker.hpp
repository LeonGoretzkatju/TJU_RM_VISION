/*
 * 这是 TJU Robomasters 上位机源码，未经管理层允许严禁传播给其他人（包括队内以及队外）
 *
 *   This file defines base class of armor tracker, and its implements. Tracker is 
 * designed to control ptz to follow the aiming target.
 *    2018.6.12 Architect the armortrackerBase
 */

#pragma once
#include <opencv2/opencv.hpp>
#include "util.hpp"
#include "configurations.hpp"
#include "camview.hpp"
#include "predictor.hpp"
#include "armorDetector.hpp"
#include "dnnManager.hpp"
#include "trackerMedianFlow.hpp"
#include "CoordinatesFusion.hpp"
#include <vector>
#include <Eigen/Dense>

using namespace cv;
using namespace std;


//装甲追踪基类
class ArmorTrackerBase
{
public:
    // Configuration variables
    int InfancyMaxTrackingFrame = 60;
    int HeroMaxTrackingFrame = 30;
    int EngineerMaxTrackingFrame = 10;
    double ArmorEvaluationDistanceWeight = 0.1;
    double DetectionROIScale = 2.5;
    double TrackerBoundRectScale = 1.5;
    int TrackerResetTime = 10;
    double TrackerPositionChangeThresh = 0.14;
    double TrackerSizeChangeThresh = 0.5;
    float fBigArmorWidth=225;
    float fBigArmorHeight=88.5;
    float fSmallArmorWidth=136.5;
    float fSmallArmorHeight=54.5;        
    ArmorResult last_box, target_box;

    // 云台到最佳射击角度的偏转角度，默认这个角度为到装甲的角度加上重力补偿角度的值
    Point2f shootOffAngle;
    // 云台到装甲的偏转角度，转过该角度云台正好对准装甲
    Point2f ptzOffAngle;
    // 追踪得到的装甲距离云台的距离
    float lastArmorDistance ;
    // 装甲的世界坐标
    Point3f targetWorldPosition;
    // 上次检测得到的装甲结果
    ArmorResult lastArmor;
    // track state: 0 lost target; 1 last detected ; 2+ last tracked
    int trackState = 0;
    // 跟踪的目标兵种
    ArmorDetailType trackTargetType;
    vector<float> armor_height_all;
    int tic = 0;

    double shoot_delay;


    ArmorTrackerBase(CameraView *_camview,ArmorBaseDetector *_detector,DnnManager *_dnn) //构造函数
    {
        camview = _camview;
        detector = _detector;
        dnnManager = _dnn;

        SET_CONFIG_INT_VARIABLE(InfancyMaxTrackingFrame,60)
        SET_CONFIG_INT_VARIABLE(HeroMaxTrackingFrame,30)
        SET_CONFIG_INT_VARIABLE(EngineerMaxTrackingFrame,10)
        SET_CONFIG_DOUBLE_VARIABLE(ArmorEvaluationDistanceWeight,0.1)
        SET_CONFIG_DOUBLE_VARIABLE(DetectionROIScale,2.5)
        SET_CONFIG_DOUBLE_VARIABLE(TrackerBoundRectScale,1.5)
        SET_CONFIG_INT_VARIABLE(TrackerResetTime,10)
        SET_CONFIG_DOUBLE_VARIABLE(TrackerPositionChangeThresh,0.14)
        SET_CONFIG_DOUBLE_VARIABLE(TrackerSizeChangeThresh,0.5)
    }

    // called every frame, and returns the deltaAngle for ptz control
    // the base class only consider shootOffAngle as the ptz control deltaAngle
    virtual Point2f UpdateFrame(ImageData frame,float deltaTime)
    {
        /*********************************进行全图检测***********************************************/
        if (!trackState || (trackState > InfancyMaxTrackingFrame && trackTargetType == ARMOR_INFAN) ||
                (trackState > HeroMaxTrackingFrame && trackTargetType == ARMOR_HERO) ||
                (trackState > EngineerMaxTrackingFrame && trackTargetType == ARMOR_ENGIN) ||
                (trackState > InfancyMaxTrackingFrame && trackTargetType == ARMOR_TYPE_UNKNOWN))
        {
            /*********************************装甲板检测***********************************************/
            detector->DetectArmors(frame.image);
            /*********************************检测到目标***********************************************/
            if (detector->result.size() > 0)
            {
                ArmorDetailType armRes[20]; 
                float confidence[20];       
                this->GetArmorTypes(armRes,confidence);
                ChooseTarget(frame,armRes,confidence);
                       
                this->InitTracker(frame);
                trackState = 1;
            }
            /*********************************未检测到目标，开启追踪**************************************/
            else if (trackerAlgorithmState) 
            {
                // still try tracking
                if (UpdateTracker(frame))
                    ApplyTracker(frame);
                else trackState = 0;
            }
            else
            /********************************未检测到目标，未开启追踪**************************************/
                trackState = 0;
        }
        /*************************************ROI区域检测***********************************************/
        else 
        {
            // 根据上一次计算出的世界坐标 给出在屏幕上的投影位置
            float dis;
            Point2f lptzAngle = ProjectWorldPositionToPTZAngle(frame,targetWorldPosition,dis);  
            Point2d scrPt = camview->PTZAngleToScreenPoint(lptzAngle * Deg2Rad,dis);    
            Rect r1 = lastArmor.leftLight.rr.boundingRect(), r2 = lastArmor.rightLight.rr.boundingRect();  
            int width = max(r1.x + r1.width, r2.x + r2.width) - min(r1.x, r2.x), height = max(r1.y + r1.height, r2.y + r2.height) - min(r1.y, r2.y);
            width *= DetectionROIScale ; height *= DetectionROIScale;
            Rect roiRect(scrPt.x - width * 0.5f,scrPt.y - height * 0.5f,width,height);  

            MakeRectSafe(frame.image,roiRect);  
            Point2f startPoint(roiRect.x,roiRect.y);
            Mat roiRegion = frame.image(roiRect).clone();
            /*********************************roi装甲板检测********************************************/
            detector->DetectArmors(roiRegion);
            /*********************************检测到目标***********************************************/
            if (detector->result.size() > 0)
            {
                ArmorDetailType armRes[20]; 
                float confidence[20];     
                trackState++;
                // transform the result in roi to frame image
                FOREACH(i,detector->result.size())
                {
                    detector->result[i].center += startPoint;
                    detector->result[i].leftLight.center += startPoint;
                    detector->result[i].leftLight.rr.center += startPoint;
                    detector->result[i].rightLight.rr.center += startPoint;
                    detector->result[i].rightLight.center += startPoint;
                }
                // ChooseTarget(frame);
                this->GetArmorTypes(armRes,confidence);
                ChooseTarget(frame,armRes,confidence);
                if (trackerAlgorithmState > TrackerResetTime || !UpdateTracker(frame))
                    InitTracker(frame);
            }
            /*********************************未检测到目标**********************************************/
            else
            {
                if (UpdateTracker(frame))
                {
                    // successfully tracked the armor
                    trackState += 2; // 追踪的结果往往不可靠
                    ApplyTracker(frame);
                }
                else // tracking failed, lost target
                    trackState = 0;
            }
            /***********************************Track Mode Debug***************************************/
            if (DEBUG_MODE && trackState)
            {
                rectangle(frame.image,roiRect,Scalar(255,255,0),2);
            }
        }
        
        /****************************如果检测成功，计算装甲的相机坐标**************************************/
        if (trackState)
        {
            Point2f worldAngle = (ptzOffAngle + frame.ptzAngle) * Deg2Rad;
            targetWorldPosition.y = sin(worldAngle.y) * lastArmorDistance ;
            float d = cos(worldAngle.y) * lastArmorDistance;
            targetWorldPosition.x = d * cos(worldAngle.x);
            targetWorldPosition.z = d * sin(worldAngle.x);
            targetWorldPosition.x += frame.worldPosition.x;
            targetWorldPosition.z += frame.worldPosition.y;
        }
        /*************************************Debug**********************************************/
        if (DEBUG_MODE)
        {
            if (trackState)
                circle(frame.image,lastArmor.center, 6 ,Scalar(255,0,255),2);
            if (trackerAlgorithmState)
                rectangle(frame.image,trackerBound,Scalar(0,255,0),1);
            DEBUG_DISPLAY(frame.image)
        }
        return shootOffAngle;
    }

    // choose an target from detector result 
    // default stratage is to find the nearest armor to hit
    virtual void ChooseTarget(ImageData &frame_data,ArmorDetailType types[],float confidence[])
    {
        Point2f bestPTZ,curPTZ;
        float bestDis,curDis,bestScore,curScore;
        int bestIndex = -1;

        FOREACH(i,detector->result.size())
        {
            solve_pnp(detector->result[i],frame_data,&curDis);  //pnp求距离
            curPTZ = CalcArmorPTZAngle(detector->result[i],curDis);
            curScore = EvaluateArmorPosition(curPTZ,curDis,types[i],confidence[i]);
            if (bestIndex == -1 || curScore > bestScore)
            {
                bestIndex = i;
                bestDis = curDis;
                bestScore = curScore;
                bestPTZ = curPTZ;
            }
        }
        // choose target 'bestIndex'
        lastArmor = detector->result[bestIndex];
        ptzOffAngle = bestPTZ;
        //cout << "before the iteration,the ptzangle is:" << " " << ptzOffAngle << endl;
        //shootOffAngle = ptzOffAngle + CalculateGravityAngle(frame_data.ptzAngle.y + ptzOffAngle.y,frame_data.shootSpeed,bestDis);
        //cout << "shoot speed" << frame_data.shootSpeed << endl;
        //cout << "gravity angle :  " << CalculateGravityAngle(frame_data.ptzAngle.y + ptzOffAngle.y,frame_data.shootSpeed,bestDis) << endl;
        lastArmorDistance = bestDis;
        this->trackTargetType = types[bestIndex];
    }

    void ChooseTarget(ImageData &frame_data)
    {
        ArmorDetailType t_types[20];
        float t_conf[20];
        FOREACH(i,detector->result.size()){
            t_types[i] = trackTargetType;
            t_conf[i] = 0.5f;
        }
        ChooseTarget(frame_data,t_types,t_conf);
    }

protected:
    CameraView *camview;
    ArmorBaseDetector *detector;
    DnnManager *dnnManager;
    Ptr<TrackerMedianFlow> tracker;
    int trackerAlgorithmState = 0; // 0 means uninitialized  ; otherwise means initialized and the time tracked
    Rect2d trackerBound;

    void InitTracker(ImageData &frame)
    {
        tracker = TrackerMedianFlow::create();
        trackerBound.width = abs(lastArmor.leftLight.center.x - lastArmor.rightLight.center.x) * TrackerBoundRectScale;
        trackerBound.height = (lastArmor.leftLight.rr.size.height + lastArmor.rightLight.rr.size.height) * TrackerBoundRectScale ;
        trackerBound.x = lastArmor.center.x - trackerBound.width / 2;
        trackerBound.y = lastArmor.center.y - trackerBound.height / 2;
        tracker->init(frame.image,trackerBound);
        trackerAlgorithmState = 1;
    }

    bool UpdateTracker(ImageData &frame)
    {
        Rect2d org = trackerBound;
        // check position sudden change or size sudden change
        if (
                (!tracker->update(frame.image,trackerBound)) || // algorithm tracking failed
                (Length(Point2d(org.x + org.width / 2 - trackerBound.x - trackerBound.width/2, // position suddenly change
                           org.y + org.height / 2 - trackerBound.y - trackerBound.height/2)) / (org.width + org.height) > TrackerPositionChangeThresh) ||
                (trackerBound.width * trackerBound.height / org.width / org.height < TrackerSizeChangeThresh)  // size suddenly change
           )
        {
            trackerAlgorithmState = 0;
            return false;
        }
        trackerAlgorithmState ++;
        return true;
    }

    void ApplyTracker(ImageData &frame)
    {
        Point2f delta (trackerBound.x +trackerBound.width /2 - lastArmor.center.x,
                       trackerBound.y + trackerBound.height/2 - lastArmor.center.y );
        lastArmor.center += delta;
        lastArmor.leftLight.center += delta;
        lastArmor.rightLight.center += delta;
        lastArmor.leftLight.rr.center += delta;
        lastArmor.rightLight.rr.center += delta;
        
    }

    Point2f ProjectWorldPositionToPTZAngle(ImageData &frameData,Point3f worldPosition,float &distance)
    {
        worldPosition.x -= frameData.worldPosition.x;
        worldPosition.z -= frameData.worldPosition.y;
        distance = Length(worldPosition);
        Point2f absoluteAngle;
        absoluteAngle.y = atan(worldPosition.y / sqrt(worldPosition.x * worldPosition.x + worldPosition.z * worldPosition.z));
        absoluteAngle.x = atan2(worldPosition.z ,worldPosition.x);
        return absoluteAngle * Rad2Deg - frameData.ptzAngle;
    }

    void GetArmorTypes(ArmorDetailType res[],float confidence[])
    {
        vector<Mat> imgs;
        FOREACH(i,detector->result.size())
        {
            // initialize res and confidence
            res[i] = ARMOR_TYPE_UNKNOWN; //armor_type 
            confidence[i] = 0;           //confidence
            Mat resized;                 //resize img to 28*28
            resize(detector->ArmorNumberAreaGray(detector->result[i]),resized,Size(28,28));
            imgs.push_back(resized);     //push to vector
        }
        dnnManager->ClassifyArmors(imgs,res,confidence);
        // if(DEBUG_MODE)
        // {
        //     FOREACH(i,imgs.size())
        //     cout<<"armor_id:"<<res[i]<<"    armor_confidence:"<<confidence[i]<<endl;
        // }
        FOREACH(i,imgs.size())
        {
            switch(res[i])
            {
            case 3:case 4:case 5: res[i] = ARMOR_INFAN; break;
            case 1:res[i] = ARMOR_HERO; break;
            case 2:res[i] = ARMOR_ENGIN; break;
            default: res[i] = ARMOR_TYPE_UNKNOWN; break;
            }
        }
    }

    // returns ptz off angle of an armor and calculates the probdis by the way;
    virtual Point2f CalcArmorPTZAngle(ArmorResult armor,float &probDis)
    {
        return camview->ScreenPointToPTZAngle(armor.center,probDis,1);
    }
    // evaluate armor worth shooting by considering ptz offset angle and estimated distance
    virtual float EvaluateArmorPosition(Point2f ptzAngle,float probDis,ArmorDetailType type,float confidence)
    {
        float typeScore = 0;
        switch(type)
        {
        case ARMOR_INFAN: typeScore = 250; break;
        case ARMOR_HERO: typeScore = 200; break;
        case ARMOR_ENGIN: typeScore = 50;break;
        }
        typeScore *= confidence;
        return typeScore -(probDis * ArmorEvaluationDistanceWeight + Length(ptzAngle));
    }

    void solve_pnp1(ArmorResult armor,ImageData &frame,float *curdistance)
    {
        
        static float last_distance;
        
        // last_distance = *curdistance;
        // cout << "last distance0 " << "  " << last_distance << endl;
        float HALF_LENGTH = 70;
        float HALF_HEIGHT = 30 ;
        float k = ConfigurationVariables::GetInt("linearfittingparameters_k",30);
        float b = ConfigurationVariables::GetInt("linearfittingparameters_b",30);
        vector<Point3f> obj=vector<Point3f>{
            cv::Point3f(-HALF_LENGTH, -HALF_HEIGHT, 0),	//tl
            cv::Point3f(HALF_LENGTH, -HALF_HEIGHT, 0),	//tr
            cv::Point3f(HALF_LENGTH, HALF_HEIGHT, 0),	//br
            cv::Point3f(-HALF_LENGTH, HALF_HEIGHT, 0)	//bl
            };
        vector<Point2f> pnts;
        Point2f point_tl,point_bl,point_tr,point_br;
        float armor_width,armor_height,armor_height_com;
        float armor_height_sum;
        //static float last_armor_height;
        armor_width = armor.rightLight.center.x -armor.leftLight.center.x + armor.leftLight.rr.size.width*0.5 + armor.rightLight.rr.size.width*0.5;
        armor_height = (armor.leftLight.rr.size.height + armor.rightLight.rr.size.height)*0.5;
        armor_height_all.push_back(armor_height);
        if (armor_height_all.size() >= 18)
        {
            armor_height_sum = 0;
            for(int i = 0; i < armor_height_all.size(); i++)
            {
                armor_height_sum += armor_height_all[i];
            }
            armor_height_com = armor_height_sum/armor_height_all.size();
            //cout << "armor_height_com" << armor_height_com << endl;
            *curdistance = k/armor_height_com + b;
            last_distance = *curdistance;
            armor_height_all.clear();
        }
        else if (tic >= 18)
        {
            *curdistance = last_distance;
        }
        else
        {
            *curdistance = k*armor_height + b;
        }
        tic++;
        if (tic >= 1000)
        {
            tic = 0;
        }
        if (DEBUG_MODE)
        {
            cout << "current distance " << "  " << (*curdistance) << endl;
            cout << "armor_height   " << armor_height << endl;
            cout << "last distance " << "  " << last_distance << endl;
        } 

    }

    void solve_pnp(ArmorResult armor,ImageData &frame,float *curdistance)
    {
        
        // last_distance = *curdistance;
        // cout << "last distance0 " << "  " << last_distance << endl;
        float HALF_LENGTH = 70;
        float HALF_HEIGHT = 30 ;
        //float k = ConfigurationVariables::GetInt("linearfittingparameters_k",30);
        //float b = ConfigurationVariables::GetInt("linearfittingparameters_b",30);
        vector<Point3f> obj=vector<Point3f>{
            cv::Point3f(-HALF_LENGTH, -HALF_HEIGHT, 0),	//tl
            cv::Point3f(HALF_LENGTH, -HALF_HEIGHT, 0),	//tr
            cv::Point3f(HALF_LENGTH, HALF_HEIGHT, 0),	//br
            cv::Point3f(-HALF_LENGTH, HALF_HEIGHT, 0)	//bl
            };
        vector<Point2f> pnts;
        Point2f point_tl,point_bl,point_tr,point_br;
        float armor_width,armor_height,armor_height_com;
        //float armor_height_sum;
        //static float last_armor_height;
        armor_width = armor.rightLight.center.x -armor.leftLight.center.x + armor.leftLight.rr.size.width*0.5 + armor.rightLight.rr.size.width*0.5;
        armor_height = (armor.leftLight.rr.size.height + armor.rightLight.rr.size.height)*0.5;
        
        point_tl = Point2f(armor.center.x - armor_width*0.5,armor.center.y - armor_height*0.5);
        point_bl = Point2f(armor.center.x - armor_width*0.5,armor.center.y + armor_height*0.5);
        point_tr = Point2f(armor.center.x + armor_width*0.5,armor.center.y - armor_height*0.5);
        point_br = Point2f(armor.center.x + armor_width*0.5,armor.center.y + armor_height*0.5);

        pnts.push_back(point_tl);
        pnts.push_back(point_tr);
        pnts.push_back(point_br);
        pnts.push_back(point_bl);

		Mat rVec,tVec;
        solvePnP(obj,pnts,camview->get_intrinsic_matrix(),camview->get_distortion_coeffs(),rVec,tVec,false,SOLVEPNP_ITERATIVE);
        *curdistance = tVec.at<double>(0,2);
		cout << "current distance " << "  " << (*curdistance) << endl;

    }

    void CountAngleXY(ArmorResult &armor,ImageData &frame)
    {
        float fHalfX=0;
        float fHalfY=0;
        vector<Point2f> pnts;
        vector<Point3f> point3D;
        Point2f point_tl,point_bl,point_tr,point_br;
        float armor_width, armor_height;
        armor_width = armor.rightLight.center.x - armor.leftLight.center.x + armor.leftLight.rr.size.width * 0.5 + armor.rightLight.rr.size.width * 0.5;
        armor_height = (armor.leftLight.rr.size.height + armor.rightLight.rr.size.height) * 0.5;
        point_tl = Point2f(armor.center.x - armor_width*0.5,armor.center.y - armor_height*0.5);
        point_bl = Point2f(armor.center.x - armor_width*0.5,armor.center.y + armor_height*0.5);
        point_tr = Point2f(armor.center.x + armor_width*0.5,armor.center.y - armor_height*0.5);
        point_br = Point2f(armor.center.x + armor_width*0.5,armor.center.y + armor_height*0.5);

        pnts.push_back(point_tl);
        pnts.push_back(point_tr);
        pnts.push_back(point_br);
        pnts.push_back(point_bl);
        if (armor.smallArmor)
        {
            cout << "small armor" << endl;
            fHalfX=fSmallArmorWidth/2.0;
            fHalfY=fSmallArmorHeight/2.0;            
        }
        else
        {
            cout << "big armor" << endl;
            fHalfX=fBigArmorWidth/2.0;
            fHalfY=fBigArmorHeight/2.0;
        }
        point3D.push_back(cv::Point3f(-fHalfX,fHalfY,0.0));
        point3D.push_back(cv::Point3f(fHalfX,fHalfY,0.0));
        point3D.push_back(cv::Point3f(fHalfX,-fHalfY,0.0));
        point3D.push_back(cv::Point3f(-fHalfX,-fHalfY,0.0));
        cv::Mat rvecs=cv::Mat::zeros(3,1,CV_64FC1);
        cv::Mat tvecs=cv::Mat::zeros(3,1,CV_64FC1);
        solvePnP(point3D,pnts,camview->get_intrinsic_matrix(),camview->get_distortion_coeffs(),rvecs,tvecs);
        double tx = tvecs.ptr<double>(0)[0];
        double ty = -tvecs.ptr<double>(0)[1];
        double tz = tvecs.ptr<double>(0)[2];        
        armor.tx = tx;
        armor.ty = ty;
        armor.tz = tz;
        armor.pitch = atan2( armor.ty, armor.tz)*180/PI;
        armor.yaw = atan2(armor.tx,armor.tz)*180/PI;
        
    }
    void ShootAdjust(float &tx, float &ty, float &tz,float Carpitch,float Caryaw)
    {
    Carpitch *=PI/180;
    Caryaw *=PI/180;

    //绕roll轴旋转，即为绕z轴旋转
    Eigen::MatrixXd r_Roll(3,3);
    r_Roll<<1 , 0 , 0,
                    0 , cos(0) , sin(0),
                    0 , -sin(0) , cos(0);
    //绕pitch轴旋转，即为绕x轴旋转
    Eigen::MatrixXd r_Pitch(3,3);
    r_Pitch<<cos(0) , 0 , -sin(0),
                        0 , 1 , 0,
                        sin(0) , 0 , cos(0);
    //绕yaw轴旋转，即为绕y轴旋转
    Eigen::MatrixXd r_Yaw(3,3);
    r_Yaw<<cos(0) , sin(0) , 0,
                     -sin(0) , cos(0) , 0 ,
                     0 , 0 , 1;

    Eigen::VectorXd original(3,1);          //按z，x，y传入，即变化对应到左手坐标系
    original<<tz,tx,ty;

    //平移变换,先旋转再平移
    Eigen::VectorXd translation(3,1);
    translation<<camview->CameraOffsetZ,camview->CameraOffsetX,camview->CameraOffsetY;
    original = original + translation;


    Eigen::VectorXd change(3,1);
    //旋转变换
    change =  r_Roll * original;
    change = r_Pitch*change;
    change = r_Yaw*change;
    //以上部分的偏移参数调节


    //去掉车云台旋转相对影响,坐标系转换到相对初始位的绝对坐标
    //pitch转换
    Eigen::MatrixXd r_pitch_car(3,3);
    r_pitch_car<<cos(Carpitch) , 0 , -sin(Carpitch),
                        0 , 1 , 0,
                        sin(Carpitch) , 0 , cos(Carpitch);
    //yaw转换
    Eigen::MatrixXd r_yaw_car(3,3);
    r_yaw_car<< cos(Caryaw), -sin(Caryaw), 0,
                sin(Caryaw), cos(Caryaw), 0, 
                0, 0, 1;
    change = r_pitch_car * change;
	change = r_yaw_car * change;

    tx = change(1);
    ty = change(2);
    tz = change(0);
    }
};


// class anti_top {
// private:
//     double center_x=0.0,center_y=0.0,radius=0.0;
//     vector<ArmorResult> armor_seq;

// public:
//     ArmorResult last_box,target_box;
//     anti_top()
//     {
//         cout<<"come in"<<endl;
//     }
 
//     bool is_top(){
//         const int LOST=5;
//         const int TOP=0;
//         const double detla_h=0.5;
//         const double detla_w=5;
//         int lost_frame_cnt=0;
//         int top=0;
//         int track=0;
//         //score==-1 not detected
//         if(target_box.score==-1){
//             lost_frame_cnt++;
//             target_box=last_box;
//             if(lost_frame_cnt>LOST){
//                 top=0;
//                 track=0;
//                 return false;
//             }
//         }
//         else{
//             armor_seq.push_back(target_box);
//             if(armor_seq.size()<4){
//                 return false;
//             }
//             else{
//                 if((target_box.center-last_box.center).y>detla_h){
//                     top=0;
//                     armor_seq.clear();
//                     return false;
//                 }
//                 if((target_box.center-last_box.center).x>detla_w){
//                     top++;
//                     if(top<TOP){
//                         return false;
//                     }   
//                 }
//             }
            
//         }
//         last_box=target_box;
//         if(armor_seq.size()>=4){
//             return true;
//         }
//     }

//     Point3f run(){
//         cout<<"RUN!"<<endl;
//         if(find_rotated_top(center_x,center_y,radius)){
//             Point3f world_position(center_x,center_y,0.0);
//             return world_position;
//         }
//     }

//     bool find_rotated_top(double& x,double& y,double& R){
//         vector<Point2f> points;
//         for(int i=0;i<armor_seq.size();i++){
//             points.push_back(armor_seq[i].center);
//         }
//         center_x = 0.0f;
//         center_y = 0.0f;
//         radius = 0.0f;
//         if (points.size() < 3)
//         {
//             return false;
//         }

//         double sum_x = 0.0f, sum_y = 0.0f;
//         double sum_x2 = 0.0f, sum_y2 = 0.0f;
//         double sum_x3 = 0.0f, sum_y3 = 0.0f;
//         double sum_xy = 0.0f, sum_x1y2 = 0.0f, sum_x2y1 = 0.0f;

//         int N = points.size();
//         for (int i = 0; i < N; i++)
//         {
//             double x = points[i].x;
//             double y = points[i].y;
//             double x2 = x * x;
//             double y2 = y * y;
//             sum_x += x;
//             sum_y += y;
//             sum_x2 += x2;
//             sum_y2 += y2;
//             sum_x3 += x2 * x;
//             sum_y3 += y2 * y;
//             sum_xy += x * y;
//             sum_x1y2 += x * y2;
//             sum_x2y1 += x2 * y;
//         }

//         double C, D, E, g, H;
//         double a, b, c;

//         C = N * sum_x2 - sum_x * sum_x;
//         D = N * sum_xy - sum_x * sum_y;
//         E = N * sum_x3 + N * sum_x1y2 - (sum_x2 + sum_y2) * sum_x;
//         g = N * sum_y2 - sum_y * sum_y;
//         H = N * sum_x2y1 + N * sum_y3 - (sum_x2 + sum_y2) * sum_y;
//         a = (H * D - E * g )/ (C * g - D * D);
//         b = (H * C - E * D) / (D * D - g * C);
//         c = -(a * sum_x + b * sum_y + sum_x2 + sum_y2) / N;

//         x = a / (-2);
//         y = b / (-2);
//         R = sqrt(a * a + b * b - 4 * c) / 2;
//         return true;
//     } 

//     RotatedRect get_armorRect(ArmorResult target_box){
//         RotatedRect result;
//         vector <Point> poly_l,poly_r;
//         poly_l=target_box.leftLight.poly;
//         poly_r=target_box.rightLight.poly;
//         vector<Point> sum=poly_l;
//         for(int i=0;i<poly_r.size();i++){
//             sum.push_back(poly_r[i]);
//         }
//         result=minAreaRect(sum);
//         return result;
//     }

// };

// class anti_topArmorTracker : public ArmorTrackerBase
// {
//     public:

//     anti_topArmorTracker(CameraView *_camview,ArmorBaseDetector *_detector,DnnManager *_dnn,anti_top *_prd)
//         : ArmorTrackerBase(_camview,_detector,_dnn)
//     {
//         predictor = _prd;
//         yawPID.SetLimitParams(0.1,5);
//         pitchPID.SetLimitParams(0.1,5);
//         // SET_CONFIG_DOUBLE_VARIABLE(PredictionTimeScale,1);
//         // SET_CONFIG_DOUBLE_VARIABLE(PredictionTimeBias,0);

//         SET_CONFIG_DOUBLE_VARIABLE(yawPID.kp,1);
//         SET_CONFIG_DOUBLE_VARIABLE(yawPID.ki,0);
//         SET_CONFIG_DOUBLE_VARIABLE(yawPID.kd,0);
//         SET_CONFIG_DOUBLE_VARIABLE(pitchPID.kp,1);
//         SET_CONFIG_DOUBLE_VARIABLE(pitchPID.ki,0);
//         SET_CONFIG_DOUBLE_VARIABLE(pitchPID.kd,0);
//     }
//     // 重写UpdateFrame函数，aim top
//     Point2f UpdateFrame(ImageData frame,float deltaTime)
//     {
//         // 调用基类的函数更新
//         ArmorTrackerBase::UpdateFrame(frame,deltaTime);
//         if (trackState)
//         {
//             predictor->target_box=lastArmor;
//             Point3f Aim;
//             if(predictor->is_top()){
//                 Aim=predictor->run();
//             }
//             Point2f ptz = camerview->VecToAngle(Aim);
//             shootOffAngle += ptz - ptzOffAngle;
//             ptzOffAngle = ptz;
//             // the target angle is shootoffangle
//             //cout<<"x:"<<shootOffAngle.x<<"y:"<<shootOffAngle.y<<endl;
//             return Point2f(
//                 yawPID.calc(shootOffAngle.x),
//                 pitchPID.calc(shootOffAngle.y)
//             );
//         }
//         else
//         {
//             return Point2f();
//         }
//     }
// protected:
//     anti_top *predictor;
//     CameraView* camerview;
//     PID yawPID,pitchPID;
// };


class PredictAngelArmorTracker : public ArmorTrackerBase
{

public:
    double PredictionTimeScale = 1;
    double PredictionTimeBias = 0.2;
    Point2f targetvalue;
    Point2f predictvalue;
    // Mat tvec17 = (Mat_<float>(3,1) << 20,0,30);// this step you need to measure the deviation between the gun and the camera

    PredictAngelArmorTracker(CameraView *_camview,ArmorBaseDetector *_detector,DnnManager *_dnn,KalmanPredictor *_prd)
        : ArmorTrackerBase(_camview,_detector,_dnn)
    {
        predictor = _prd;
        yawPID.SetLimitParams(200,5);
        pitchPID.SetLimitParams(22.9,5);
        SET_CONFIG_DOUBLE_VARIABLE(PredictionTimeScale,1);
        SET_CONFIG_DOUBLE_VARIABLE(PredictionTimeBias,0);

        SET_CONFIG_DOUBLE_VARIABLE(yawPID.kp,1);
        SET_CONFIG_DOUBLE_VARIABLE(yawPID.ki,0);
        SET_CONFIG_DOUBLE_VARIABLE(yawPID.kd,0);
        SET_CONFIG_DOUBLE_VARIABLE(pitchPID.kp,1);
        SET_CONFIG_DOUBLE_VARIABLE(pitchPID.ki,0);
        SET_CONFIG_DOUBLE_VARIABLE(pitchPID.kd,0);
    }
    // 重写UpdateFrame函数，增加预测器和KF控制
    Point2f UpdateFrame(ImageData frame,float deltaTime)
    {
        // 调用基类的函数更新
		cout << "deltaTime: "<<deltaTime << endl;	
        ArmorTrackerBase::UpdateFrame(frame,deltaTime);
        CountAngleXY(lastArmor,frame);
		cout<<"x: "<<lastArmor.tx<<"y: "<<lastArmor.ty<<"z: "<<lastArmor.tz<<endl;
        cout << "pitch: "<<lastArmor.pitch << "yaw: " << lastArmor.yaw << endl;
        //Point3f RelativePoisition = Point3f(lastArmor.tx,lastArmor.ty,lastArmor.tz);              //保留相对坐标
        ShootAdjust(lastArmor.tx,lastArmor.ty,lastArmor.tz,frame.ptzAngle.y,frame.ptzAngle.x);
		cout<<"ptzx" <<frame.ptzAngle.x<<"ptzy"<< frame.ptzAngle.y<<endl; 
        cout<<"x2: "<<lastArmor.tx<<"y2: "<<lastArmor.ty<<"z2: "<<lastArmor.tz<<endl;
        // RelativePoisition.y = lastArmor.ty;                       //ty坐标使用绝对坐标
        Point2f Absangle =  Point2f(atan2(lastArmor.tx,lastArmor.tz)*180/CV_PI,atan2(lastArmor.ty,lastArmor.tz)*180/CV_PI);
        cout <<"Absangle.x: "<< Absangle.x <<"Absangle.y: "<< Absangle.y << endl;
        predictvalue = predictor->predict(Absangle.x,Absangle.y,lastArmorDistance,deltaTime,frame.shootSpeed);
		//return Point2f(predictvalue.x, ptzOffAngle.y);
		cout <<"before g"<< predictvalue.y<<endl;
		cout << "dis: "<<lastArmorDistance<<endl;
		predictvalue = predictvalue + CalculateGravityAngle(predictvalue.y, frame.shootSpeed, lastArmorDistance);
		cout << "after g" << predictvalue.y << endl;
		cout <<"predictvalue.x: "<<predictvalue.x << " predictvalue.y: "<<predictvalue.y<<endl;
		return predictvalue;
    }
protected:
    // CoordinatesFusion *fusion;
    KalmanPredictor *predictor;
    PID yawPID,pitchPID;
};



