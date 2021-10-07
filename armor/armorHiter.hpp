/*
 * 这是 TJU Robomasters 上位机源码，未经管理层允许严禁传播给其他人（包括队内以及队外）
 *
 * armorHiter类控制机器人进行打击操作
 */


#pragma once

#include "armorTracker.hpp"
#include "serial.hpp"
#include "util.hpp"
#include "configurations.hpp"

using namespace cv;
using namespace std;

// 模块ID为2
class ArmorHiter : public ModuleBase{

public:
    ArmorHiter(SerialManager *_serial,ArmorTrackerBase *_armor_tracker):ModuleBase(2)
    {
        serial = _serial;
        armor_tracker = _armor_tracker;
    }
protected:
    ArmorTrackerBase *armor_tracker;
    SerialManager *serial;
    
};

// 适用于哨兵的打击程序
class SentinelArmorHiter : public ArmorHiter
{
private:
    double ShootAngleThresh = 500;
    //xiaoyugaijiaodusheji
    double trackingSpeed;// -> (+), <- (-)
    bool foundTarget = false;

    // correction because v_x 
    Point2f corection(Point2f result,float* curdistance){
        trackingSpeed = movingSpeed;
        double yaw=result.x;
        //cout<<"yaw: "<<yaw<<endl;
        float distance = *(curdistance);
        double detla_x=distance*sin(yaw);
        double detla_y=distance*cos(yaw);
        Point2f result2;
        result2.y=0;
        result2.x=( 2*trackingSpeed*detla_x-1+sqrt(((2*trackingSpeed*detla_x-1)*(2*trackingSpeed*detla_x-1)-4*trackingSpeed*trackingSpeed*(detla_x*detla_x+detla_y*detla_y-400))) )
                    /(2*trackingSpeed);
        //cout<<"result: "<<result2<<endl;
        return result2;
    }

public:

    SentinelArmorHiter(SerialManager *_serial,ArmorTrackerBase *_armor_tracker)
    : ArmorHiter(_serial,_armor_tracker){
        serial->SendFiringOrder(false, true);
    }

    void Update(ImageData& frame, float dtTime){
        Point2f result = (armor_tracker->UpdateFrame(frame,dtTime));
        // serial->SendFiringOrder(false, false);
        // result += corection(result, &armor_tracker->lastArmorDistance);
        foundTarget = armor_tracker->trackState;
        serial->SendContralMode(foundTarget);
        if (foundTarget){
            serial->SendFiringOrder(Length(armor_tracker->shootOffAngle) < ShootAngleThresh ,true);
            serial->SendPTZAbsoluteAngle(result.x, result.y);
			cout<<"result.x:"<<(result.x)<<" "<<"result.y :"<<(result.y)<<endl;
        }
     
    }

};


//适用于步兵的打击程序
class InfancyArmorHiter : public ArmorHiter
{
public:

    double ShootAngleThresh = 1;
    
    InfancyArmorHiter(SerialManager *_serial,ArmorTrackerBase *_armor_tracker) : ArmorHiter(_serial,_armor_tracker)
    {
        SET_CONFIG_DOUBLE_VARIABLE(ShootAngleThresh,1)
    }


    void Update(ImageData &frame,float dtTime)
    {
        Point2f result = armor_tracker->UpdateFrame(frame,dtTime);
        if (armor_tracker->trackState) // found target
        {
            // 发送消息
            serial->SendFiringOrder(Length(armor_tracker->shootOffAngle) < ShootAngleThresh ,true);
            serial->SendPTZAbsoluteAngle(result.x + frame.ptzAngle.x,result.y + frame.ptzAngle.y);
        }
    }

protected:
    
};


//适用于英雄的打击程序
class HeroArmorHiter : public ArmorHiter
{
public:

    double ShootAngleThresh = 1;
    
    HeroArmorHiter(SerialManager *_serial,ArmorTrackerBase *_armor_tracker) : ArmorHiter(_serial,_armor_tracker)
    {
        SET_CONFIG_DOUBLE_VARIABLE(ShootAngleThresh,1)
    }

    void Update(ImageData &frame,float dtTime)
    {
        //cout << "update frame" << endl;
        Point2f result = (armor_tracker->UpdateFrame(frame,dtTime))*0.3;
        if (armor_tracker->trackState) // found target
        {
            // 发送消息
            // serial->SendFiringOrder(Length(armor_tracker->shootOffAngle) < ShootAngleThresh ,true);
            serial->SendPTZAbsoluteAngle(result.x+0.00126 + frame.ptzAngle.x,result.y+ 0.0101 + frame.ptzAngle.y);
            
            if(DEBUG_MODE)
            {
                cout<<"result.x:"<<result.x+0.00126<<" "<<"result.y:"<<result.y<<endl;
            }
        }
    }

protected:
    
};

