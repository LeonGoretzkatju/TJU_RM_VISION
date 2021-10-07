#pragma once
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
#include <iostream>
#include <opencv2/opencv.hpp>
#include "util.hpp"
#include "configurations.hpp"

using namespace std;
using namespace cv;

//dfc检测基类
class DfcBaseDetetor
{
    public:
    DfcBaseDetetor(){}   
    //Detect Dfc Armor 
    virtual void DetectDfcArmors(Mat frame) = 0;    //构造成虚函数，方便派生类对象继承

    void DisallowDebug()
    {
        allowDebug = false;
    }
    protected :
    bool allowDebug = true;
};

class EllipseDfcDetector:public DfcBaseDetetor
{
    public:
    EllipseDfcDetector():DfcBaseDetetor()
    {
        binary_thresh = 80; //二值化thresh
        aim_thresh_min = 1.7; //装甲长宽比例最小值
        aim_thresh_max = 2.6; //装甲长宽比例最大值
    }

    void SetTargetArmor(ArmorColorType tgtColor)
    {
        targetArmorColor = tgtColor;
    }

    void DetectDfcArmors(Mat frame)
    {
        frame.copyTo(img);  
        split(img, spl); 
        subtract(spl[targetArmorColor],spl[2 - targetArmorColor],gray);  //通道相减

        threshold(gray, binary, binary_thresh, 255, THRESH_BINARY);

        dilate(binary,binary,Mat());
        dilate(binary,binary,Mat());

        floodFill(binary,Point(0,0),Scalar(255),0,FLOODFILL_FIXED_RANGE); //漫水填充,分割出装甲板
        threshold(binary, binary, binary_thresh, 255, THRESH_BINARY_INV);

        findContours(binary, contours, RETR_LIST, CHAIN_APPROX_NONE);
        for (size_t i = 0; i < contours.size(); i++){
            vector<Point> points;
            double area = contourArea(contours[i]); //获取轮廓面积
            if (area < 50 || 1e4 < area) continue;  //滤除过小或者过大的轮廓
            drawContours(binary, contours, static_cast<int>(i), Scalar(0), 2);

            points = contours[i];  //获取一个轮廓的特征点
            RotatedRect rrect = fitEllipse(points); //用椭圆拟合

            rrect.points(vertices);                 //找出角点

            float aim = rrect.size.height/rrect.size.width;
            if (aim > aim_thresh_min && aim < aim_thresh_max)
            {
            for (int j = 0; j < 4; j++)
                {
                    cv::line(img, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 255, 0),4);
                }
            }
        }
        if(DEBUG_MODE)
        DEBUG_DISPLAY(binary);
    }


    protected:
    Mat img,gray,binary,spl[3];
    vector<vector<Point>> contours;
    Point2f vertices[4];

    float aim_thresh_min,aim_thresh_max;
    int binary_thresh; //二值化thresh

    ArmorColorType targetArmorColor = ARMOR_UNKNOWN; //敌方机器人装甲板颜色
};
