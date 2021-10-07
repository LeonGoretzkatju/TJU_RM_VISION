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
#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "util.hpp"
#include<algorithm>
using namespace std;
using namespace cv;

/*
 * 深度学习对装甲兵种分类
 * 输入图像为中间图样的二值化图像，输入大小28*28
 * 输出为下面几种类别之一
 */
typedef int ArmorDetailType;
#define ARMOR_TYPE_UNKNOWN 0   // 不能确定装甲类型
#define ARMOR_HERO  1     // 装甲为英雄
#define ARMOR_ENGIN 2     // 装甲为工程
#define ARMOR_INFAN 3     // 装甲为步兵

class DnnManager {
public:
    /*
     * 构造函数，参数为模型的路径
     * armor_cfg 装甲类型识别的模型框架路径
     * armor_model 装甲类型识别的模型路径
     * 路径最好通过头文件util.hpp中的FILEDIR(fileName)得到
     */
    DnnManager( char* armor_cfg, char* armor_weights)
    {
        DnnClassifyThresh = SET_CONFIG_DOUBLE_VARIABLE(DnnClassifyThresh,0.5);
        armor_net = cv::dnn::readNetFromDarknet(String(armor_cfg),String(armor_weights));
    }

    // 多装甲同时分类
    void ClassifyArmors(vector<Mat> images, ArmorDetailType result[], float confidence[])
    {
        cv::Mat blob;
        int idx_max[2], idx_min[2];
        double minv, maxv;
        for(int i=0;i<images.size();i++)
        {
            blob = cv::dnn::blobFromImage(images[i],1.0/255.0,{28,28},0.0,true);
            armor_net.setInput(blob);
            Mat prob = armor_net.forward();
            minMaxIdx(prob,&minv,&maxv,idx_min,idx_max);
            if (maxv>DnnClassifyThresh) //belif大于阈值才进行分类，阈值可调
            {
                    result[i] = idx_max[1]; //保存识别出来的数字
                    confidence[i] = maxv;   //保存最大置信度
            }
            else
            {
                result[i] = ARMOR_TYPE_UNKNOWN;
                confidence[i] = 0.0;
            }
        }
    }

protected:
    cv::dnn::Net armor_net;
    double DnnClassifyThresh;
};
