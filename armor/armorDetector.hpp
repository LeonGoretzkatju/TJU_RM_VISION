/*
 * 这是 TJU Robomasters 上位机源码，未经管理层允许严禁传播给其他人（包括队内以及队外）
 *
 * This file defined the base class for armor detector, and the implemention of
 * 1.PolyMatchArmorDetector for rolling shutter cameras (LogiTech C270,KS2A17)
 * 
 * The features and modifications are as follows:
 *    2018.6.7  Transplant old armor detector to PolyMatchArmorDetector.
 *    2018.6.10 Replaced the imshow functions
 *    2018.6.11 Improved poly match algorithm to give a stable output
 *    2018.6.11 Add pair score condition base on hist graph for armor center. this improve accuracy if lights' enough.
 *    2018.6.11 Add tracking algorithm in case that long distance target missed
 *    2018.6.12 Optimize the efficiency of algorithm.Add paralle calculation in major operations.
 *    2018.6.12 Add method to distinguish whether the armor is small or big
 *    2018.6.13 优化了并行计算，使任务分配更均匀
 */
#pragma once

#include <cmath>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "util.hpp"
#include "configurations.hpp"
#include <cmath>
using namespace std;
using namespace cv;

/*
 * Light detection result data structure
 * stores light shape, light center point, and bounding rotated rect
 * 灯条信息结构体，保存灯条的中心点等信息
 */
struct ArmorLightExtraInfo
{
    int tag;    
    vector<Point> poly;
    Point2f center;
    float len, colorFeature;
    RotatedRect rr;

    ArmorLightExtraInfo(vector<Point> _poly, float _colorFeature, int _tag) //构造函数
    {
        poly = _poly;
        colorFeature = _colorFeature;
        len = arcLength(poly, true);    //多边形轮廓的周长
        tag = _tag;
        rr = minAreaRect(_poly);    //计算最小外接矩形
       if (abs(rr.angle) > 45)
        {
            rr.angle += 90;
            swap(rr.size.width,rr.size.height);
        }
    }
    ArmorLightExtraInfo()   //缺省构造函数
    {

    }
    void ProcessPoly()
    {
        Point2f total;
        float weight = 0,w;
        for (int i = 0; i < poly.size(); i++)
        {
            w = Length(poly[i],poly[(i+1)%poly.size()]) + Length(poly[i],poly[(i+poly.size()-1)%poly.size()]);
            total += Point2f(poly[i].x * w , poly[i].y * w);
            weight += w;
        }
        center = total * (1.0f / weight);
        Point centeri = Point((int)center.x,(int)center.y);
        for (int i = 0; i < poly.size(); i++)
            poly[i] -= centeri;
    }
};

/*
 * Armor matching result data structure
 * stores armor's left\right armor light infomation , armor center , and match score
 * 装甲板识别结果，保存左右灯条信息，中心点等信息
 */
struct ArmorResult
{
    ArmorLightExtraInfo leftLight, rightLight;
    Point2f center;
    float score;
    bool smallArmor;
    float pitch = 0;
    float yaw = 0;
    float tx = 0;
    float ty = 0;
    float tz = 0;
    bool IsTuoluo = false;          //是否为陀螺状态
    double get_waith = 0;               //对应陀螺宽度
    int status = 0;
    ArmorResult(ArmorLightExtraInfo l1, ArmorLightExtraInfo l2, float _score,bool _small_armor)
    {
        if (l1.center.x < l2.center.x)
        {
            leftLight = l1;
            rightLight = l2;
        }
        else
        {
            leftLight = l2;
            rightLight = l1;
        }
        center = (l1.center + l2.center) * 0.5f;
        score = _score;
        smallArmor = _small_armor;
    }
    ArmorResult()
    {
        score = -1;
    }
};

/*
 * The base class for armor detectors.
 * This is a virtual class and unusable without implemention 只有在实现后才可以使用
 * 基类装甲检测器
 */
class ArmorBaseDetector
{
public:
    ArmorBaseDetector(){}
    // This function Update armor detection result
    virtual void DetectArmors(Mat frame) = 0;
    // This function sets the detecting color of armors
    virtual void SetTargetArmor(ArmorColorType tgtColor) = 0;
    // This function track a last appear armor as detection algorithm sometimes fail
    virtual bool TrackArmor(ArmorResult &lastArmor) = 0;
    // Get center area image of an armor
    virtual Mat ArmorCenterAreaGrey(ArmorResult arRes)=0;
    // Get armor number image
    virtual Mat ArmorNumberAreaGray(ArmorResult arRes) = 0;
    // 'result' stores final results of good armors
    // 'combination' stores every light pair result
    vector<ArmorResult> result, combination;

    void DisallowDebug()
    {
        allowDebug = false;
    }
protected :
    bool allowDebug = true;
};


/*
 *   PolyMatchArmorDetector is a detector based on matching polys
 *   Poly matching suits rolling shutter cameras by adding some probabilities
 * when the target moving fast or gimbal self moving fast. To stablize the
 * detection result , add tracking algorithm to
 */
class PolyMatchArmorDetector : public ArmorBaseDetector
{
public :
    double TeamColorDifferThresh = 60;
    double TeamColorCheckDistance = 2;
    double PairScoreThresh = 0.5;
    double LightAspectRatioThresh = 2.0;
    double TeamColorSelectionRate = 0.8;
    double PolyMatchScoreRate = 10;
    double LightPairAngleScoreRate = 0.3;
    double LightPairShapeScoreRate = 1.5;
    double HistMaxScore = 0.2;
    int LightTopNumber = 6;
    int ContourSizeFiler = 6;
    int LightWhiteThresh = 220;
    int HistMinThresh = 40;
    int HistMaxThresh = 150;
    int TrackingWindowMargin = 4;
    int TeamColorAbsoluteThresh = 200;

    // This variable sets the armor color which algorithm detects
    ArmorColorType targetArmorColor = ARMOR_UNKNOWN;

    //比较装甲板评分
    static bool CompareResult(const ArmorResult &a, const ArmorResult &b)
    {
        return a.score > b.score;
    }
    //比较灯条评分
    static bool CompareLight(const ArmorLightExtraInfo &a, const ArmorLightExtraInfo &b)
    {
        return a.colorFeature > b.colorFeature;
    }

    //缺省构造函数
    PolyMatchArmorDetector():ArmorBaseDetector()    
    {
        //debugMode = ConfigurationVariables::MainEntry == 1;
        //TeamColorDifferThresh = ConfigurationVariables::GetDouble("TeamColorDifferThresh", 60);
        //TeamColorCheckDistance = ConfigurationVariables::GetDouble("TeamColorCheckDistance", 2);
        //PairScoreThresh = ConfigurationVariables::GetDouble("PairScoreThresh", 0.5);
	//LightTopNumber = ConfigurationVariables::GetInt("LightTopNumber", 6);
        //ContourSizeFiler = ConfigurationVariables::GetInt("ContourSizeFiler",5);

        SET_CONFIG_DOUBLE_VARIABLE(TeamColorDifferThresh,60);
        SET_CONFIG_DOUBLE_VARIABLE(TeamColorCheckDistance,2);
        SET_CONFIG_DOUBLE_VARIABLE(PairScoreThresh,0.5);
        SET_CONFIG_INT_VARIABLE(LightTopNumber,6);
        SET_CONFIG_INT_VARIABLE(ContourSizeFiler,5);
        SET_CONFIG_DOUBLE_VARIABLE(LightAspectRatioThresh,2.0);
        SET_CONFIG_DOUBLE_VARIABLE(TeamColorSelectionRate,0.8);
        SET_CONFIG_DOUBLE_VARIABLE(PolyMatchScoreRate,10);
        SET_CONFIG_DOUBLE_VARIABLE(LightPairAngleScoreRate,0.3);
        SET_CONFIG_DOUBLE_VARIABLE(LightPairShapeScoreRate,1.5);
        SET_CONFIG_INT_VARIABLE(LightWhiteThresh,220);
        SET_CONFIG_INT_VARIABLE(HistMinThresh,40);
        SET_CONFIG_INT_VARIABLE(HistMaxThresh,150);
        SET_CONFIG_DOUBLE_VARIABLE(HistMaxScore,0.2);
        SET_CONFIG_INT_VARIABLE(TrackingWindowMargin,4);
        SET_CONFIG_INT_VARIABLE(TeamColorAbsoluteThresh,200);
    }
    //设置敌方机器人颜色
    void SetTargetArmor(ArmorColorType tgtColor)
    {
        targetArmorColor = tgtColor;
    }
    //装甲板检测函数
    void DetectArmors(Mat org)
    {
        frame = org;
        GetTeamColorGraph(org);
        findContours(binary, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
//LOG(A)
        light_poses.clear();
        #pragma omp parallel for
        for (int i = 0; i < contours.size(); i++)
        {
            vector<Point> ct = contours[i];
            //滤除过小的轮廓
            if (ct.size() < ContourSizeFiler) continue;  
            // check lighting
            //根据轮廓点数组，求出轮廓点的最小外接矩形
            ArmorLightExtraInfo al(ct, 1, light_poses.size());  //把一个轮廓看成一个灯条，用矩形拟合
            if (al.rr.size.width / al.rr.size.height > LightAspectRatioThresh) continue;    //删除长宽比不合适的矩形,灯条的宽度大概应该是长度的0.2左右
            if (al.len > 800 || al.len < 20) continue; //删除周长过长或者过短的矩形
            if (al.rr.size.height < 22 &&  al.rr.size.width > 22) {continue;} //删除高度过低的矩形和宽度过宽的矩形
            // get light area
            Rect bound = al.rr.boundingRect();  //求出灯条的竖直外接矩形
            bound.x -= TeamColorCheckDistance;
            bound.y -= TeamColorCheckDistance;
            bound.width +=2 * TeamColorCheckDistance;
            bound.height += 2 * TeamColorCheckDistance; //稍微扩大一点矩形面积
            al.colorFeature = CountTeamColor(bound); //提取roi区域内颜色分量

            float score = al.colorFeature / al.len;
            if (score < TeamColorSelectionRate) continue;
                al.ProcessPoly();
            light_poses.push_back(al);  //将所有符合上述条件的灯条push到vector里面

            if(DEBUG_MODE)
            {
                //画出所有灯条
                Point2f vertices[4];
                al.rr.points(vertices);
                for(int i=0;i<4;i++)
                {
                    line(org, vertices[i], vertices[(i+1)%4], Scalar(255,0,255));
                }
                Point p(480,360);
                circle(org,p,5,Scalar(0,255,0),-1);
            }
 
        }
        sort(light_poses.begin(), light_poses.end(), CompareLight); //按照灯条颜色特征进行排序
//LOG(B)
        int actualSize = min((int)light_poses.size(), LightTopNumber);  //每一张图最大的灯条数量
        // make pairs
        combination.clear();
//LOG(C)
        // parallel calculation 先配好对，后期并行计算任务分配更均匀
        FORPAIR(i,j,actualSize) combination.push_back(ArmorResult(light_poses[i], light_poses[j],0,false)); //将每两个灯条配好对，生成装甲
        //并行处理
        #pragma omp parallel for    
        FOREACH(i,combination.size())
                combination[i].score = CheckPair(combination[i].leftLight,combination[i].rightLight,combination[i].smallArmor); //计算装甲板匹配分数
//LOG(D)
        sort(combination.begin(), combination.end(), CompareResult);    //按照匹配得分进行排序
        // pick pairs
        result.clear(); //保存配对成功的装甲板
        bool *tags = new bool[light_poses.size()]; //每一个灯条标志位，是否是装甲板灯条
        FOREACH(i, light_poses.size())  tags[i] = false;
        FOREACH(i, combination.size())
        {
            if (combination[i].score < PairScoreThresh) break;      //当前score低于阈值的直接break，后续不需要再匹配了
            if (tags[combination[i].leftLight.tag] || tags[combination[i].rightLight.tag])  //如果灯条组合的左右灯条中有一个已经匹配到了装甲板。跳过此次匹配
                continue;
            result.push_back(combination[i]);   //高于阈值的armor放进vector里
            tags[combination[i].leftLight.tag] = tags[combination[i].rightLight.tag] = true;
        }
        // if(DEBUG_MODE)
        // {
        //     FOREACH(i,result.size())
        //     {
        //         cout<<"result"<<i<<":"<<result[i].score<<endl;
        //     }
        //     FOREACH(i,combination.size())
        //     {
        //         cout<<"combination"<<i<<":"<<combination[i].score<<endl;
        //     }

        // }
        delete[] tags;

    }

    float CheckPair(ArmorLightExtraInfo lt1, ArmorLightExtraInfo lt2 ,bool &smallArmor)
    {
//LOG(a)
        //cout << "Pair " << lt1.tag << " " << lt2.tag << " : ";
        Point dt = lt1.center - lt2.center;
        float len = Length(dt, Point());    //计算两个灯条中心的L2距离
        if (len * 0.8 > lt1.len + lt2.len)  //舍弃两个灯条之间距离太长的匹配
            return -1;
        if (len - (lt1.rr.size.width + lt2.rr.size.width) / 2 < (lt1.rr.size.width+lt2.rr.size.width) * 0.3)    //舍弃两个灯条特别近的匹配
            return -1;
        // abundon if veritcal
        if (abs(dt.y/(dt.x+0.001)) > 0.8) return -1;    //删除水平平行的灯条
//LOG(b)
        float ang1, ang2, score = -PolyMatchScoreRate * JudgeLights(lt1,lt2);   //计算灯条之间的距离
        //cout << score << " ";
        ang2 = (float)atan2(dt.y, dt.x) * Rad2Deg;
        ang1 = CircularTo(lt1.rr.angle - ang2, -45, 45) / 45;  
        // cout<<lt1.rr.angle<<" "<<lt2.rr.angle<<endl;

        if (abs(ang1) < 0.7)
            score += LightPairAngleScoreRate * (1 - ang1*ang1); //左边灯条和装甲板匹配分数
        //cout << score << " ";
        ang1 = CircularTo(lt2.rr.angle - ang2, -45, 45) / 45;
        if (abs(ang1) < 0.7)
            score += LightPairAngleScoreRate * (1 - ang1*ang1); //右边灯条和装甲板匹配分数
        //cout << score << " ";
//LOG(c)
        ang1 = (lt1.rr.size.height + lt2.rr.size.height) / len; 
        float scr1 = ShapeRateScore(ang1),scr2 = ShapeRateScore(3-ang1) ;
        if (scr1 > scr2 )
        {
            score += scr1;
            smallArmor = true;
        }
        else
        {
            smallArmor = false;
            score += scr2;
        }
        // score armor center area
        // current score is impossible to be selected
        if (score < PairScoreThresh - HistMaxScore)
        {
        //    cout << endl;
            return score;
        }
//LOG(d)
        float histScore = GetHistRangePercentage(ArmorCenterAreaGrey(ArmorResult(lt1,lt2,0,false)),HistMinThresh,HistMaxThresh);
        if (histScore > 0.7) score += HistMaxScore;
        else if (histScore > 0.2) score += HistMaxScore * (histScore - 0.2) * 2;
        //cout << score << endl;

        return score;
    }

    float ShapeRateScore(float rate)
    {
        return rate > 0 ? LightPairShapeScoreRate * rate / exp(rate) : 0;
    }

    void GetTeamColorGraph(Mat &src)
    {
        split(src, spl);
        subtract(spl[targetArmorColor],spl[2 - targetArmorColor],teamColor);//red channel - blue channel to get the red information
        threshold(teamColor, teamColor, TeamColorDifferThresh, 255, THRESH_BINARY);
        cvtColor(src, grey, CV_BGR2GRAY);
        threshold(grey, binary, LightWhiteThresh, 255, THRESH_BINARY);
        Mat kernel = getStructuringElement(MORPH_RECT,Size(5,5),Point(-1,-1));
        morphologyEx(binary,binary,CV_MOP_CLOSE,kernel);    //进行闭操作
        if(DEBUG_MODE)
        {
            // DEBUG_DISPLAY(binary);
        }
#if defined ROBOT_INFANCY
        teamColor = teamColor & (spl[targetArmorColor] > TeamColorAbsoluteThresh);//this step is to better distinguish the enemy armor area
#elif defined ROBOT_HERO
        teamColor = teamColor & (spl[targetArmorColor] > TeamColorAbsoluteThresh);//this step is to better distinguish the enemy armor area
#endif
    }

    int CountTeamColor(Rect r)
    {
        MakeRectSafe(frame,r);  //把rect限制在图片内部
        int count = 0;
        FOREACH(i, r.height)
        {
            uchar* p = teamColor.ptr<uchar>(r.y + i) + r.x; //指向矩形每行第一个点的指针
            FOREACH(j, r.width)
                count += p[j];  //所有矩形里面像素相加
        }
        return count / 255;
    }
/*
    int CountTeamColor(Rect r)
    {
        MakeRectSafe(frame,r闭
        Mat hist;
        GetHist(teamColor(r),hist);
        return (int)hist.at<float>(255,0);
    }
*/
    bool TrackArmor(ArmorResult &trackArmor)
    {
        // 过于不可靠
        return false;
        Point2f dt = trackArmor.rightLight.center - trackArmor.leftLight.center;
        int width = (int)abs(dt.x), height = (int)abs(max(trackArmor.leftLight.rr.size.height,trackArmor.rightLight.rr.size.height));
        Rect roiRect((int)trackArmor.center.x - width,(int)trackArmor.center.y - height,2*width,2*height);
        MakeRectSafe(grey,roiRect);
        Mat binaryROI = binary(roiRect),teamColorROI = teamColor(roiRect);
        // make dilated operation
        Mat element = getStructuringElement(MORPH_ELLIPSE, Size(5, 5), Point(2, 2));
        dilate(binaryROI, dilated, element);
        teamColor = dilated & teamColorROI;
        // meanshift algorithm to track lights
        TermCriteria tc(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 1);
        Rect left_window = trackArmor.leftLight.rr.boundingRect(), right_window= trackArmor.rightLight.rr.boundingRect();
        left_window.x -= roiRect.x + TrackingWindowMargin; left_window.y -= roiRect.y + TrackingWindowMargin;
        right_window.x -= roiRect.x + TrackingWindowMargin; right_window.y -= roiRect.y + TrackingWindowMargin;
        left_window.width = abs(left_window.width); right_window.width = abs(right_window.width);
        left_window.width += 2 * TrackingWindowMargin ; right_window.width += 2 * TrackingWindowMargin;
        left_window.height += 2 * TrackingWindowMargin; right_window.height += 2 * TrackingWindowMargin;
        int widthMark1 = left_window.width ,widthMark2 = right_window.width,
                heightMark1 = left_window.height,heightMark2 = right_window.height;
        MakeRectSafe(dilated,left_window);
        MakeRectSafe(dilated,right_window);
        CamShift(dilated, left_window, tc);
        CamShift(dilated, right_window, tc);
        if (left_window.width == widthMark1 && left_window.height == heightMark1) return false;
        if (right_window.width == widthMark2 && right_window.height == heightMark2) return false;

        Point2f startPoint((float)roiRect.x ,(float)roiRect.y);
        trackArmor.leftLight.rr.center = GetRectCenterPoint(left_window) + startPoint;
        trackArmor.rightLight.rr.center = GetRectCenterPoint(right_window) + startPoint;
        trackArmor.center = (trackArmor.leftLight.rr.center + trackArmor.rightLight.rr.center) * 0.5f;

        if (DEBUG_MODE && allowDebug)
        {
            Mat orgROI = frame(roiRect);
            rectangle(orgROI,left_window,Scalar(0,0,255),2);
            rectangle(orgROI,right_window,Scalar(0,0,255),2);
        }
        // 用配对打分的机制检验灯柱跟踪的结果
        bool t;
        return CheckPair(trackArmor.leftLight,trackArmor.rightLight,t) > PairScoreThresh;
    }

    Mat ArmorCenterAreaGrey(ArmorResult arRes)
    {
        int size = int(arRes.leftLight.rr.size.height + arRes.rightLight.rr.size.height);
        Rect region((int)arRes.center.x - size / 2,(int)arRes.center.y - size / 2, size,size);
        MakeRectSafe(grey,region);
        return grey(region);
    }

    Mat ArmorNumberAreaGray(ArmorResult arRes)
    {
        int size = int((arRes.leftLight.rr.size.height + arRes.rightLight.rr.size.height));
        Rect region((int)arRes.center.x - size / 2,(int)arRes.center.y - size / 2, size,size);
        MakeRectSafe(grey,region);
        return grey(region);
    }

protected :
    Mat frame,grey, binary ,dilated , teamColor, spl[3];
    vector<vector<Point>> contours;
    vector<ArmorLightExtraInfo> light_poses;
    //圆形角度映射到min-max之间
    static float CircularTo(float val, float min, float max)
    {
        float dt = max - min;
        while (val < min) val += dt;
        while (val > max) val -= dt;
        return val;
    }

	// return 0 for bad 1 for teamcolor 2 for white
    int CheckTeamColor(Mat &org, Point pos, int &dtcol)
    {
        if (pos.x >= 0 && pos.x < org.cols && pos.y >= 0 && pos.y <= org.rows)
        {
                Vec3b color = org.at<Vec3b>(pos);
                if ((dtcol = (targetArmorColor == ARMOR_RED ? 1 : -1) * (color[2] - color[0])) > TeamColorDifferThresh)
                    return 1;
                return color[targetArmorColor] > 230 && color[2 - targetArmorColor] > 200 && color[1] > 200 ? 2 : 0;
        }
        return 0;
    }

    static float PointDistanceToPoly(Point p, vector<Point> poly)
    {
        float mindis = 999999, t;
        for (int i = 0; i < poly.size(); i++)
        {
                t = PointDistanceToLine(p, poly[i], poly[(i + 1) % poly.size()]);   //计算点到多边形直线的距离
                mindis = min(t, mindis);    //限幅
        }
        return mindis;
    }

    static float PolyDistanceToPoly(vector<Point> poly1, vector<Point> poly2)
    {
        double sum = 0, sumlen = 0;
        for (int i = 0; i < poly1.size(); i++)
        {
            double weight = Length(poly1[i], poly1[(i + 1) % poly1.size()]) + Length(poly1[i], poly1[(i + poly1.size() - 1) % poly1.size()]);   //按边长进行加权
            sum += PointDistanceToPoly(poly1[i], poly2) * weight;
            sumlen += weight;
        }
        return sum / sumlen;
    }

    static double JudgeLights(ArmorLightExtraInfo l1, ArmorLightExtraInfo l2)
    {
        return (PolyDistanceToPoly(l1.poly, l2.poly) + PolyDistanceToPoly(l2.poly, l1.poly)) / max(l1.len ,l2.len); //计算灯条之间的距离
    }

    static float PointDistanceToLine(Point p, Point l1, Point l2)
    {
        if ((p - l1).dot(l2 - l1) >= 0 && (p - l2).dot(l1 - l2) >= 0)
                return abs((p - l1).cross(l2 - l1) / Length(l2, l1));
        return min(Length(p, l1), Length(p, l2));
    }


    static inline float RelativeRate(float a, float b)
    {
        return a > b ? b / a : a / b;
    }
};
