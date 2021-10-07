/*
 * 这是 TJU Robomasters 上位机源码，未经管理层允许严禁传播给其他人（包括队内以及队外）
 *
 * CameraView 封装了对相机计算的相关操作，保存内参矩阵，畸变参数等
 * Adv版本的改动主要在于增加了对摄像头固定的旋转误差进行了矫正
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include "util.hpp"
#include "configurations.hpp"
#include "serial.hpp"
using namespace std;
using namespace cv;

class CameraView {
public:

    // statics---------------------------------------------------------------

    static Point2f VecToAngle(Point3f vec, float ang_rate = 0.5f) {
        Point2f angles;
        angles.x = atan(vec.x / vec.z) * Rad2Deg;
        angles.y = atan(vec.y / sqrt(vec.x * vec.x + vec.z * vec.z)) * Rad2Deg;
        return angles * ang_rate;
    }


    static double VecLen(Point2f vec) {
        return sqrt(vec.x * vec.x + vec.y * vec.y);
    }

    static double VecLen(Point3f vec) {
        return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
    }

    static double VecLen(Point3d vec){
        return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
    }


// 这部分是相机标定程序，流程为首先设定好用于标定的图像数量 然后将图片转化为灰度图像 寻找角点-》亚像素检测角点-》imageindex加1-》达到标定数量 退出
    static void CalibrateCameraProcess(string savingpath, VideoCapture &cap) {
        cout << "Calibration Started,press p to pick a frame, press q to quit" << endl;
        cout << "Enter Img Count : ";
        int imageIndex = 0,imgCount = 10;
        cin >> imgCount;

        Mat frame, greyframe;
        vector<vector<Point2f>> corner_seq;
        Size patternSize(15, 11), imageSize;//pattern size 棋盘格角点的行列数
        cout << "Enter Pattern Size (width,height): ";
        cin >> patternSize.width >> patternSize.height;
        while (1) {
            cap >> frame;
            if (ConfigurationVariables::resolutionType == 0)
                resize(frame,frame,Size(960,720));
            imshow("Camera Calibration", frame);
            char key =(char) waitKey(20);
            if (key == 'q') {
                cout << "Abundon Calibration." << endl;
                return;
            }
            else if (key == 'p') {
                vector<Point2f> corners;//检测到的棋盘格的角点
                imageSize = frame.size(); //get the height and width from the frame
                cvtColor(frame, greyframe, CV_RGB2GRAY);//convert the image to the grey image
                // findChessboardCorners棋盘格角点检测函数
                if (findChessboardCorners(greyframe, patternSize, corners,
                                          CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK)) {
                    cout << "Captured the " << imageIndex << "th image.";
                    cornerSubPix(greyframe, corners, Size(11, 11), Size(-1, -1),
                                 TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1)); //cornerSubPix亚像素检测函数
                    cout << "Press Y to continue ,N to abundon." << endl;
                    for (int i = 0; i < corners.size(); i++)
                        circle(frame, corners[i], 4, Scalar(255, 0, 0)); //draw circle in the every detected corners
                    imshow("Camera Calibration", frame);
                    while (1) {
                        char k = (char)waitKey(50);
                        if (k == 'n') break;
                        if (k == 'y') {
                            corner_seq.push_back(corners);
                            imageIndex++;
                            break;
                        }
                    }
                    if (imageIndex == imgCount)
                        break;
                } else {
                    cout << "Useless Image!" << endl;
                }
            }
        }
        cout << "Capture Done\nEnter the actual size of each Checkboard\nwidth,height:" << endl;
        
        Size2f squareSize;
        vector<vector<Point3f>> object_seq;
        cin >> squareSize.width >> squareSize.height;
        for (int t = 0; t < imgCount; t++) {
            vector<Point3f> tempPointSet;
            for (int i = 0; i < patternSize.height; i++) {
                for (int j = 0; j < patternSize.width; j++) {
                    
                    Point3f tempPoint;
                    tempPoint.x = i * squareSize.width;
                    tempPoint.y = j * squareSize.height;
                    tempPoint.z = 0;
                    tempPointSet.push_back(tempPoint);
                }
            }
            object_seq.push_back(tempPointSet);
        }
        Mat intrinsic_matrix, distortion_coeffs;
        vector<Mat> rotation_vectors, translation_vectors;
        double calRes = calibrateCamera(object_seq, corner_seq, imageSize, intrinsic_matrix, distortion_coeffs,
                                        rotation_vectors, translation_vectors);//计算相机的内参和外参
        cout << calRes << "Calibration Done\nInstrinsic Matrix" << endl << intrinsic_matrix << endl <<
             "Distortion Coeffs" << endl << distortion_coeffs << endl << endl << "Calc Error Analyse" << endl;
        double total_err = 0;
        for (int i = 0; i < imgCount; i++) {
            vector<Point3f> tempPointSet = object_seq[i];
            vector<Point2f> image_points2;
            
            projectPoints(tempPointSet, rotation_vectors[i], translation_vectors[i], intrinsic_matrix,
                          distortion_coeffs, image_points2);//将世界坐标系中的点转换到图像坐标系 image_points2是输出的转换后的图像坐标系的点
            
            vector<Point2f> tempImagePoint = corner_seq[i];//corner_seq 是检测到的角点
            Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
            Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);
            for (int j = 0; j < tempImagePoint.size(); j++) {
                image_points2Mat.at<Vec2f>(0, j) = Vec2f(image_points2[j].x, image_points2[j].y);
                tempImagePointMat.at<Vec2f>(0, j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
            }
            double err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
            total_err += err /= image_points2.size();
            cout << "The " << i + 1 << "th image:Average Error " << err << " px" << endl; //每一张图片的平均误差
        }
        cout << "Total average error  " << total_err / imgCount << " px" << endl; //所有图片的总误差
        cout << "Analyse done!" << endl << endl;
        // show rotation vectors and translation vectors
        for (int i = 0; i < imgCount; i++) {
            cout << "The " << i + 1 << "th rotation vector:" << endl << rotation_vectors[i] << endl;
            cout << "Translation :" << endl << translation_vectors[i] << endl << endl;
        }

        // now saving
        FileStorage fs(savingpath, FileStorage::WRITE);
        fs << "intrinsic_matrix" << intrinsic_matrix;
        fs << "distortion_coeffs" << distortion_coeffs;
        fs << "imageSize" << imageSize;

        cout << "saved" << endl;
        fs.release();

        //system("pause");
    }

    static vector<Point2f> MakeRectPointsListInOrder(Point2f pts[], int count = 4) {
        vector<Point2f> vps;
        FOREACH(i, count)vps.push_back(pts[i]);
        return MakeRectPointsListInOrder(vps);
    }

    static vector<Point2f> MakeRectPointsListInOrder(vector<Point2f> pts) {
        float plus_max = -9999999, plus_min = 9999999, minus_max = -99999999, minus_min = 99999999;
        int plus_maxi, plus_mini, minus_maxi, minus_mini;
        for (int i = 0; i < pts.size(); i++) {
            float t = pts[i].x + pts[i].y;
            if (t > plus_max) {
                plus_max = t;
                plus_maxi = i;
            }
            if (t < plus_min) {
                plus_min = t;
                plus_mini = i;
            }
            t = pts[i].y - pts[i].x;
            if (t > minus_max) {
                minus_max = t;
                minus_maxi = i;
            }
            if (t < minus_min) {
                minus_min = t;
                minus_mini = i;
            }
        }
        vector<Point2f> vcpts;
        vcpts.push_back(pts[plus_mini]);
        vcpts.push_back(pts[minus_mini]);
        vcpts.push_back(pts[plus_maxi]);
        vcpts.push_back(pts[minus_maxi]);
        return vcpts;
    }

    static Mat EulerAnglesToRotationMatrix(Vec3f &theta)
    {
        // 计算旋转矩阵的X分量
        Mat R_x = (Mat_<double>(3,3) <<
                   1,       0,              0,
                   0,       cos(theta[0]),   -sin(theta[0]),
                   0,       sin(theta[0]),   cos(theta[0])
                   );
        // 计算旋转矩阵的Y分量
        Mat R_y = (Mat_<double>(3,3) <<
                   cos(theta[1]),    0,      sin(theta[1]),
                   0,               1,      0,
                   -sin(theta[1]),   0,      cos(theta[1])
                   );
        // 计算旋转矩阵的Z分量
        Mat R_z = (Mat_<double>(3,3) <<
                   cos(theta[2]),    -sin(theta[2]),      0,
                   sin(theta[2]),    cos(theta[2]),       0,
                   0,               0,                  1);
        // 合并 
        Mat R = R_z * R_y * R_x;
        return R;
    }
    // ------------------------------------------------------------------------

    // world offset length for camera to ptz, use mm unit
    double CameraOffsetZ = 140,
            CameraOffsetX = 20,
            CameraOffsetY = 20,
            CameraOffsetYaw = 0 ,
            CameraOffsetPitch = 0 ,
            CameraOffsetRoll = 0;

    // initialize directly 从标定好的相机文件中直接读取相应的矩阵
    CameraView(Mat intrinsic_matrix, Mat distortion_coeffs, Size imageSize) {
        this->intrinsic_matrix = intrinsic_matrix;
        this->distortion_coeffs = distortion_coeffs;
        this->imageSize = imageSize;

        InitUndistortParams();
        LoadConfigParams();
    }

    // initialize from file data
    CameraView(string filePath) {
        LoadFile(filePath);
        LoadConfigParams();
    }

    CameraView()
    {
        LoadConfigParams();
    }

    void LoadFile(string filePath)
    {
        FileStorage fs(filePath, FileStorage::READ);
        fs["intrinsic_matrix"] >> intrinsic_matrix;
        fs["distortion_coeffs"] >> distortion_coeffs;
        fs["imageSize"] >> imageSize;
        fs.release();
        InitUndistortParams();
    }

     Mat ConvertCameraVectoPTZVec(Mat dt) {
        // 先应用旋转变换
        // Mat pmat = (Mat_<double>(3,1) << dt.x , dt.y , dt.z);
        Mat pmat = camera_offset_rot_matrix * dt;
        // 叠加上平移变换
        // cout << "pmat.at<double>(0,0)" << pmat.at<double>(0,0) << endl;
        // cout << "pmat.at<double>(1,0)" << pmat.at<double>(1,0) << endl;
        // cout << "pmat.at<double>(2,0)" << pmat.at<double>(2,0) << endl;
        Mat pmt = (Mat_<double>(3,1)<< CameraOffsetX + pmat.at<double>(0,0),
                       CameraOffsetY + pmat.at<double>(1,0),
                       CameraOffsetZ + pmat.at<double>(2,0));
        return pmt;
    }

    Point3d ConvertCameraVectoPTZVec(Point3d dt) {
        // 先应用旋转变换
        Mat pmat = (Mat_<double>(3,1) << dt.x , dt.y , dt.z);
        pmat = camera_offset_rot_matrix * pmat;
        // 叠加上平移变换
        // cout << "pmat.at<double>(0,0)" << pmat.at<double>(0,0) << endl;
        // cout << "pmat.at<double>(1,0)" << pmat.at<double>(1,0) << endl;
        // cout << "pmat.at<double>(2,0)" << pmat.at<double>(2,0) << endl;
        return Point3d(CameraOffsetX + pmat.at<double>(0,0),
                       CameraOffsetY + pmat.at<double>(1,0),
                       CameraOffsetZ + pmat.at<double>(2,0));
    }

    Point3d ScreenPointToVec(Point2f scrp) {
        Mat scrpMat(3, 1, CV_64F), res;
        scrpMat.at<double>(0, 0) = scrp.x;
        scrpMat.at<double>(1, 0) = scrp.y;
        scrpMat.at<double>(2, 0) = 1;
        res = invert_intrinsic * scrpMat;
        return Point3d(res.at<double>(0, 0), -res.at<double>(1, 0), res.at<double>(2, 0));
    }

    Point2d ScreenPointToAngle(Point2d scrp, float ang_rate = 0.5f) {
        return VecToAngle(ScreenPointToVec(scrp), ang_rate);
    }

    Point2d ScreenPointToPTZAngle(Point2d scrp, float dis, float ang_rate = 1.0f) {
        Point3d vec = ScreenPointToVec(scrp);   //图像坐标系转相机坐标系
        vec = ConvertCameraVectoPTZVec(vec * (dis/VecLen(vec)));    //相机坐标系转世界坐标系
        return VecToAngle(vec, ang_rate);
        
    }
    Point2d AngleToScreenPoint(Point2f rad)
    {
        double d = 1.0 / cos(rad.x);
        Mat vec(3,1,CV_64F),res;
        vec.at<double>(0,0) = tan(rad.x);
        vec.at<double>(1,0) = d * tan(-rad.y);
        vec.at<double>(2,0) = 1;
        res = intrinsic_matrix * vec;
        return Point2d(res.at<double>(0,0),res.at<double>(1,0));
    }
    Point2d PTZAngleToScreenPoint(Point2f rad,float dis)
    {
        // convert to camera angle first
        Point3d worldP;
        worldP.y = dis * sin(rad.y) - CameraOffsetY;
        double d = dis * cos(rad.y);
        worldP.x = d * sin(rad.x) - CameraOffsetX;
        worldP.z = d * cos(rad.x) - CameraOffsetZ;
        // make rotation transform
        Mat v,r;
        v = (Mat_<double>(3,1)<<worldP.x,worldP.y,worldP.z);
        r = inv_camera_offset_rot_matrix * v;
        r /= r.at<double>(2,0);
        r.at<double>(1,0) = -r.at<double>(1,0);
        v = intrinsic_matrix * r;
        return Point2d(v.at<double>(0,0),v.at<double>(1,0));
    }

    // ��������
    void Undistort(Mat &image, Mat &dstImage) {
        remap(image, dstImage, map1, map2, INTER_LINEAR);
    }

    double CalcRectDistance(vector<Point2f> scrpts, float worldWidth, float worldHeight) {
        vector<Point3f> worldpts;
        worldpts.push_back(Point3f(-worldWidth / 2, worldHeight / 2, 0));
        worldpts.push_back(Point3f(worldWidth / 2, worldHeight / 2, 0));
        worldpts.push_back(Point3f(worldWidth / 2, -worldHeight / 2, 0));
        worldpts.push_back(Point3f(-worldWidth / 2, -worldHeight / 2, 0));
        Mat rotationVec = Mat::zeros(3, 1, CV_64FC1),
                translationVec = Mat::zeros(3, 1, CV_64FC1);
        //Mat rotationVec , translationVec ;
        Mat objPM;
        Mat(worldpts).convertTo(objPM, CV_32F);
        solvePnP(objPM, Mat(scrpts), intrinsic_matrix, distortion_coeffs, rotationVec, translationVec, false, CV_EPNP);
        //cout << translationVec << endl;
        double sum = 0;
        FOREACH(i, 3) sum += translationVec.at<double>(i, 0) * translationVec.at<double>(i, 0);
        return sqrt(sum);
    }

    bool CalcChessboardGesture(Mat frame, Mat &matr, Size2f worldSize, Size pattern) {
        Mat grey;
        vector<Point2f> corners, rect_cor;
        imageSize = frame.size();
        cvtColor(frame, grey, CV_RGB2GRAY);

        if (findChessboardCorners(grey, pattern, corners,
                                  CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK)) {
            cornerSubPix(grey, corners, Size(11, 11), Size(-1, -1),
                         TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
            rect_cor = MakeRectPointsListInOrder(corners);
            for (int i = 0; i < rect_cor.size(); i++)
                circle(frame, rect_cor[i], 4 + i * 2, Scalar(255, 0, 0));
            // calc gesture
            vector<Point3f> worldpts;
            worldpts.push_back(Point3f(-worldSize.width / 2, worldSize.height / 2, 0));
            worldpts.push_back(Point3f(worldSize.width / 2, worldSize.height / 2, 0));
            worldpts.push_back(Point3f(worldSize.width / 2, -worldSize.height / 2, 0));
            worldpts.push_back(Point3f(-worldSize.width / 2, -worldSize.height / 2, 0));
            Mat rotationVec = Mat::zeros(3, 1, CV_64FC1),
                    translationVec = Mat::zeros(3, 1, CV_64FC1);
            //Mat rotationVec , translationVec ;
            Mat objPM;
            Mat(worldpts).convertTo(objPM, CV_32F);
            solvePnP(objPM, Mat(rect_cor), intrinsic_matrix, distortion_coeffs, rotationVec, translationVec, false,
                     CV_EPNP);
            // make the matrix
            matr = Mat::zeros(4, 4, CV_64F);
            Rodrigues(rotationVec, matr(Rect(0, 0, 3, 3)));
            translationVec.copyTo(matr(Rect(3, 0, 1, 3)));
            matr.at<double>(3, 3) = 1;

            cout << "solved distance : " << CalcRectDistance(rect_cor, worldSize.width, worldSize.height) << endl;
            return true;
        } else {
            return false;
        }
    }

    void Navy_HandEye(Mat &Hcg, vector<Mat> Hgij, vector<Mat> Hcij) {
        CV_Assert(Hgij.size() == Hcij.size());
        int nStatus = Hgij.size();
        Mat Rgij(3, 3, CV_64FC1);
        Mat Rcij(3, 3, CV_64FC1);

        Mat alpha1(3, 1, CV_64FC1);
        Mat beta1(3, 1, CV_64FC1);
        Mat alpha2(3, 1, CV_64FC1);
        Mat beta2(3, 1, CV_64FC1);
        Mat A(3, 3, CV_64FC1);
        Mat B(3, 3, CV_64FC1);

        Mat alpha(3, 1, CV_64FC1);
        Mat beta(3, 1, CV_64FC1);
        Mat M(3, 3, CV_64FC1, Scalar(0));

        Mat MtM(3, 3, CV_64FC1);
        Mat veMtM(3, 3, CV_64FC1);
        Mat vaMtM(3, 1, CV_64FC1);
        Mat pvaM(3, 3, CV_64FC1, Scalar(0));

        Mat Rx(3, 3, CV_64FC1);

        Mat Tgij(3, 1, CV_64FC1);
        Mat Tcij(3, 1, CV_64FC1);

        Mat eyeM = Mat::eye(3, 3, CV_64FC1);

        Mat tempCC(3, 3, CV_64FC1);
        Mat tempdd(3, 1, CV_64FC1);

        Mat C;
        Mat d;
        Mat Tx(3, 1, CV_64FC1);

        //Compute rotation
        if (Hgij.size() == 2) // Two (Ai,Bi) pairs
        {
            Rodrigues(Hgij[0](Rect(0, 0, 3, 3)), alpha1);
            Rodrigues(Hgij[1](Rect(0, 0, 3, 3)), alpha2);
            Rodrigues(Hcij[0](Rect(0, 0, 3, 3)), beta1);
            Rodrigues(Hcij[1](Rect(0, 0, 3, 3)), beta2);

            alpha1.copyTo(A.col(0));
            alpha2.copyTo(A.col(1));
            (alpha1.cross(alpha2)).copyTo(A.col(2));

            beta1.copyTo(B.col(0));
            beta2.copyTo(B.col(1));
            (beta1.cross(beta2)).copyTo(B.col(2));

            Rx = A * B.inv();

        } else // More than two (Ai,Bi) pairs
        {
            for (int i = 0; i < nStatus; i++) {
                Hgij[i](Rect(0, 0, 3, 3)).copyTo(Rgij);
                Hcij[i](Rect(0, 0, 3, 3)).copyTo(Rcij);

                Rodrigues(Rgij, alpha);
                Rodrigues(Rcij, beta);

                M = M + beta * alpha.t();
            }

            MtM = M.t() * M;
            eigen(MtM, vaMtM, veMtM);

            pvaM.at<double>(0, 0) = 1 / sqrt(vaMtM.at<double>(0, 0));
            pvaM.at<double>(1, 1) = 1 / sqrt(vaMtM.at<double>(1, 0));
            pvaM.at<double>(2, 2) = 1 / sqrt(vaMtM.at<double>(2, 0));

            Rx = veMtM * pvaM * veMtM.inv() * M.t();
        }

        //Computer Translation
        for (int i = 0; i < nStatus; i++) {

            Hgij[i](Rect(0, 0, 3, 3)).copyTo(Rgij);
            Hcij[i](Rect(0, 0, 3, 3)).copyTo(Rcij);
            Hgij[i](Rect(3, 0, 1, 3)).copyTo(Tgij);
            Hcij[i](Rect(3, 0, 1, 3)).copyTo(Tcij);

            tempCC = eyeM - Rgij;
            tempdd = Tgij - Rx * Tcij;

            C.push_back(tempCC);
            d.push_back(tempdd);
        }

        Tx = (C.t() * C).inv() * (C.t() * d);

        Hcg = Mat::zeros(4, 4, CV_64F);
        Rx.copyTo(Hcg(Rect(0, 0, 3, 3)));
        Tx.copyTo(Hcg(Rect(3, 0, 1, 3)));
        Hcg.at<double>(3, 0) = 0.0;
        Hcg.at<double>(3, 1) = 0.0;
        Hcg.at<double>(3, 2) = 0.0;
        Hcg.at<double>(3, 3) = 1.0;

    }

    Mat get_intrinsic_matrix()
    {
        return intrinsic_matrix;
    }
    Mat get_distortion_coeffs()
    {
        return distortion_coeffs;
    }
protected:

    Mat intrinsic_matrix, distortion_coeffs, invert_intrinsic, camera_offset_rot_matrix,inv_camera_offset_rot_matrix;
    Mat map1, map2;
    Size imageSize;

    void InitUndistortParams() {
        initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, Mat(), Mat(),
                                imageSize, CV_16SC2, map1, map2);
        invert(intrinsic_matrix, invert_intrinsic, DECOMP_LU);
    }

    void LoadConfigParams()
    {
        SET_CONFIG_DOUBLE_VARIABLE(CameraOffsetX,20);
        SET_CONFIG_DOUBLE_VARIABLE(CameraOffsetY,20);
        SET_CONFIG_DOUBLE_VARIABLE(CameraOffsetZ,140);
        SET_CONFIG_DOUBLE_CONST(CameraOffsetYaw,0);
        SET_CONFIG_DOUBLE_CONST(CameraOffsetPitch,0);
        SET_CONFIG_DOUBLE_CONST(CameraOffsetRoll,0);
        // 初始化旋转偏置的旋转矩阵
        Vec3f vec(CameraOffsetPitch*Deg2Rad,CameraOffsetYaw*Deg2Rad,CameraOffsetRoll*Deg2Rad);
        camera_offset_rot_matrix = EulerAnglesToRotationMatrix(vec);
        invert(camera_offset_rot_matrix,inv_camera_offset_rot_matrix,DECOMP_LU);
    }

};
