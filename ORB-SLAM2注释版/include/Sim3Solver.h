/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef SIM3SOLVER_H
#define SIM3SOLVER_H

#include <opencv2/opencv.hpp>
#include <vector>

#include "KeyFrame.h"



namespace ORB_SLAM2
{

class Sim3Solver
{
public:

    Sim3Solver(KeyFrame* pKF1, KeyFrame* pKF2, const std::vector<MapPoint*> &vpMatched12, const bool bFixScale = true);

    //设置RANSAC随机求解时的参数
    void SetRansacParameters(double probability = 0.99, int minInliers = 6 , int maxIterations = 300);

    cv::Mat find(std::vector<bool> &vbInliers12, int &nInliers);

    cv::Mat iterate(int nIterations, bool &bNoMore, std::vector<bool> &vbInliers, int &nInliers);

    cv::Mat GetEstimatedRotation();
    cv::Mat GetEstimatedTranslation();
    float GetEstimatedScale();


protected:
    //计算质心
    void ComputeCentroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C);

    void ComputeSim3(cv::Mat &P1, cv::Mat &P2);

    void CheckInliers();

    void Project(const std::vector<cv::Mat> &vP3Dw, std::vector<cv::Mat> &vP2D, cv::Mat Tcw, cv::Mat K);
    void FromCameraToImage(const std::vector<cv::Mat> &vP3Dc, std::vector<cv::Mat> &vP2D, cv::Mat K);


protected:

    // KeyFrames and matches
    KeyFrame* mpKF1;
    KeyFrame* mpKF2;

    std::vector<cv::Mat> mvX3Dc1;//保存pKF1中地图点在当前相机坐标系下的坐标
    std::vector<cv::Mat> mvX3Dc2;//保存pKF2中地图点在当前相机坐标系下的坐标
    std::vector<MapPoint*> mvpMapPoints1;
    std::vector<MapPoint*> mvpMapPoints2;
    std::vector<MapPoint*> mvpMatches12;
    std::vector<size_t> mvnIndices1;//容器存放（两个用于计算的KFs中的一个）在pKF2中能够找到对应匹配地图点的pKF1地图点索引。值就是索引
    std::vector<size_t> mvSigmaSquare1;
    std::vector<size_t> mvSigmaSquare2;
    std::vector<size_t> mvnMaxError1;
    std::vector<size_t> mvnMaxError2;

    int N;//pKF间相匹配的地图点的个数
    int mN1;//pKF1中地图点的个数

    // Current Estimation
    cv::Mat mR12i;
    cv::Mat mt12i;
    float ms12i;
    cv::Mat mT12i;
    cv::Mat mT21i;
    std::vector<bool> mvbInliersi;//对所有匹配的地图点进行其是否是内点的标记。下标是pKF1的地图点索引
    int mnInliersi;//计算出Sim3后，其内点数量

    // Current Ransac State当前状态下RANSAC的状态
    int mnIterations;//整个迭代器总的迭代数
    std::vector<bool> mvbBestInliers;
    int mnBestInliers;
    cv::Mat mBestT12;
    cv::Mat mBestRotation;
    cv::Mat mBestTranslation;
    float mBestScale;

    // Scale is fixed to 1 in the stereo/RGBD case
    bool mbFixScale;

    // Indices for random selection
    std::vector<size_t> mvAllIndices;//容器mvX3Dc2、mvX3Dc1中元素的索引

    // Projections
    std::vector<cv::Mat> mvP1im1;//保存图像一中特征点的像素坐标
    std::vector<cv::Mat> mvP2im2;//图像2的

    // RANSAC probability
    double mRansacProb;

    // RANSAC min inliers
    int mRansacMinInliers;//RANSAC求解所要求的最少内点数量

    // RANSAC max iterations
    int mRansacMaxIts;//最大的迭代次数（这里的“最大”表示的是这些匹配点对最多只能迭代多少次。不是每次的极限）

    // Threshold inlier/outlier. e = dist(Pi,T_ij*Pj)^2 < 5.991*mSigma2
    float mTh;
    float mSigma2;

    // Calibration
    cv::Mat mK1;
    cv::Mat mK2;

};

} //namespace ORB_SLAM

#endif // SIM3SOLVER_H
