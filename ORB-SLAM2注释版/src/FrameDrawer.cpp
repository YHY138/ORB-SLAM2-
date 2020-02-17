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

#include "FrameDrawer.h"
#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include<mutex>

namespace ORB_SLAM2
{

FrameDrawer::FrameDrawer(Map* pMap):mpMap(pMap)
{
    mState=Tracking::SYSTEM_NOT_READY;
    // 初始化图像显示画布
    // 包括：图像、特征点连线形成的轨迹（初始化时）、框（跟踪时的MapPoint）、圈（跟踪时的特征点）
    // ！！！固定画布大小为640*480
    mIm = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
}

// 准备需要显示的信息，包括图像、特征点、地图、跟踪状态
cv::Mat FrameDrawer::DrawFrame()
{
    cv::Mat im;
    vector<cv::KeyPoint> vIniKeys; // Initialization: KeyPoints in reference frame在参考帧中的特征点
    vector<int> vMatches; // Initialization: correspondeces with reference keypoints参考帧和当前帧的匹配关系
    vector<cv::KeyPoint> vCurrentKeys; // KeyPoints in current frame当前帧特征点
    vector<bool> vbVO, vbMap; // Tracked MapPoints in current frame在当前帧中被跟踪到的地图点
    int state; // Tracking state跟踪状态

    //Copy variables within scoped mutex
    // 步骤1：将成员变量赋值给局部变量（包括图像、状态、其它的提示）
    // 加互斥锁，避免与FrameDrawer::Update函数中图像拷贝发生冲突？？
    {
        unique_lock<mutex> lock(mMutex);
        state=mState;
        if(mState==Tracking::SYSTEM_NOT_READY)
            mState=Tracking::NO_IMAGES_YET;

        // 这里使用copyTo进行深拷贝是因为后面会把单通道灰度图像转为3通道图像
        mIm.copyTo(im);//把mIm拷贝到im

        if(mState==Tracking::NOT_INITIALIZED)//还没进行初始化
        {
			//设置参考、当前帧的关键点，以及两者间的匹配关系
            vCurrentKeys = mvCurrentKeys;
            vIniKeys = mvIniKeys;
            vMatches = mvIniMatches;
        }
        else if(mState==Tracking::OK)//正在工作
        {
			//
            vCurrentKeys = mvCurrentKeys;
            vbVO = mvbVO;
            vbMap = mvbMap;
        }
        else if(mState==Tracking::LOST)//跟踪丢失
        {
            vCurrentKeys = mvCurrentKeys;//只将当前帧的关键点保存下来
        }
    } // destroy scoped mutex -> release mutex

    if(im.channels()<3) //this should be always true
        cvtColor(im,im,CV_GRAY2BGR);//将im从灰度图转成RGB图

    //Draw
    // 步骤2：绘制初始化轨迹连线，绘制特征点边框（特征点用小框圈住）
    // 步骤2.1：初始化时，当前帧的特征坐标与初始帧的特征点坐标连成线，形成轨迹
    if(state==Tracking::NOT_INITIALIZED) //INITIALIZING？？？im就一张图的大小，怎么画两个图的连线？？
    {
        for(unsigned int i=0; i<vMatches.size(); i++)
        {
            if(vMatches[i]>=0)
            {
                cv::line(im,vIniKeys[i].pt,vCurrentKeys[vMatches[i]].pt,
                        cv::Scalar(0,255,0));
            }
        }
    }
    else if(state==Tracking::OK) //TRACKING
    {
        mnTracked=0;//跟踪到的地图点的数目
        mnTrackedVO=0;//跟踪到上一帧产生的地图点的数目

        // Draw keypoints
        const float r = 5;//画的框框的半径
        const int n = vCurrentKeys.size();//当前帧特征点数量
        for(int i=0;i<n;i++)
        {
            if(vbVO[i] || vbMap[i])
            {
                //在特征点附近正方形选择四个点
                cv::Point2f pt1,pt2;
                pt1.x=vCurrentKeys[i].pt.x-r;
                pt1.y=vCurrentKeys[i].pt.y-r;
                pt2.x=vCurrentKeys[i].pt.x+r;
                pt2.y=vCurrentKeys[i].pt.y+r;

                // This is a match to a MapPoint in the map
                // 步骤2.2：正常跟踪时，在画布im中标注特征点
                if(vbMap[i])//将图中跟踪到的地图点用框框画出来
                {
                    // 通道顺序为bgr，地图中MapPoints用绿色圆点表示，并用绿色小方框圈住
                    cv::rectangle(im,pt1,pt2,cv::Scalar(0,255,0));
                    cv::circle(im,vCurrentKeys[i].pt,2,cv::Scalar(0,255,0),-1);
                    mnTracked++;
                }
                else // This is match to a "visual odometry" MapPoint created in the last frame将当前帧观测到的上一帧产生的地图点画出来
                {
                    // 通道顺序为bgr，仅当前帧能观测到的MapPoints用蓝色圆点表示，并用蓝色小方框圈住
                    cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,0));
                    cv::circle(im,vCurrentKeys[i].pt,2,cv::Scalar(255,0,0),-1);
                    mnTrackedVO++;
                }
            }
        }
    }

    cv::Mat imWithInfo;
    DrawTextInfo(im,state, imWithInfo);//在画出的图像上加一些文字

    return imWithInfo;
}


void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
{
    stringstream s;
    if(nState==Tracking::NO_IMAGES_YET)
        s << " WAITING FOR IMAGES";
    else if(nState==Tracking::NOT_INITIALIZED)
        s << " TRYING TO INITIALIZE ";
    else if(nState==Tracking::OK)
    {
        if(!mbOnlyTracking)//判断什么模式
            s << "SLAM MODE |  ";
        else
            s << "LOCALIZATION | ";
        int nKFs = mpMap->KeyFramesInMap();//在地图中有多少个关键帧和地图点
        int nMPs = mpMap->MapPointsInMap();
        s << "KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;//当前帧跟踪到的地图点的数量
        if(mnTrackedVO>0)
            s << ", + VO matches: " << mnTrackedVO;//当前帧跟踪到的上一帧产生的地图点的数量
    }
    else if(nState==Tracking::LOST)
    {
        s << " TRACK LOST. TRYING TO RELOCALIZE ";
    }
    else if(nState==Tracking::SYSTEM_NOT_READY)
    {
        s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
    }

    int baseline=0;
    cv::Size textSize = cv::getTextSize(s.str(),cv::FONT_HERSHEY_PLAIN,1,1,&baseline);//如何来写这段文字（字体啥的）

    imText = cv::Mat(im.rows+textSize.height+10,im.cols,im.type());
    im.copyTo(imText.rowRange(0,im.rows).colRange(0,im.cols));//把图像放到imText中的一部分区域内
    imText.rowRange(im.rows,imText.rows) = cv::Mat::zeros(textSize.height+10,im.cols,im.type());
    cv::putText(imText,s.str(),cv::Point(5,imText.rows-5),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1,8);//在图像上绘制文字

}

//将跟踪线程的数据拷贝到绘图线程（图像、特征点、地图、跟踪状态）
void FrameDrawer::Update(Tracking *pTracker)//这里只是获得了当前帧的关键点，而程序能够将当前帧关键点转移给参考帧关键点，使之成为新的参考？？？
{
    unique_lock<mutex> lock(mMutex);
    //拷贝跟踪线程的图像
    pTracker->mImGray.copyTo(mIm);//将跟踪线程的图像放到要画的图
    //拷贝跟踪线程的特征点
    mvCurrentKeys=pTracker->mCurrentFrame.mvKeys;//获得当前帧的关键点
    N = mvCurrentKeys.size();
    mvbVO = vector<bool>(N,false);
    mvbMap = vector<bool>(N,false);
    //mbOnlyTracking等于false表示正常VO模式（有地图更新），mbOnlyTracking等于true表示用户手动选择定位模式
    mbOnlyTracking = pTracker->mbOnlyTracking;


    if(pTracker->mLastProcessedState==Tracking::NOT_INITIALIZED)//跟踪线程的上一个状态时未初始化
    {
        mvIniKeys=pTracker->mInitialFrame.mvKeys;//获得上一帧关键点
        mvIniMatches=pTracker->mvIniMatches;
    }
    else if(pTracker->mLastProcessedState==Tracking::OK)//跟踪线程之前的就已经在工作了
    {
        for(int i=0;i<N;i++)
        {
            MapPoint* pMP = pTracker->mCurrentFrame.mvpMapPoints[i];//获得当前帧关键点随对应的地图点
            if(pMP)//是否有能对应的地图点（根据有无有效深度信息判断）
            {
                if(!pTracker->mCurrentFrame.mvbOutlier[i])//该点是否是外点（无效点）
                {
                    //该mappoints可以被多帧观测到，则为有效的地图点（即当前帧跟踪到了地图中的地图点），否则这个地图点只适用于两帧间的位姿估计
                    if(pMP->Observations()>0)
                        mvbMap[i]=true;
                    else
                        mvbVO[i]=true;//该是否跟踪到了上一帧的地图点？
                }
            }
        }
    }
    mState=static_cast<int>(pTracker->mLastProcessedState);
}

} //namespace ORB_SLAM
