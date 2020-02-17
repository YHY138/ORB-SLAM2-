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

#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include "KeyFrame.h"
#include "Map.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"

#include <mutex>


namespace ORB_SLAM2
{

class Tracking;
class LoopClosing;
class Map;

class LocalMapping
{
public:
    LocalMapping(Map* pMap, const float bMonocular);

    void SetLoopCloser(LoopClosing* pLoopCloser);

    void SetTracker(Tracking* pTracker);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame* pKF);

    // Thread Synch
    void RequestStop();
    void RequestReset();
    bool Stop();
    void Release();
    bool isStopped();
    bool stopRequested();
    bool AcceptKeyFrames();//局部建图线程能否接受KF
    void SetAcceptKeyFrames(bool flag);
    bool SetNotStop(bool flag);

    void InterruptBA();//打断BA优化

    void RequestFinish();
    bool isFinished();

    //有多少KFs在队列中（建图的队列中）
    int KeyframesInQueue(){
        unique_lock<std::mutex> lock(mMutexNewKFs);
        return mlNewKeyFrames.size();
    }

protected:

    bool CheckNewKeyFrames();//等待处理的KFs队列是否为空
    void ProcessNewKeyFrame();
    void CreateNewMapPoints();

    void MapPointCulling();//地图点剔除
    void SearchInNeighbors();

    void KeyFrameCulling();//KF剔除

    cv::Mat ComputeF12(KeyFrame* &pKF1, KeyFrame* &pKF2);//计算2->1的单应矩阵

    cv::Mat SkewSymmetricMatrix(const cv::Mat &v);//？？

    bool mbMonocular;

    //程序重启
    void ResetIfRequested();
    bool mbResetRequested;
    std::mutex mMutexReset;

    //程序是否停止
    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    Map* mpMap;

    LoopClosing* mpLoopCloser;
    Tracking* mpTracker;

    // Tracking线程向LocalMapping中插入关键帧是先插入到该队列中
    std::list<KeyFrame*> mlNewKeyFrames; ///< 等待处理的关键帧列表

    KeyFrame* mpCurrentKeyFrame;//当前正在处理的KF

    std::list<MapPoint*> mlpRecentAddedMapPoints;//最近添加的地图点？？

    std::mutex mMutexNewKFs;

    bool mbAbortBA;//是否终止BA

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;

    bool mbAcceptKeyFrames;//能否接受KF标志
    std::mutex mMutexAccept;
};

} //namespace ORB_SLAM

#endif // LOCALMAPPING_H
