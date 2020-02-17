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

#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "Frame.h"
#include "KeyFrameDatabase.h"

#include <mutex>


namespace ORB_SLAM2
{

class Map;
class MapPoint;
class Frame;
class KeyFrameDatabase;

/* KeyFrame
 * 关键帧，和普通的Frame不一样，但是可以由Frame来构造
 * 许多数据会被三个线程同时访问，所以用锁的地方很普遍
 */
class KeyFrame
{
public:
    KeyFrame(Frame &F, Map* pMap, KeyFrameDatabase* pKFDB);

    // Pose functions
    // 这里的set,get需要用到锁
    void SetPose(const cv::Mat &Tcw);
    cv::Mat GetPose();
    cv::Mat GetPoseInverse();
    cv::Mat GetCameraCenter();
    cv::Mat GetStereoCenter();
    cv::Mat GetRotation();
    cv::Mat GetTranslation();

    // Bag of Words Representation
    void ComputeBoW();//计算关键帧的词袋向量表达

    // Covisibility graph functions共视图类函数
    void AddConnection(KeyFrame* pKF, const int &weight);//添加该关键帧与pKF的联系
    void EraseConnection(KeyFrame* pKF);//删除与pKF的联系
    void UpdateConnections();
    void UpdateBestCovisibles();
    std::set<KeyFrame *> GetConnectedKeyFrames();//获得与该KF联系的其他KFs
    std::vector<KeyFrame* > GetVectorCovisibleKeyFrames();
    std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int &N);//获得与该KF共视性最高的N个KFs
    std::vector<KeyFrame*> GetCovisiblesByWeight(const int &w);//获得共视程度为w的与该KF关联的KFs
    int GetWeight(KeyFrame* pKF);//获得某个KF与该KF的共视程度

    // Spanning tree functions
    void AddChild(KeyFrame* pKF);
    void EraseChild(KeyFrame* pKF);
    void ChangeParent(KeyFrame* pKF);
    std::set<KeyFrame*> GetChilds();
    KeyFrame* GetParent();
    bool hasChild(KeyFrame* pKF);

    // Loop Edges
    void AddLoopEdge(KeyFrame* pKF);
    std::set<KeyFrame*> GetLoopEdges();

    // MapPoint observation functions
    void AddMapPoint(MapPoint* pMP, const size_t &idx);
    void EraseMapPointMatch(const size_t &idx);
    void EraseMapPointMatch(MapPoint* pMP);
    void ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP);
    std::set<MapPoint*> GetMapPoints();
    std::vector<MapPoint*> GetMapPointMatches();
    int TrackedMapPoints(const int &minObs);
    MapPoint* GetMapPoint(const size_t &idx);

    // KeyPoint functions
    std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;
    cv::Mat UnprojectStereo(int i);

    // Image
    bool IsInImage(const float &x, const float &y) const;

    // Enable/Disable bad flag changes
    void SetNotErase();
    void SetErase();

    // Set/check bad flag
    void SetBadFlag();
    bool isBad();

    // Compute Scene Depth (q=2 median). Used in monocular.
    float ComputeSceneMedianDepth(const int q);

    static bool weightComp( int a, int b)
	{
        return a>b;
    }

    static bool lId(KeyFrame* pKF1, KeyFrame* pKF2)
	{
        return pKF1->mnId<pKF2->mnId;
    }


    // The following variables are accesed from only 1 thread or never change (no mutex needed).
public:

    // nNextID名字改为nLastID更合适，表示上一个KeyFrame的ID号
    static long unsigned int nNextId;//上一个KF的ID号
    // 在nNextID的基础上加1就得到了mnID，为当前KeyFrame的ID号
    long unsigned int mnId;
    // 每个KeyFrame基本属性是它是一个Frame，KeyFrame初始化的时候需要Frame，
    // mnFrameId记录了该KeyFrame是由哪个Frame初始化的
    const long unsigned int mnFrameId;//来源于那个ID号的Frame

    const double mTimeStamp;

    // Grid (to speed up feature matching)将关键帧分成了多少个网格以及每个网格的长宽
    // 和Frame类中的定义相同
    const int mnGridCols;
    const int mnGridRows;
    const float mfGridElementWidthInv;
    const float mfGridElementHeightInv;

    // Variables used by the tracking
    long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnFuseTargetForKF;

    // Variables used by the local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnBAFixedForKF;

    // Variables used by the keyframe database
    long unsigned int mnLoopQuery;//关键帧ID号，表明当前KF是这个ID号的KF的候选闭环KF
    int mnLoopWords;//有多少个共有的words
    float mLoopScore;//作为闭环候选的得分
    long unsigned int mnRelocQuery;//重定位：和上述类似
    int mnRelocWords;
    float mRelocScore;

    // Variables used by loop closing
    cv::Mat mTcwGBA;//经过全局BA优化的位姿
    cv::Mat mTcwBefGBA;//未经过全局BA优化的位姿
    long unsigned int mnBAGlobalForKF;

    // Calibration parameters
    const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;

    // Number of KeyPoints
    const int N;

    // KeyPoints, stereo coordinate and descriptors (all associated by an index)
    // 和Frame类中的定义相同
    const std::vector<cv::KeyPoint> mvKeys;
    const std::vector<cv::KeyPoint> mvKeysUn;
    const std::vector<float> mvuRight; // negative value for monocular points
    const std::vector<float> mvDepth; // negative value for monocular points
    const cv::Mat mDescriptors;

    //BoW
    DBoW2::BowVector mBowVec; ///< Vector of words to represent images描述当前图像的词袋向量
    DBoW2::FeatureVector mFeatVec; ///< Vector of nodes with indexes of local features存放某一帧中的所有节点编号，以及节点中包含的
	//所有特征点在整个特征点容器中的索引
	//大致形式{(node1,feature_vector1) (node2,feature_vector2)...}

    // Pose relative to parent (this is computed when bad flag is activated)
    cv::Mat mTcp;//从父KF到子KF的变换矩阵

    // Scale
    const int mnScaleLevels;//尺度层数
    const float mfScaleFactor;//每层缩放比例
    const float mfLogScaleFactor;
    const std::vector<float> mvScaleFactors;// 尺度因子，scale^n，scale=1.2，n为层数
    const std::vector<float> mvLevelSigma2;// 尺度因子的平方
    const std::vector<float> mvInvLevelSigma2;

    // Image bounds and calibration
    const int mnMinX;
    const int mnMinY;
    const int mnMaxX;
    const int mnMaxY;
    const cv::Mat mK;


    // The following variables need to be accessed trough a mutex to be thread safe.
protected:

    // SE3 Pose and camera center
    cv::Mat Tcw;
    cv::Mat Twc;
    cv::Mat Ow;//相机光心（单目相机或者双目的左相机）

    cv::Mat Cw; // Stereo middel point. Only for visualization双目的中点

    // MapPoints associated to keypoints
    std::vector<MapPoint*> mvpMapPoints;

    // BoW
    KeyFrameDatabase* mpKeyFrameDB;
    ORBVocabulary* mpORBvocabulary;

    // Grid over the image to speed up feature matching
    std::vector< std::vector <std::vector<size_t> > > mGrid;//mGrid[i]表示KF图像中的第i行网格；mGrid[i][j]表示图像中坐标为（i，j）的网格；mGrid[i][j]中包含的是该网格
	//中所包含的特征点在其容器中的索引

    // Covisibility Graph
    std::map<KeyFrame*,int> mConnectedKeyFrameWeights; ///< 与该关键帧连接的关键帧与权重
    std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames; ///< 排序后的关键帧
    std::vector<int> mvOrderedWeights; ///< 排序后的权重(从大到小)

    // Spanning Tree and Loop Edges
    // std::set是集合，相比vector，进行插入数据这样的操作时会自动排序
    //生成树：KF的父子节点
    bool mbFirstConnection;
    KeyFrame* mpParent;
    std::set<KeyFrame*> mspChildrens;
    std::set<KeyFrame*> mspLoopEdges;//回环边？

    // Bad flags好坏的标志
    bool mbNotErase;//检查当前KF是否被抛弃了
    bool mbToBeErased;//当前KF是否将被抛弃
    bool mbBad;    //当前KF好坏

    float mHalfBaseline; // Only for visualization双目相机的中心是两个相机的中点，该点与单个相机的中心仅相差半个基线（在x轴上）

    Map* mpMap;

    std::mutex mMutexPose;
    std::mutex mMutexConnections;
    std::mutex mMutexFeatures;
};

} //namespace ORB_SLAM

#endif // KEYFRAME_H
