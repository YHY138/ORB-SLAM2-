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

#ifndef MAP_H
#define MAP_H

#include "MapPoint.h"
#include "KeyFrame.h"
#include <set>

#include <mutex>



namespace ORB_SLAM2
{

class MapPoint;
class KeyFrame;

class Map
{
public:
    Map();

    void AddKeyFrame(KeyFrame* pKF);//向地图中插入地图点和关键帧
    void AddMapPoint(MapPoint* pMP);
    void EraseMapPoint(MapPoint* pMP);//从地图中删除地图点和关键帧
    void EraseKeyFrame(KeyFrame* pKF);
    void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);

    std::vector<KeyFrame*> GetAllKeyFrames();//获得所有关键帧的指针
    std::vector<MapPoint*> GetAllMapPoints();//指向地图点的指针
    std::vector<MapPoint*> GetReferenceMapPoints();//参考地图点指针（局部地图点）

    long unsigned int MapPointsInMap();//在地图中有多少地图点和关键帧
    long unsigned  KeyFramesInMap();

    long unsigned int GetMaxKFid();//获得最大关键帧的ID

    void clear();

    vector<KeyFrame*> mvpKeyFrameOrigins;//所有关键帧的指针

	std::mutex mMutexMapUpdate;//??????

    // This avoid that two points are created simultaneously in separate threads (id conflict)
    std::mutex mMutexPointCreation;

protected:
    std::set<MapPoint*> mspMapPoints; ///< MapPoints    set是关联容器，通过关键字来查找对应元素
    std::set<KeyFrame*> mspKeyFrames; ///< Keyframs

    std::vector<MapPoint*> mvpReferenceMapPoints;

    long unsigned int mnMaxKFid;

    std::mutex mMutexMap;
};

} //namespace ORB_SLAM

#endif // MAP_H
