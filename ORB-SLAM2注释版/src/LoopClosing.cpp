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

#include "LoopClosing.h"

#include "Sim3Solver.h"

#include "Converter.h"

#include "Optimizer.h"

#include "ORBmatcher.h"

#include<mutex>
#include<thread>


namespace ORB_SLAM2
{
//
LoopClosing::LoopClosing(Map *pMap, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale):
    mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
    mbStopGBA(false), mpThreadGBA(NULL), mbFixScale(bFixScale), mnFullBAIdx(0)
{
    mnCovisibilityConsistencyTh = 3;
}

void LoopClosing::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

//整个闭环检测运行的程序（流程）
/*1、检测是否还有需要进行闭环检测的KF（从队列中获得）
2、从队列中获得KF，为他挑选闭环匹配勾选帧
3、确定匹配之后，计算该KF的相似变换尺度S
4、修正闭环，进行图优化*/
void LoopClosing::Run()
{
    mbFinished =false;

    while(1)
    {
        // Check if there are keyframes in the queue
        // Loopclosing中的关键帧是LocalMapping发送过来的，LocalMapping是Tracking中发过来的
        // 在LocalMapping中通过InsertKeyFrame将关键帧插入闭环检测队列mlpLoopKeyFrameQueue
        // 闭环检测队列mlpLoopKeyFrameQueue中的关键帧不为空
        if(CheckNewKeyFrames())//判断是否有新关键帧到来？
        {
            // Detect loop candidates and check covisibility consistency
            if(DetectLoop())//给新来的KF找闭环候补
            {
               // Compute similarity transformation [sR|t]
               // In the stereo/RGBD case s=1
               if(ComputeSim3())//计算新来KF的相似变换尺度S
               {
                   // Perform loop fusion and pose graph optimization
                   CorrectLoop();//修正闭环，并进行图优化
               }
            }
        }

        ResetIfRequested();//在要求停止时停止

        if(CheckFinish())
            break;//跳出工作循环

        //usleep(5000);
		std::this_thread::sleep_for(std::chrono::milliseconds(5));

	}

    SetFinish();
}

void LoopClosing::InsertKeyFrame(KeyFrame *pKF)//将KF插入到闭环的KFs队列中，等待进行闭环检测
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    if(pKF->mnId!=0)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

/**
 * 查看列表中是否有等待被插入的关键帧
 * @return 如果存在，返回true
 */
bool LoopClosing::CheckNewKeyFrames()//对闭环KFs队列中的KFs一个个地进行闭环检测
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return(!mlpLoopKeyFrameQueue.empty());
}

/*闭环检测
1、*/
bool LoopClosing::DetectLoop()//检测队首KF的闭环候补KF
{
    {
        // 从队列中取出一个关键帧
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
        mlpLoopKeyFrameQueue.pop_front();
        // Avoid that a keyframe can be erased while it is being process by this thread
        mpCurrentKF->SetNotErase();//因为是用指针表示，所以要确保指针指向的对象在闭环检测的过程中不会被删除
    }

    //If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
    // 步骤1：如果距离上次闭环没多久（小于10帧），或者map中关键帧总共还没有10帧，则不进行闭环检测
    if(mpCurrentKF->mnId<mLastLoopKFid+10)
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    // Compute reference BoW similarity score
    // This is the lowest score to a connected keyframe in the covisibility graph
    // We will impose loop candidates to have a higher similarity than this
    // VIII-A
    // 步骤2：遍历所有共视关键帧，计算当前关键帧与每个共视关键的bow相似度得分，并得到最低得分minScore。通过这个最低得分去筛选闭环候补帧
    const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
    const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;
    float minScore = 1;
    for(size_t i=0; i<vpConnectedKeyFrames.size(); i++)//遍历与当前帧共视的KFs，获得与当前帧最低的相似得分（此得分作为闭环候选帧的筛选阈值）
    {
        KeyFrame* pKF = vpConnectedKeyFrames[i];
        if(pKF->isBad())
            continue;
        const DBoW2::BowVector &BowVec = pKF->mBowVec;

        float score = mpORBVocabulary->score(CurrentBowVec, BowVec);

        if(score<minScore)
            minScore = score;
    }

    // Query the database imposing the minimum score
    // 步骤3：在所有关键帧中找出闭环备选帧
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);

    // If there are no loop candidates, just add new keyframe and return false没有任何闭环候补，就把当前帧作为新的KF插入到KF数据库中
    if(vpCandidateKFs.empty())
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mvConsistentGroups.clear();
        mpCurrentKF->SetErase();
        return false;
    }

    // For each loop candidate check consistency with previous loop candidates//这个之前的闭环候选帧是不是上一次闭环的候选帧？？
    // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
    // A group is consistent with a previous group if they share at least a keyframe
    // We must detect a consistent loop in several consecutive keyframes to accept it
    // 步骤4：在候选帧中检测具有连续性的候选帧
    // 1、每个候选帧将与自己相连的关键帧构成一个“子候选组spCandidateGroup”，vpCandidateKFs-->spCandidateGroup
    // 2、检测“子候选组”中每一个关键帧是否存在于“连续组”，如果存在nCurrentConsistency++，则将该“子候选组”放入“当前连续组vCurrentConsistentGroups”
    // 3、如果nCurrentConsistency大于等于3，那么该”子候选组“代表的候选帧过关，进入mvpEnoughConsistentCandidates
    mvpEnoughConsistentCandidates.clear();// 经过一轮筛选后剩下的一些候补帧

    // ConsistentGroup数据类型为pair<set<KeyFrame*>,int>
    // ConsistentGroup.first对应每个“连续组”中的关键帧，ConsistentGroup.second为每个“连续组”的序号
    vector<ConsistentGroup> vCurrentConsistentGroups;//typedef pair<set<KeyFrame*>,int> ConsistentGroup;
    vector<bool> vbConsistentGroup(mvConsistentGroups.size(),false);
    for(size_t i=0, iend=vpCandidateKFs.size(); i<iend; i++)//遍历所有候选帧
    {
        KeyFrame* pCandidateKF = vpCandidateKFs[i];

        // 将自己以及与自己相连的关键帧构成一个“子候选组”
        set<KeyFrame*> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
        spCandidateGroup.insert(pCandidateKF);

        bool bEnoughConsistent = false;
        bool bConsistentForSomeGroup = false;
        // 遍历之前的“子连续组”
        //mvConsistentGroups应该是代表了每个候选帧与其共视的KFs构成的子连续组。但是这个连续组从哪儿来的？？？
        for(size_t iG=0, iendG=mvConsistentGroups.size(); iG<iendG; iG++)//mvConsistentGroups代表了先前的子连续组，但是这是第一个候选帧，哪儿来的先前子连续组？？
        {
            // 取出一个之前的子连续组
            set<KeyFrame*> sPreviousGroup = mvConsistentGroups[iG].first;

            // 遍历每个“子候选组”，检测候选组中每一个关键帧在“子连续组”中是否存在
            // 如果有一帧共同存在于“子候选组”与之前的“子连续组”，那么“子候选组”与该“子连续组”连续
            bool bConsistent = false;
            for(set<KeyFrame*>::iterator sit=spCandidateGroup.begin(), send=spCandidateGroup.end(); sit!=send;sit++)
            {
                if(sPreviousGroup.count(*sit))
                {
                    bConsistent=true;// 该“子候选组”与该“子连续组”相连
                    bConsistentForSomeGroup=true;// 该“子候选组”至少与一个”子连续组“相连
                    break;
                }
            }

            if(bConsistent)
            {
                int nPreviousConsistency = mvConsistentGroups[iG].second;
                int nCurrentConsistency = nPreviousConsistency + 1;
                if(!vbConsistentGroup[iG])// 这里作者原本意思是不是应该是vbConsistentGroup[i]而不是vbConsistentGroup[iG]呢？（wubo???）
                {
                    // 将该“子候选组”的该关键帧打上编号加入到“当前连续组”
                    ConsistentGroup cg = make_pair(spCandidateGroup,nCurrentConsistency);
                    vCurrentConsistentGroups.push_back(cg);
                    vbConsistentGroup[iG]=true; //this avoid to include the same group more than once
                }
                if(nCurrentConsistency>=mnCovisibilityConsistencyTh && !bEnoughConsistent)
                {
                    mvpEnoughConsistentCandidates.push_back(pCandidateKF);
                    bEnoughConsistent=true; //this avoid to insert the same candidate more than once
                }

                //这里是不是缺一个break来提高效率呢？(wubo???)
            }
        }

        // If the group is not consistent with any previous group insert with consistency counter set to zero
        // 如果该“子候选组”的所有关键帧都不存在于“子连续组”，那么vCurrentConsistentGroups将为空，
        // 于是就把“子候选组”全部拷贝到vCurrentConsistentGroups，并最终用于更新mvConsistentGroups，计数器设为0，重新开始
        if(!bConsistentForSomeGroup)
        {
            // 这个地方是不是最好clear一下vCurrentConsistentGroups呢？(wubo???)

            ConsistentGroup cg = make_pair(spCandidateGroup,0);
            vCurrentConsistentGroups.push_back(cg);
        }
    }

    // Update Covisibility Consistent Groups
    mvConsistentGroups = vCurrentConsistentGroups;


    // Add Current Keyframe to database
    mpKeyFrameDB->add(mpCurrentKF);

    if(mvpEnoughConsistentCandidates.empty())
    {
        mpCurrentKF->SetErase();
        return false;
    }
    else
    {
        return true;
    }

    mpCurrentKF->SetErase();
    return false;
}

/**
 * @brief 计算当前帧与闭环帧的Sim3变换等
 *
 * 1. 通过Bow加速描述子的匹配，利用RANSAC粗略地计算出当前帧与闭环帧的Sim3（当前帧---闭环帧）
 * 2. 根据估计的Sim3，对3D点进行投影找到更多匹配，通过优化的方法计算更精确的Sim3（当前帧---闭环帧）
 * 3. 将闭环帧以及闭环帧相连的关键帧的MapPoints与当前帧的点进行匹配（当前帧---闭环帧+相连关键帧）
 * 
 * 注意以上匹配的结果均都存在成员变量mvpCurrentMatchedPoints中，
 * 实际的更新步骤见CorrectLoop()步骤3：Start Loop Fusion
 */
/*函数步骤：
1、将当前KF与mvpEnoughConsistentCandidates容器中的候选KFs逐个进行地图点匹配，并为每个候选帧都设置一个求解Sim3的求解器
2、通过匹配好的地图点，计算当前KF与每个候选帧的Sim3变换
3、Sim3变换最初是通过RANSAC求解，对整个候选帧队列进行循环求解，每次循环每个KF迭代5次求Sim3。
4、这样持续循环，直到某个候选帧求出了Sim3变换或者是每个候选帧都达到了迭代上限
5、将求出Sim3变换的KF（LKF），通过Sim3变换，获得更多与当前帧匹配的地图点。再进行Sim3变换的优化
6、优化后的内点数目达标后，认为LKF是当前帧的最终匹配。
7、获得LKF的共视KFs，提取它们所包含的所有地图点。将地图点通过上面求得的Sim3变换投影到当前KF，寻找匹配点。
8、只有当匹配上的点对的数量足够多时，才算是真正求出了Sim3变换*/
bool LoopClosing::ComputeSim3()
{
    // For each consistent loop candidate we try to compute a Sim3

    const int nInitialCandidates = mvpEnoughConsistentCandidates.size();

    // We compute first ORB matches for each candidate
    // If enough matches are found, we setup a Sim3Solver
    ORBmatcher matcher(0.75,true);

    vector<Sim3Solver*> vpSim3Solvers;
    vpSim3Solvers.resize(nInitialCandidates);// 每个候选帧都有一个Sim3Solver

    vector<vector<MapPoint*> > vvpMapPointMatches;//每个候选帧所包含的地图点
    vvpMapPointMatches.resize(nInitialCandidates);

    vector<bool> vbDiscarded;//某个候选帧是否要被剔除的标志
    vbDiscarded.resize(nInitialCandidates);

    int nCandidates=0; //candidates with enough matches

    for(int i=0; i<nInitialCandidates; i++)
    {
        // 步骤1：从筛选的闭环候选帧中取出一帧关键帧pKF
        KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

        // avoid that local mapping erase it while it is being processed in this thread
        // 防止在LocalMapping中KeyFrameCulling函数将此关键帧作为冗余帧剔除
        pKF->SetNotErase();

        if(pKF->isBad())
        {
            vbDiscarded[i] = true;// 直接将该候选帧舍弃
            continue;
        }

        // 步骤2：将当前帧mpCurrentKF与闭环候选关键帧pKF匹配
        // 通过bow加速得到mpCurrentKF与pKF之间的匹配特征点，vvpMapPointMatches是匹配特征点对应的MapPoints
        int nmatches = matcher.SearchByBoW(mpCurrentKF,pKF,vvpMapPointMatches[i]);//第三个参数用于保存两个KFs匹配的地图点

        // 匹配的特征点数太少，该候选帧剔除
        if(nmatches<20)
        {
            vbDiscarded[i] = true;
            continue;
        }
        else
        {
            // if bFixScale is true, 6DoF optimization (stereo,rgbd), 7DoF otherwise (mono)
            // 构造Sim3求解器
            // 如果mbFixScale为true，则是6DoFf优化（双目 RGBD），如果是false，则是7DoF优化（单目）
            Sim3Solver* pSolver = new Sim3Solver(mpCurrentKF,pKF,vvpMapPointMatches[i],mbFixScale);//通过两个KFs和包含匹配地图点信息的容器，计算出两帧间的相似变换
            pSolver->SetRansacParameters(0.99,20,300);// 至少20个inliers 300次迭代（设置RANSAC参数）
            vpSim3Solvers[i] = pSolver;
        }

        // 参与Sim3计算的候选关键帧数加1
        nCandidates++;
    }

    bool bMatch = false;// 用于标记是否有一个候选帧通过Sim3的求解与优化

    // Perform alternatively RANSAC iterations for each candidate
    // until one is succesful or all fail
    // 一直循环所有的候选帧，每个候选帧迭代5次，如果5次迭代后得不到结果，就换下一个候选帧
    // 直到有一个候选帧首次迭代成功bMatch为true，或者某个候选帧总的迭代次数超过限制，直接将它剔除
    while(nCandidates>0 && !bMatch)//迭代求解当前帧与每个候选帧间的相似变换
    {
        for(int i=0; i<nInitialCandidates; i++)
        {
            if(vbDiscarded[i])
                continue;

            KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;//标注某个匹配地图点对是否是计算出来的Sim3变换的内点
            int nInliers;
            bool bNoMore;// 这是局部变量，在pSolver->iterate(...)内进行初始化

            // 步骤3：对步骤2中有较好的匹配的关键帧求取Sim3变换
            Sim3Solver* pSolver = vpSim3Solvers[i];
            // 最多迭代5次，返回的Scm是候选帧pKF到当前帧mpCurrentKF的Sim3变换（T12）
            cv::Mat Scm  = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            // 经过n次循环，每次迭代5次，总共迭代 n*5 次
            // 总迭代次数达到最大限制还没有求出合格的Sim3变换，该候选帧剔除
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
            if(!Scm.empty())
            {
                vector<MapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint*>(NULL));
                for(size_t j=0, jend=vbInliers.size(); j<jend; j++)
                {
                    // 保存inlier的MapPoint
                    if(vbInliers[j])
                       vpMapPointMatches[j]=vvpMapPointMatches[i][j];
                }

                // 步骤4：通过步骤3求取的Sim3变换引导关键帧匹配弥补步骤2中的漏匹配
                // [sR t;0 1]
                cv::Mat R = pSolver->GetEstimatedRotation();// 候选帧pKF到当前帧mpCurrentKF的R（R12）
                cv::Mat t = pSolver->GetEstimatedTranslation();// 候选帧pKF到当前帧mpCurrentKF的t（t12），当前帧坐标系下，方向由pKF指向当前帧
                const float s = pSolver->GetEstimatedScale();// 候选帧pKF到当前帧mpCurrentKF的变换尺度s（s12）
                // 查找更多的匹配（成功的闭环匹配需要满足足够多的匹配特征点数，之前使用SearchByBoW进行特征点匹配时会有漏匹配）
                // 通过Sim3变换，确定pKF1的特征点在pKF2中的大致区域，同理，确定pKF2的特征点在pKF1中的大致区域
                // 在该区域内通过描述子进行匹配捕获pKF1和pKF2之前漏匹配的特征点，更新匹配vpMapPointMatches
                matcher.SearchBySim3(mpCurrentKF,pKF,vpMapPointMatches,s,R,t,7.5);//增加在当前相似变换的前提下所包含的匹配点

                // 步骤5：Sim3优化，只要有一个候选帧通过Sim3的求解与优化，就跳出停止对其它候选帧的判断
                // OpenCV的Mat矩阵转成Eigen的Matrix类型
                g2o::Sim3 gScm(Converter::toMatrix3d(R),Converter::toVector3d(t),s);
                // 如果mbFixScale为true，则是6DoFf优化（双目 RGBD），如果是false，则是7DoF优化（单目）
                // 优化mpCurrentKF与pKF对应的MapPoints间的Sim3，得到优化后的量gScm
                const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale);// 卡方chi2检验阈值

                // If optimization is succesful stop ransacs and continue
                if(nInliers>=20)
                {
                    bMatch = true;
                    // mpMatchedKF就是最终闭环检测出来与当前帧形成闭环的关键帧
                    mpMatchedKF = pKF;
                    // 得到从世界坐标系到该候选帧的Sim3变换，Scale=1
                    g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()),Converter::toVector3d(pKF->GetTranslation()),1.0);
                    // 得到g2o优化后从世界坐标系到当前帧的Sim3变换
                    mg2oScw = gScm*gSmw;
                    mScw = Converter::toCvMat(mg2oScw);

                    mvpCurrentMatchedPoints = vpMapPointMatches;
                    break;// 只要有一个候选帧通过Sim3的求解与优化，就跳出停止对其它候选帧的判断
                }
            }
        }
    }

    // 没有一个闭环匹配候选帧通过Sim3的求解与优化
    if(!bMatch)
    {
        // 清空mvpEnoughConsistentCandidates
        for(int i=0; i<nInitialCandidates; i++)
             mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }

    // Retrieve MapPoints seen in Loop Keyframe and neighbors
    // 步骤6：取出闭环匹配上关键帧的相连关键帧，得到它们的MapPoints放入mvpLoopMapPoints
    // 注意是匹配上的那个关键帧：mpMatchedKF
    // 将mpMatchedKF相连的关键帧全部取出来放入vpLoopConnectedKFs
    // 将vpLoopConnectedKFs的MapPoints取出来放入mvpLoopMapPoints
    vector<KeyFrame*> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();
    // 包含闭环匹配关键帧本身
    vpLoopConnectedKFs.push_back(mpMatchedKF);
    mvpLoopMapPoints.clear();
    for(vector<KeyFrame*>::iterator vit=vpLoopConnectedKFs.begin(); vit!=vpLoopConnectedKFs.end(); vit++)
    {
        KeyFrame* pKF = *vit;
        vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad() && pMP->mnLoopPointForKF!=mpCurrentKF->mnId)// 标记该MapPoint被mpCurrentKF闭环时观测到并添加，避免重复添加
                {
                    mvpLoopMapPoints.push_back(pMP);
                    
                    pMP->mnLoopPointForKF=mpCurrentKF->mnId;
                }
            }
        }
    }

    // Find more matches projecting with the computed Sim3
    // 步骤7：将闭环匹配上关键帧以及相连关键帧的MapPoints投影到当前关键帧进行投影匹配
    // 根据投影查找更多的匹配（成功的闭环匹配需要满足足够多的匹配特征点数）
    // 根据Sim3变换，将每个mvpLoopMapPoints投影到mpCurrentKF上，并根据尺度确定一个搜索区域，
    // 根据该MapPoint的描述子与该区域内的特征点进行匹配，如果匹配误差小于TH_LOW即匹配成功，更新mvpCurrentMatchedPoints
    // mvpCurrentMatchedPoints将用于SearchAndFuse中检测当前帧MapPoints与匹配的MapPoints是否存在冲突
    matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints,10);// 搜索范围系数为10

    // If enough matches accept Loop
    // 步骤8：判断当前帧与检测出的所有闭环关键帧是否有足够多的MapPoints匹配
    int nTotalMatches = 0;
    for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
    {
        if(mvpCurrentMatchedPoints[i])
            nTotalMatches++;
    }

    // 步骤9：清空mvpEnoughConsistentCandidates，只保留与当前帧匹配上的那个KF
    if(nTotalMatches>=40)
    {
        for(int i=0; i<nInitialCandidates; i++)
            if(mvpEnoughConsistentCandidates[i]!=mpMatchedKF)
                mvpEnoughConsistentCandidates[i]->SetErase();
        return true;
    }
    else//如果当前帧与检测出的所有闭环关键帧没有足够多的地图点匹配，那么就认为这个闭环检测失败
    {
        for(int i=0; i<nInitialCandidates; i++)
            mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }
}

/**
 * @brief 闭环
 *
 * 1. 通过求解的Sim3以及相对姿态关系，调整与当前帧相连的关键帧位姿以及这些关键帧观测到的MapPoints的位置（相连关键帧---当前帧）
 * 2. 将闭环帧以及闭环帧相连的关键帧的MapPoints和与当前帧相连的关键帧的点进行匹配（相连关键帧+当前帧---闭环帧+相连关键帧）
 * 3. 通过MapPoints的匹配关系更新这些帧之间的连接关系，即更新covisibility graph
 * 4. 对Essential Graph（Pose Graph）进行优化，MapPoints的位置则根据优化后的位姿做相对应的调整
 * 5. 创建线程进行全局Bundle Adjustment
 */
/**/
void LoopClosing::CorrectLoop()
{
    cout << "Loop detected!" << endl;

    // Send a stop signal to Local Mapping
    // Avoid new keyframes are inserted while correcting the loop
    // 步骤0：请求局部地图停止，防止局部地图线程中InsertKeyFrame函数插入新的关键帧
    mpLocalMapper->RequestStop();

    // If a Global Bundle Adjustment is running, abort it停止全局BA
    if(isRunningGBA())
    {
        // 这个标志位仅用于控制输出提示，可忽略
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx++;

        if(mpThreadGBA)
        {
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
    }

    // Wait until Local Mapping has effectively stopped
    while(!mpLocalMapper->isStopped())
    {
        //usleep(1000);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Ensure current keyframe is updated
    // 步骤1：根据共视关系更新当前帧与其它关键帧之间的连接
    mpCurrentKF->UpdateConnections();

    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    // 步骤2：通过位姿传播，得到Sim3优化后，与当前帧相连的关键帧的位姿，以及它们的MapPoints
    // 当前帧与世界坐标系之间的Sim变换在ComputeSim3函数中已经确定并优化，
    // 通过相对位姿关系（其余KFs与当前帧的相对位姿），可以确定这些相连的关键帧与世界坐标系之间的Sim3变换

    // 取出与当前帧相连的关键帧，包括当前关键帧
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);

    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;//容器。下标是KF指针，元素是KF的相似变换
    // 先将mpCurrentKF的Sim3变换存入，固定不动
    CorrectedSim3[mpCurrentKF]=mg2oScw;
    cv::Mat Twc = mpCurrentKF->GetPoseInverse();

    //下面的程序：
    {
        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

        // 步骤2.1：通过位姿传播，得到Sim3调整后其它与当前帧相连关键帧的位姿（只是得到，还没有修正）
        for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKFi = *vit;

            cv::Mat Tiw = pKFi->GetPose();//世界-->相机的变换

            // currentKF在前面已经添加
            if(pKFi!=mpCurrentKF)
            {
                // 得到当前帧到pKFi帧的相对变换（不带相似变换的）
                cv::Mat Tic = Tiw*Twc;
                cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);
                cv::Mat tic = Tic.rowRange(0,3).col(3);
                g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric),Converter::toVector3d(tic),1.0);//将相对变换转换成相似变换（S=1）
                // 当前帧的位姿固定不动，其它的关键帧根据相对关系得到Sim3调整的位姿
                g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oScw;//简单调整后的各KFs的相似变换（世界->相机）
                // Pose corrected with the Sim3 of the loop closure
                // 得到闭环g2o优化后各个关键帧的位姿
                CorrectedSim3[pKFi]=g2oCorrectedSiw;//保存
            }

            cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
            cv::Mat tiw = Tiw.rowRange(0,3).col(3);
            g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);//直接将普通变换设置成KF的相似变换
            // Pose without correction
            // 当前帧相连关键帧，没有进行闭环g2o优化的位姿
            NonCorrectedSim3[pKFi]=g2oSiw;//其实还是原来的变换（尺度S=1）
            //NonCorrectedSim3保存到内容和CorrectedSim3类似，只是保存的是普通的变换矩阵
        }

        // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        // 步骤2.2：步骤2.1得到调整相连帧位姿后（与当前KF相连的KFs的位姿被调整成了相似变换的位姿），修正这些关键帧的MapPoints
        //遍历与当前KF相连的KFs以及它们的相似变换。将KFs中所包含的地图点都通过反相似变换转换成新的地图点坐标
        for(KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)//CorrectedSim3容器保存元素是pair。pair由KeyFrame*和Sim3组成
        {
            KeyFrame* pKFi = mit->first;
            g2o::Sim3 g2oCorrectedSiw = mit->second;
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

            g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi];

            vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
            for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)//修正KF中的地图点坐标
            {
                MapPoint* pMPi = vpMPsi[iMP];
                if(!pMPi)
                    continue;
                if(pMPi->isBad())
                    continue;
                if(pMPi->mnCorrectedByKF==mpCurrentKF->mnId) // 防止重复修正
                    continue;

                // Project with non-corrected pose and project back with corrected pose
                // 将该未校正的eigP3Dw先从世界坐标系映射到未校正的pKFi相机坐标系，然后再通过修正后的相似变换反映射到校正后的世界坐标系下
                cv::Mat P3Dw = pMPi->GetWorldPos();
                Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
                Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));//获得修正后的世界坐标

                cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                pMPi->SetWorldPos(cvCorrectedP3Dw);//更新地图点坐标
                //下面两行是为了防止对同一地图点进行重复更新
                pMPi->mnCorrectedByKF = mpCurrentKF->mnId;//该地图点是由当前KF修正的
                pMPi->mnCorrectedReference = pKFi->mnId;//该地图点是以pKFi作为参考KF（由这个KF创造的这个地图点）
                pMPi->UpdateNormalAndDepth();
            }

            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
            // 步骤2.3：将Sim3转换为SE3，根据更新的Sim3，更新关键帧的位姿
            //更新的KF位姿使用的仍是普通的变换矩阵形式（它将之前求得的相似变换矩阵除去了变换尺度S）
            Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
            Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
            double s = g2oCorrectedSiw.scale();

            eigt *=(1./s); //[R t/s;0 1]

            cv::Mat correctedTiw = Converter::toCvSE3(eigR,eigt);

            pKFi->SetPose(correctedTiw);

            // Make sure connections are updated
            // 步骤2.4：根据共视关系更新当前帧与其它关键帧之间的连接
            pKFi->UpdateConnections();
        }
        //上述代码通过当前KF的相似变换位姿对所有与之相连的KFs的位姿进行了调整，同时也对它们所包含的所有地图点也进行了调整

        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        // 步骤3：检查当前帧的MapPoints与闭环匹配帧的MapPoints是否存在冲突，对冲突的MapPoints进行替换或填补
        //将当前帧的特征点所对应的地图点都和LoopMapoint中的地图点对应上
        for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)//mvpCurrentMatchedPoints容器下标：当前KF的特征点；值：对应的地图点
        {
            if(mvpCurrentMatchedPoints[i])
            {
                MapPoint* pLoopMP = mvpCurrentMatchedPoints[i];//集体（相连的KFs看到的地图点）来代替个人（当前帧看到的地图点）
                MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);//获得当前KF中的某个地图点
                if(pCurMP)// 如果有重复的MapPoint（当前帧和匹配帧各有一个），则用匹配帧的代替现有的
                    pCurMP->Replace(pLoopMP);
                else// 如果当前帧没有该MapPoint，则直接添加
                {
                    mpCurrentKF->AddMapPoint(pLoopMP,i);
                    pLoopMP->AddObservation(mpCurrentKF,i);
                    pLoopMP->ComputeDistinctiveDescriptors();
                }
            }
        }//这个for循环用闭环中包含的地图点替换掉了当前帧中原来所对应的地图点

    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    // 步骤4：通过将闭环时相连关键帧的mvpLoopMapPoints投影到这些关键帧（当前KF以及与它相连的KFs）中，进行MapPoints检查与替换
    SearchAndFuse(CorrectedSim3);
    //由于是将闭环中的地图点投影到其他KFs中，所以会增加KFs观测到的地图点，因此会产生新的共视连接关系


    // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
    // 步骤5：更新当前关键帧之间的共视相连关系，得到因闭环时MapPoints融合而新得到的连接关系
    map<KeyFrame*, set<KeyFrame*> > LoopConnections;

    // 步骤5.1：遍历当前帧相连关键帧（一级相连）
    for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;//pKFi是与当前KF有共视的KF（一级）
        // 步骤5.2：得到与当前帧相连的关键帧的相连关键帧（二级相连）
        vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();//vpPreviousNeighbors容器保存与pKFi有共视的KFs（二级）

        // Update connections. Detect new links.
        // 步骤5.3：更新一级相连关键帧的连接关系
        pKFi->UpdateConnections();//更新与pKFi有共视的KFs与pKFi间的连接
        // 步骤5.4：取出该帧更新后的连接关系
        LoopConnections[pKFi]=pKFi->GetConnectedKeyFrames();
        // 步骤5.5：从连接关系中去除闭环之前的二级连接关系，剩下的连接就是由闭环得到的连接关系
        //将更新后的连接的共视图中的原来的连接剔除掉，剩下的就是通过闭环增加的连接
        for(vector<KeyFrame*>::iterator vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
        {
            LoopConnections[pKFi].erase(*vit_prev);//将新获得的pKFi共视KFs连接中存在的vpPreviousNeighbors容器的元素去除掉（去除掉原二级连接中已经包含过的KFs）
        }
        // 步骤5.6：从连接关系中去除闭环之前的一级连接关系，剩下的连接就是由闭环得到的连接关系
        for(vector<KeyFrame*>::iterator vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
        {
            LoopConnections[pKFi].erase(*vit2);//去除掉一级连接中的KFs）
        }
    }//该循环为了获得由于闭环而产生的与当前KF有共视关系的新的连接边，之后可用于位姿优化

    // Optimize graph
    // 步骤6：进行EssentialGraph优化，LoopConnections是形成闭环后新生成的连接关系，不包括步骤7中当前帧与闭环匹配帧之间的连接关系
    Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);
    //该函数的操作：首先通过获得共视图的函数，将mpCurrentKF帧的共视帧获得，并为他们构造g2o顶点。再为闭环中每个地图点都构建顶点。再通过LoopConnections为图添加新的边。
    //以此构建一个完整的图优化
    //优化的最后，会将根据结果重新设置每个KF的位姿以及地图点坐标

    // Add loop edge
    // 步骤7：添加当前帧与闭环匹配帧之间的边（这个连接关系不优化）
    // 这两句话应该放在OptimizeEssentialGraph之前，因为OptimizeEssentialGraph的步骤4.2中有优化，（wubo???）
    mpMatchedKF->AddLoopEdge(mpCurrentKF);
    mpCurrentKF->AddLoopEdge(mpMatchedKF);

    // Launch a new thread to perform Global Bundle Adjustment
    // 步骤8：新建一个线程用于全局BA优化
    // OptimizeEssentialGraph只是优化了一些主要关键帧的位姿，这里进行全局BA可以全局优化所有位姿和MapPoints
    mbRunningGBA = true;
    mbFinishedGBA = false;
    mbStopGBA = false;
    mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment,this,mpCurrentKF->mnId);

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();    

    cout << "Loop Closed!" << endl;

    mLastLoopKFid = mpCurrentKF->mnId;
}

// 通过将闭环时相连关键帧的MapPoints投影到这些关键帧中，进行MapPoints检查与替换
//将闭环中的地图点投影到闭环中所有的KFs中，增加KFs中的地图点，以及用闭环中的地图点替换掉一些原来的地图点
void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)//传递的参数是闭环关联的KFs和它们对应的相似变换（组成的pair）
{
    ORBmatcher matcher(0.8);

    // 遍历闭环相连的关键帧
    for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
    {
        KeyFrame* pKF = mit->first;

        g2o::Sim3 g2oScw = mit->second;
        cv::Mat cvScw = Converter::toCvMat(g2oScw);

        // 将闭环相连帧的MapPoints坐标变换到pKF帧坐标系，然后投影，检查冲突并融合
        vector<MapPoint*> vpReplacePoints(mvpLoopMapPoints.size(),static_cast<MapPoint*>(NULL));
        matcher.Fuse(pKF,cvScw,mvpLoopMapPoints,4,vpReplacePoints);// 搜索区域系数为4
        //最后一个容器参数的下标：pKF特征点索引；值：指向需要进行替换的地图点的指针。指针为NULL时则说明不要进行替换

        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        const int nLP = mvpLoopMapPoints.size();
        for(int i=0; i<nLP;i++)
        {
            MapPoint* pRep = vpReplacePoints[i];
            if(pRep)//看pKF中第i个特征点对应的地图点是否需要替换
            {
                pRep->Replace(mvpLoopMapPoints[i]);// 用mvpLoopMapPoints替换掉之前的
            }
        }
    }
}

//闭环系统的重置
void LoopClosing::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)//直到成功重置才结束
    {
        {
        unique_lock<mutex> lock2(mMutexReset);
        if(!mbResetRequested)
            break;
        }
		//usleep(5000);
		std::this_thread::sleep_for(std::chrono::milliseconds(5));

    }
}

//根据要求将整个闭环检测对象格式化，清空等候检测的KF队列
void LoopClosing::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlpLoopKeyFrameQueue.clear();
        mLastLoopKFid=0;
        mbResetRequested=false;
    }
}

/*1、获取地图中所有的地图和KFs，更新他们的位姿或者坐标
2、对于参加了之前的全局BA的KFs和地图点，那么直接就能够更新他们的值
3、否则就需要使用一些间接方法来进行更新*/
void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
{
    cout << "Starting Global Bundle Adjustment" << endl;

    int idx =  mnFullBAIdx;
    Optimizer::GlobalBundleAdjustemnt(mpMap,10,&mbStopGBA,nLoopKF,false);

    // Update all MapPoints and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    {
        unique_lock<mutex> lock(mMutexGBA);
        if(idx!=mnFullBAIdx)
            return;

        if(!mbStopGBA)
        {
            cout << "Global Bundle Adjustment finished" << endl;
            cout << "Updating map ..." << endl;
            mpLocalMapper->RequestStop();
            // Wait until Local Mapping has effectively stopped

            while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
            {
				//usleep(1000);
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}

            // Get Map Mutex
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

            // Correct keyframes starting at map first keyframe
            list<KeyFrame*> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(),mpMap->mvpKeyFrameOrigins.end());//这就像是一个队列一样。先对队首的KF进行处理，并将每个KF的
            //子KFs放到队末，之后再处理

            while(!lpKFtoCheck.empty())//对地图中所有KFs进行处理，用BA优化后的位姿去更新他们的位姿
            {
                KeyFrame* pKF = lpKFtoCheck.front();
                const set<KeyFrame*> sChilds = pKF->GetChilds();//获得关键帧的子关键帧
                cv::Mat Twc = pKF->GetPoseInverse();
                for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)//遍历其子关键帧
                {
                    KeyFrame* pChild = *sit;
                    if(pChild->mnBAGlobalForKF!=nLoopKF)//判断该子KF是否参加了nLoopKF的全局BA。如果没参加，就通过相对位姿来获得它的全局BA位姿
                    {
                        cv::Mat Tchildc = pChild->GetPose()*Twc;//当前关键帧到子关键帧的相对变换
                        pChild->mTcwGBA = Tchildc*pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;通过相对变换，由该KF获得子KF经过优化后的位姿（TGBA）
                        pChild->mnBAGlobalForKF=nLoopKF;

                    }
                    lpKFtoCheck.push_back(pChild);
                }

                pKF->mTcwBefGBA = pKF->GetPose();
                pKF->SetPose(pKF->mTcwGBA);
                lpKFtoCheck.pop_front();
            }

            // Correct MapPoints
            const vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();

            for(size_t i=0; i<vpMPs.size(); i++)
            {
                MapPoint* pMP = vpMPs[i];

                if(pMP->isBad())
                    continue;

                if(pMP->mnBAGlobalForKF==nLoopKF)//和上面情况一样
                {
                    // If optimized by Global BA, just update
                    pMP->SetWorldPos(pMP->mPosGBA);
                }
                else
                {
                    // Update according to the correction of its reference keyframe
                    //如果地图点没参加之前的全局优化，那么就考虑使用它的参考KF的位姿来获得其间接BA优化的坐标
                    KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

                    if(pRefKF->mnBAGlobalForKF!=nLoopKF)//如果它的参考KF仍未参加之前的全局优化，那么就没招了
                        continue;

                    // Map to non-corrected camera
                    cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);
                    cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);
                    cv::Mat Xc = Rcw*pMP->GetWorldPos()+tcw;

                    // Backproject using corrected camera
                    cv::Mat Twc = pRefKF->GetPoseInverse();
                    cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
                    cv::Mat twc = Twc.rowRange(0,3).col(3);

                    pMP->SetWorldPos(Rwc*Xc+twc);
                }
            }

            mpLocalMapper->Release();

            cout << "Map updated!" << endl;
        }

        mbFinishedGBA = true;
        mbRunningGBA = false;
    }
}

void LoopClosing::RequestFinish()//要求LoopClosing停止
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LoopClosing::CheckFinish()//是否要求程序停止
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LoopClosing::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool LoopClosing::isFinished()//程序停止与否的情况
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}


} //namespace ORB_SLAM
