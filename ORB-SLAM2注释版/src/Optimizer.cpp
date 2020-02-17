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

#include "Optimizer.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include<Eigen/StdVector>

#include "Converter.h"

#include<mutex>

namespace ORB_SLAM2
{

// pMap中所有的MapPoints和关键帧做bundle adjustment优化
// 这个全局BA优化在本程序中有两个地方使用：
// a.单目初始化：CreateInitialMapMonocular函数
// b.闭环优化：RunGlobalBundleAdjustment函数
void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
    BundleAdjustment(vpKFs,vpMP,nIterations,pbStopFlag, nLoopKF, bRobust);
}

/**
 * @brief bundle adjustment Optimization
 * 
 * 3D-2D 最小化重投影误差 e = (u,v) - project(Tcw*Pw) \n
 * 
 * 1. Vertex: g2o::VertexSE3Expmap()，即当前帧的Tcw
 *            g2o::VertexSBAPointXYZ()，MapPoint的mWorldPos
 * 2. Edge:
 *     - g2o::EdgeSE3ProjectXYZ()，BaseBinaryEdge
 *         + Vertex：待优化当前帧的Tcw
 *         + Vertex：待优化MapPoint的mWorldPos
 *         + measurement：MapPoint在当前帧中的二维位置(u,v)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *         
 * @param   vpKFs    关键帧 
 *          vpMP     MapPoints
 *          nIterations 迭代次数（20次）
 *          pbStopFlag  是否强制暂停
 *          nLoopKF  关键帧的个数
 *          bRobust  是否使用核函数
 */
/*优化一大堆帧和地图点
1、配置优化器
2、向优化器添加KF和地图点的顶点
3、通过地图点包含的观测帧容器，建立各个地图点与添加的KF顶点的边
4、将不能建立边的地图点顶点从优化图中剔除
5、进行优化，并获得结果*/
void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    // 步骤1：初始化g2o优化器
    g2o::SparseOptimizer optimizer;//稀疏优化器
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();//设置优化器的线性求解类型

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);//根据优化器的线性求解类型构造线性求解指针

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);//由求解指针设置列文伯格求导方法（优化算法）
    optimizer.setAlgorithm(solver);//初始化优化器的求解算法

    if(pbStopFlag)//是否强制暂停
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;//优化过程中最大的KF编号，方便之后给地图点设置编号（因为优化器顶点编号是连续的，不论是什么类型的顶点）

    // 步骤2：向优化器添加顶点

    // Set KeyFrame vertices
    // 步骤2.1：向优化器添加关键帧位姿顶点
    for(size_t i=0; i<vpKFs.size(); i++)//遍历关键帧
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));//求解器是使用四元数来表示变换矩阵的，故将关键帧的位姿转换成四元数的形式来构成位姿顶点
        vSE3->setId(pKF->mnId);//顶点的编号和对应KF的编号相同
        vSE3->setFixed(pKF->mnId==0);//该顶点在优化过程中是否是固定不动的（初始KF）
        optimizer.addVertex(vSE3);
        if(pKF->mnId>maxKFid)
            maxKFid=pKF->mnId;
    }

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    // Set MapPoint vertices
    // 步骤2.2：向优化器添加MapPoints顶点
    for(size_t i=0; i<vpMP.size(); i++)
    {
        MapPoint* pMP = vpMP[i];
        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));//设置地图点顶点的估计值
        const int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        int nEdges = 0;
        //SET EDGES//通过该地图点的观测帧容器，构造地图点顶点和其观测帧顶点的边
        // 步骤3：向优化器添加投影边边
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {

            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId>maxKFid)//第二项：该观测帧不再优化范围内
                continue;

            nEdges++;

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];

            //单目/RGB-D和双目的边是不同的！！！！！
            // 单目或RGBD相机
            if(pKF->mvuRight[mit->second]<0)//？？RGB-D相机的mvuRight是通过计算获得的，但为什么一定小于0？？
            {
                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));//与地图点的顶点相连
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));//与帧相连
                e->setMeasurement(obs);//设置地图点在帧上的像素坐标
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);//？？

                if(bRobust)//是否使用核函数：防止出现一个边的误差巨大，导致优化器一直优化这条边而忽视了其他边
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;

                optimizer.addEdge(e);
            }
            else// 双目相机：有两张图像，所以mvuRight的元素肯定为正
            {
                Eigen::Matrix<double,3,1> obs;
                const float kp_ur = pKF->mvuRight[mit->second];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);
            }
        }

        if(nEdges==0)//将不能建立边的点剔除
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;
        }
        else
        {
            vbNotIncludedMP[i]=false;
        }
    }

    // Optimize!
    // 步骤4：开始优化
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // Recover optimized data
    // 步骤5：得到优化的结果

    //Keyframes
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();//获得KF顶点的优化后位姿
        if(nLoopKF==0)//？？
        {
            pKF->SetPose(Converter::toCvMat(SE3quat));
        }
        else
        {
            pKF->mTcwGBA.create(4,4,CV_32F);
            Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
            pKF->mnBAGlobalForKF = nLoopKF;
        }
    }

    //Points
    for(size_t i=0; i<vpMP.size(); i++)
    {
        if(vbNotIncludedMP[i])//不考虑没有参加优化的点
            continue;

        MapPoint* pMP = vpMP[i];

        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));

        //设置地图点优化后坐标，并更新其平均观测方向以及观测距离
        if(nLoopKF==0)
        {
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }
        else
        {
            pMP->mPosGBA.create(3,1,CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }

}

/**
 * @brief Pose Only Optimization
 * 
 * 3D-2D 最小化重投影误差 e = (u,v) - project(Tcw*Pw) \n
 * 只优化Frame的Tcw，不优化MapPoints的坐标
 * 
 * 1. Vertex: g2o::VertexSE3Expmap()，即当前帧的Tcw
 * 2. Edge:
 *     - g2o::EdgeSE3ProjectXYZOnlyPose()，BaseUnaryEdge
 *         + Vertex：待优化当前帧的Tcw
 *         + measurement：MapPoint在当前帧中的二维位置(u,v)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *     - g2o::EdgeStereoSE3ProjectXYZOnlyPose()，BaseUnaryEdge
 *         + Vertex：待优化当前帧的Tcw
 *         + measurement：MapPoint在当前帧中的二维位置(ul,v,ur)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *
 * @param   pFrame Frame
 * @return  inliers数量
 */
//仅仅优化pFrame的位姿
/*1、初始化优化器
2、向优化器中添加当前帧位姿顶点
3、根据当前帧所包含的地图点，构建一元边，仅优化当前帧位姿
4、进行四次优化，每次迭代10次
5、在每次四次优化的过程中，不断地进行内、外点判断，不断地更改参加优化的边
6、获得最终优化结果，以及内点数量*/
int Optimizer::PoseOptimization(Frame *pFrame)
{
    // 该优化函数主要用于Tracking线程中：运动跟踪、参考帧跟踪、地图跟踪、重定位

    // 步骤1：构造g2o优化器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;//构造了多少个边

    // Set Frame vertex
    // 步骤2：添加顶点：待优化当前帧的Tcw（就一个）
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;

    // for Monocular
    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    // for Stereo
    //两个容器元素的位置是一一对应的
    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;//保存有哪些边
    vector<size_t> vnIndexEdgeStereo;//保存边所对应当前帧中的哪个地图点
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);

    // 步骤3：添加一元边：相机投影模型
    {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);

    //将帧中的所有地图点用于位姿优化。其中优化图中的边有的是单目/RGB-D边，有的是双目边
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = pFrame->mvpMapPoints[i];
        if(pMP)
        {
            // Monocular observation
            // 单目情况, 也有可能在双目下, 当前帧的左兴趣点找不到匹配的右兴趣点
            if(pFrame->mvuRight[i]<0)
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                Eigen::Matrix<double,2,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                //一元边：只连接位姿顶点（通过固定地图点坐标来优化位姿顶点）
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaMono);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                cv::Mat Xw = pMP->GetWorldPos();
                //设定地图点的坐标，但不参加优化
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vnIndexEdgeMono.push_back(i);
            }
            else  // Stereo observation 双目
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                //SET EDGE
                Eigen::Matrix<double,3,1> obs;// 这里和单目不同
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                const float &kp_ur = pFrame->mvuRight[i];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;// 这里和单目不同

                g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();// 这里和单目不同

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaStereo);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                e->bf = pFrame->mbf;
                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vnIndexEdgeStereo.push_back(i);
            }
        }

    }
    }


    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    // 步骤4：开始优化，总共优化四次，每次优化后，将观测分为outlier和inlier，outlier不参与下次优化
    // 由于每次优化后是对所有的观测进行outlier和inlier判别，因此之前被判别为outlier有可能变成inlier，反之亦然
    // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};// 四次迭代，每次迭代的次数

    int nBad=0;
    for(size_t it=0; it<4; it++)//迭代四次
    {

        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        optimizer.initializeOptimization(0);// 对level为0的边进行优化
        optimizer.optimize(its[it]);//这里设置完之后，实际已经开始了优化

        nBad=0;
        //根据边的误差大小，对每个单目/RGB-D和双目边区分外点和内点边。
        //在下一次位姿估计值产生后，都这次判定为外点的边在进行一次判断。也许此时该边会成为内点边
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            //可能在之前的位姿估计值中某点是外点，但是在之后的位姿估计值中该点又能够成为内点
            if(pFrame->mvbOutlier[idx])//对属于外点的边单独进行基于当前位姿的误差计算。
            {
                e->computeError(); // NOTE g2o只会计算active edge（内点边）的误差
            }

            const float chi2 = e->chi2();//边的误差

            if(chi2>chi2Mono[it])//将误差大于阈值的边设置为外点边，并将其对应的地图点设置为外点
            {                
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);                 // 设置为outlier
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);                 // 设置为inlier
            }

            if(it==2)
                e->setRobustKernel(0); // 除了前两次优化需要RobustKernel以外, 其余的优化都不需要
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {                
                e->setLevel(0);
                pFrame->mvbOutlier[idx]=false;
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        if(optimizer.edges().size()<10)//边的数目太少（内点数太少）则退出
            break;
    }    

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose);//设置优化后位姿

    return nInitialCorrespondences-nBad;//返回内点数量（感觉nBad的计数不准）？？？
}

/**
 * @brief Local Bundle Adjustment
 *
 * 1. Vertex:
 *     - g2o::VertexSE3Expmap()，LocalKeyFrames，即当前关键帧的位姿、与当前关键帧相连的关键帧的位姿
 *     - g2o::VertexSE3Expmap()，FixedCameras，即能观测到LocalMapPoints的关键帧（并且不属于LocalKeyFrames）的位姿，在优化中这些关键帧的位姿不变
 *     - g2o::VertexSBAPointXYZ()，LocalMapPoints，即LocalKeyFrames能观测到的所有MapPoints的位置
 * 2. Edge:
 *     - g2o::EdgeSE3ProjectXYZ()，BaseBinaryEdge
 *         + Vertex：关键帧的Tcw，MapPoint的Pw
 *         + measurement：MapPoint在关键帧中的二维位置(u,v)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *     - g2o::EdgeStereoSE3ProjectXYZ()，BaseBinaryEdge
 *         + Vertex：关键帧的Tcw，MapPoint的Pw
 *         + measurement：MapPoint在关键帧中的二维位置(ul,v,ur)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *         
 * @param pKF        KeyFrame
 * @param pbStopFlag 是否停止优化的标志
 * @param pMap       在优化后，更新状态时需要用到Map的互斥量mMutexMapUpdate
 */
/*1、在增加新KF时进行一次局部优化
2、将与增加的KF关联的所有KFs作为局部KFs，并将这些KFs观测到的地图点作为局部地图点
3、将局部KFs和局部地图点构造成优化器中的顶点
4、进行优化。有时甚至会进行第二遍优化（先进行一次内外点判断，再进行第二次优化）
4、获得优化值，依据误差较大的边，切断其包含的KF--MapPoint间的联系
5、更新局部KFs和局部地图点的估计值*/
void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap)
{
    // 该优化函数用于LocalMapping线程的局部BA优化

    // Local KeyFrames: First Breadth Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames;

    // 步骤1：将当前关键帧加入lLocalKeyFrames
    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;//为了该KF进行的局部BA

    // 步骤2：找到关键帧连接的关键帧（一级相连），加入lLocalKeyFrames中
    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();//返回当前帧所链接着的KFs（排序后）
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;//设置该联系KF要为当前帧进行局部BA
        if(!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames
    // 步骤3：遍历lLocalKeyFrames中关键帧，将它们观测的MapPoints加入到lLocalMapPoints
    list<MapPoint*> lLocalMapPoints;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();//获得帧中包含的地图点
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
            {
                if(!pMP->isBad())
                    if(pMP->mnBALocalForKF!=pKF->mnId)// 防止重复添加
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;// 防止重复添加
                    }
            }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    // 步骤4：得到能被局部MapPoints观测到，但不属于局部关键帧的关键帧，这些关键帧在局部BA优化时不优化（仅提供参考作用）
    list<KeyFrame*> lFixedCameras;
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)//遍历局部地图点
    {
        map<KeyFrame*,size_t> observations = (*lit)->GetObservations();//得到地图点的观测帧和其在当中对应特征点的索引
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)//遍历该地图点的所有KFs
        {
            KeyFrame* pKFi = mit->first;

            // pKFi->mnBALocalForKF!=pKF->mnId表示局部关键帧，
            // 其它的关键帧虽然能观测到，但不属于局部关键帧
            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)// 第一项判断该KF在不在局部KFs中第二项防止重复添加
            {                
                pKFi->mnBAFixedForKF=pKF->mnId;// 防止重复添加
                if(!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // Setup optimizer
    // 步骤5：构造g2o优化器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    // Set Local KeyFrame vertices
    // 步骤6：添加顶点：Pose of Local KeyFrame
    //将局部KFs设置成优化器中的顶点
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId==0);//第一帧位置固定
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // Set Fixed KeyFrame vertices
    // 步骤7：添加顶点：Pose of Fixed KeyFrame，注意这里调用了vSE3->setFixed(true)。
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);//这些KFs的位姿是固定的
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // Set MapPoint vertices
    // 步骤7：添加3D顶点
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();//边的数量

    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;//单目边
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;//单目边对应的KF
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;//构建单目边的地图点
    vpMapPointEdgeMono.reserve(nExpectedSize);

    //同上
    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)//遍历局部地图点，建立地图点顶点
    {
        // 添加顶点：MapPoint
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        // Set edges
        // 步骤8：对每一对关联的MapPoint和KeyFrame构建边
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)//将地图点顶点与其观测KFs连接，构建边
        {
            KeyFrame* pKFi = mit->first;

            if(!pKFi->isBad())
            {                
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                // Monocular observation和之前的函数类似
                if(pKFi->mvuRight[mit->second]<0)
                {
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else // Stereo observation
                {
                    Eigen::Matrix<double,3,1> obs;
                    const float kp_ur = pKFi->mvuRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);//设置核函数
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
            }
        }
    }

    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    // 步骤9：开始优化
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    bool bDoMore= true;//是否做更多--即区分当前优化位姿下的内外点，进行新一轮优化

    if(pbStopFlag)//？？？？
        if(*pbStopFlag)
            bDoMore = false;

    if(bDoMore)
    {

    // Check inlier observations
    // 步骤10：检测outlier，并设置下次不优化
    //遍历单目、双目边，检测出内外点，进行新一轮优化
    //新一轮优化不使用核函数。且不会考虑现在的外点是否能够在下次优化后成为内点的情况
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
        if(e->chi2()>5.991 || !e->isDepthPositive())//边的误差大于阈值，或者其深度为负（点跑到了相机后面）---认为其是外点
        {
            e->setLevel(1);// 不优化
        }

        e->setRobustKernel(0);// 不使用核函数
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)//遍历所有双目边
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            e->setLevel(1);//将边放到第1层，之后不再优化
        }

        e->setRobustKernel(0);
    }

    // Optimize again without the outliers
    // 步骤11：排除误差较大的outlier后再次优化
    optimizer.initializeOptimization(0);//只优化第0层的边
    optimizer.optimize(10);//开始第二次优化

    }

    vector<pair<KeyFrame*,MapPoint*> > vToErase;//保存需要剔除的KF--地图点的连接边
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());

    // Check inlier observations
    // 步骤12：在优化后位姿重新计算误差，剔除连接误差比较大的关键帧和MapPoint
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)//先检查单目边
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
        //疑问：这里是对所有单目边进行操作，但是优化器只对内点边进行误差计算，故之前判定为外点的边的误差都是落伍了，这样还能可靠地剔除误匹配吗？
        if(e->chi2()>5.991 || !e->isDepthPositive())//或许可以，因为之前判定外点的依据和这里的一样
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // 连接偏差比较大，在关键帧中剔除对该MapPoint的观测
    // 连接偏差比较大，在MapPoint中剔除对该关键帧的观测
    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);//将地图点从KF中剔除（因为观测误差较大）
            pMPi->EraseObservation(pKFi);//将KF从地图点的观测帧中剔除
        }
    }

    // Recover optimized data
    // 步骤13：优化后更新关键帧位姿以及MapPoints的位置、平均观测方向等属性

    //Keyframes更新每个关键帧优化后的位姿值
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->SetPose(Converter::toCvMat(SE3quat));
    }

    //Points更新优化后每个地图点的值
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }
}

/**
 * @brief 闭环检测后，EssentialGraph优化
 *
 * 1. Vertex:
 *     - g2o::VertexSim3Expmap，Essential graph中关键帧的位姿
 * 2. Edge:
 *     - g2o::EdgeSim3()，BaseBinaryEdge
 *         + Vertex：关键帧的Tcw，MapPoint的Pw
 *         + measurement：经过CorrectLoop函数步骤2，Sim3传播校正后的位姿
 *         + InfoMatrix: 单位矩阵     
 *
 * @param pMap               全局地图
 * @param pLoopKF            闭环匹配上的关键帧（与CurKF匹配上的）
 * @param pCurKF             当前关键帧
 * @param NonCorrectedSim3   未经过Sim3传播调整过的关键帧位姿（是一个容器，由pair组成：KeyFrame*+Sim3）
 * @param CorrectedSim3      经过Sim3传播调整过的关键帧位姿（Sim3就是相似变换）：加入了尺度变化的相机位位姿（同上）
 * @param LoopConnections    因闭环时MapPoints调整而新生成的边
 */
//所有程序都是为了能够充分利用地图中的所有KFs，建立尽可能多的KFs间的连接边，以便优化出最好的位姿
/*1、形成闭环之后，要对整个地图进行一次位姿优化（消除累积误差）
2、将当前Map中的所有KFs拿出来，构造顶点，但是要将当前KF和LoopKF的顶点设置为固定的（参考）。此时KFs顶点使用的是Sim3变换位姿
3、由于闭环的出现，会增加各个KFs与新的共视KF间的连接（共视边）。故通过新的共视关系，以帧间相对位姿为测量值，构造两帧间的连接边
4、再对各个KF构造其与其父KF间的边，以及其与它之前（编号小于它）的KFs间的边。边的测量值最好采用未加入尺度变换S的变换阵
5、再对各个KFs增添其与它共视程度较高的KFs（高于阈值）。这里最好用非Sim3变换阵*/
void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections, const bool &bFixScale)
{
    // Setup optimizer
    // 步骤1：构造优化器
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    // 指定线性方程求解器使用Eigen的块求解器
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    // 构造线性求解器
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    // 使用LM算法进行非线性迭代
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    //获得地图中的KFs和地图点
    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    // 仅经过Sim3传播调整，未经过优化的keyframe的pose（是剔除了相似变换的尺度？？）
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);//保存对应编号的KF的位姿
    // 经过Sim3传播调整，经过优化的keyframe的pose
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);
    // 这个变量没有用
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);

    const int minFeat = 100;//共视程度的阈值

    // Set KeyFrame vertices
    // 步骤2：将地图中所有keyframe的pose作为顶点添加到优化器
    // 尽可能使用经过Sim3调整的位姿作为顶点的初值
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();// 一直new，不用释放？(wubo???)

        const int nIDi = pKF->mnId;//这里是为了使vScw的下标与KF的编号相对应

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);//看该KF是否经过Sim3调整（是否乘了尺度变换值）

        //设置顶点位姿初始值
        // 如果该关键帧在闭环时通过Sim3传播调整过，用校正后的位姿
        if(it!=CorrectedSim3.end())
        {
            vScw[nIDi] = it->second;//给对应编号的KF设置位姿
            VSim3->setEstimate(it->second);
        }
        else// 如果该关键帧在闭环时没有通过Sim3传播调整过，用自身的位姿
        {
            Eigen::Matrix<double,3,3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
            Eigen::Matrix<double,3,1> tcw = Converter::toVector3d(pKF->GetTranslation());
            g2o::Sim3 Siw(Rcw,tcw,1.0);//变换尺度为1.0？？（没变换）
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }

        // 闭环匹配上的帧（与当前帧形成闭环的帧）不进行位姿优化
        if(pKF==pLoopKF)
            VSim3->setFixed(true);//将闭环匹配上的帧设置为固定的（参考帧？？Yu）

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = bFixScale;

        optimizer.addVertex(VSim3);

        // 优化前的pose顶点，后面代码中没有使用
        vpVertices[nIDi]=VSim3;
    }


    set<pair<long unsigned int,long unsigned int> > sInsertedEdges;//保存插入的边连接的是哪两个KFs

    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();//？？？？？

    // Set Loop edges
    // 步骤3：添加边：LoopConnections是闭环时因为MapPoints调整而出现的新关键帧连接关系（不是当前帧与闭环匹配帧之间的连接关系）
    //地图点经过调整后，更新的KF与其共视KFs的连接关系
    for(map<KeyFrame *, set<KeyFrame *> >::const_iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)//遍历该闭环中的所有KFs
    {
        KeyFrame* pKF = mit->first;//获得闭环中的KF
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame*> &spConnections = mit->second;//与该KF相连的KFs
        const g2o::Sim3 Siw = vScw[nIDi];//根据该KF的编号获得其位姿
        const g2o::Sim3 Swi = Siw.inverse();//反变换

        //使用KF与其想连的其他KFs之间的相对位姿作为边。这个边是其他KFs到当前KF的连线，不是相邻KF的连线
        for(set<KeyFrame*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)//遍历与该KF有联系（共视程度高的）的KFs
        {
            const long unsigned int nIDj = (*sit)->mnId;
            if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat)//？？？？
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];
            // 得到两个pose间的Sim3变换
            const g2o::Sim3 Sji = Sjw * Swi;//计算该KF与这个相连KF间的相对位姿

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            // 根据两个Pose顶点的位姿算出相对位姿作为边，那还存在误差？优化有用？因为闭环MapPoints调整新形成的边不优化？（wubo???）
            //什么是闭环地图点调整而形成的边？
            e->setMeasurement(Sji);

            e->information() = matLambda;

            optimizer.addEdge(e);

            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));
        }
    }

    // Set normal edges
    // 步骤4：添加跟踪时形成的边、闭环匹配成功形成的边
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Swi;

        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

        // 尽可能得到未经过Sim3传播调整的位姿
        if(iti!=NonCorrectedSim3.end())
            Swi = (iti->second).inverse();//使用未经过Sim3调整的位姿
        else
            Swi = vScw[nIDi].inverse();//使用调整了的位姿

        KeyFrame* pParentKF = pKF->GetParent();

        // Spanning tree edge
        // 步骤4.1：只添加扩展树的边（有父关键帧）
        //仅建立KF与其父KF的连接边
        if(pParentKF)
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Sjw;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            // 尽可能得到未经过Sim3传播调整的位姿
            if(itj!=NonCorrectedSim3.end())
                Sjw = itj->second;
            else
                Sjw = vScw[nIDj];

            g2o::Sim3 Sji = Sjw * Swi;//获得KF与其父KF间的相对变换（为经过Sim3调整的）

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // Loop edges
        // 步骤4.2：添加在CorrectLoop函数中AddLoopEdge函数添加的闭环连接边
        // 使用经过Sim3调整前关键帧之间的相对关系作为边
        //只对编号在选中的KF之前的KFs进行操作，构建相对位姿的边
        const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();//？？？看了闭环那个模块应该就能懂了
        for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame* pLKF = *sit;
            if(pLKF->mnId<pKF->mnId)
            {
                g2o::Sim3 Slw;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                // 尽可能得到未经过Sim3传播调整的位姿
                if(itl!=NonCorrectedSim3.end())
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->mnId];

                g2o::Sim3 Sli = Slw * Swi;
                g2o::EdgeSim3* el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));//一头连着当前KF之前的一个KF
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));//一头连着当前选中的KF
                // 根据两个Pose顶点的位姿算出相对位姿作为边，那还存在误差？优化有用？（wubo???）
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        // Covisibility graph edges
        // 步骤4.3：最有很好共视关系的关键帧也作为边进行优化
        // 使用经过Sim3调整前关键帧之间的相对关系作为边
        const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);//获得与被选中的KF共视程度大于阈值的KFs
        for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;
            if(pKFn && pKFn!=pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))//防止重复添加边
            {
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId)//编号也要小于当前选中的KF
                {
                    if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))//看看是否已经添加了这个边
                        continue;

                    g2o::Sim3 Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);//最好使用未经过Sim3变换的位姿

                    // 尽可能得到未经过Sim3传播调整的位姿
                    if(itn!=NonCorrectedSim3.end())
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->mnId];//使用经过Sim3调整后的位姿

                    g2o::Sim3 Sni = Snw * Swi;//计算相对位姿

                    g2o::EdgeSim3* en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }
    }
    //上面的所有程序都是为了能够充分利用地图中的所有KFs，建立尽可能多的KFs间的连接边，以便优化出最好的位姿

    // Optimize!
    // 步骤5：开始g2o优化
    optimizer.initializeOptimization();
    optimizer.optimize(20);//优化完成

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    // 步骤6：设定优化后的位姿
    //给每个KFs更新优化后的位姿
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();//优化后的位姿
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();//保存优化后的位姿的逆（从相机坐标系到世界坐标系）
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        double s = CorrectedSiw.scale();

        eigt *=(1./s); //[R t/s;0 1]

        cv::Mat Tiw = Converter::toCvSE3(eigR,eigt);

        pKFi->SetPose(Tiw);
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    // 步骤7：步骤5和步骤6优化得到关键帧的位姿后，MapPoints根据参考帧优化前后的相对关系调整自己的位置
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        int nIDr;
        // 该MapPoint经过Sim3调整过，(LoopClosing.cpp，CorrectLoop函数，步骤2.2_
        if(pMP->mnCorrectedByKF==pCurKF->mnId)//如果地图点被当前帧修正过，说明该点知道创建它的那个KF的编号
        {
            nIDr = pMP->mnCorrectedReference;
        }
        else
        {
            // 通过情况下MapPoint的参考关键帧就是创建该MapPoint的那个关键帧
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }

        // 得到MapPoint参考关键帧步骤5优化前的位姿
        g2o::Sim3 Srw = vScw[nIDr];
        // 得到MapPoint参考关键帧优化后的位姿（从相机坐标系到世界坐标系）
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        cv::Mat P3Dw = pMP->GetWorldPos();
        Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));//先将地图点从世界坐标系转换到相机坐标系（用未优化的位姿），在将其转回去（用优化的位姿）

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMP->SetWorldPos(cvCorrectedP3Dw);

        pMP->UpdateNormalAndDepth();
    }
}

/**
 * @brief 形成闭环时进行Sim3优化
 *
 * 1. Vertex:
 *     - g2o::VertexSim3Expmap()，两个关键帧的位姿
 *     - g2o::VertexSBAPointXYZ()，两个关键帧共有的MapPoints
 * 2. Edge:
 *     - g2o::EdgeSim3ProjectXYZ()，BaseBinaryEdge
 *         + Vertex：关键帧的Sim3，MapPoint的Pw
 *         + measurement：MapPoint在关键帧中的二维位置(u,v)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *     - g2o::EdgeInverseSim3ProjectXYZ()，BaseBinaryEdge
 *         + Vertex：关键帧的Sim3，MapPoint的Pw
 *         + measurement：MapPoint在关键帧中的二维位置(u,v)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *         
 * @param pKF1        KeyFrame
 * @param pKF2        KeyFrame
 * @param vpMatches1  两个关键帧的匹配关系--下标：pKF1中地图点索引；值：pKF2中的匹配的地图点
 * @param g2oS12      两个关键帧间的Sim3变换
 * @param th2         核函数阈值
 * @param bFixScale   是否优化尺度，弹目进行尺度优化，双目不进行尺度优化
 */
/*1、初始化求解器
2、通过g2oS12设置位姿顶点的初始值
3、通过vpMatches1，获得两帧中匹配的地图点。将匹配的地图点（两帧的）转换成对应的顶点
4、根据相似变换关系，建立对应地图点和位姿顶点的误差边
5、进行优化。优化后剔除外点，消除误匹配（两帧地图点的错误匹配）
6、都剩下的点再进行优化
7、再次剔除外点。获得优化后位姿，再次剔除误匹配。
8、统计最终内点数目并返回*/
int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2, const bool bFixScale)
{
    // 步骤1：初始化g2o优化器
    // 先构造求解器
    g2o::SparseOptimizer optimizer;

    // 构造线性方程求解器，Hx = -b的求解器
    g2o::BlockSolverX::LinearSolverType * linearSolver;
    // 使用dense的求解器，（常见非dense求解器有cholmod线性求解器和shur补线性求解器）
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    // 使用L-M迭代
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Calibration
    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    // Camera poses
    const cv::Mat R1w = pKF1->GetRotation();
    const cv::Mat t1w = pKF1->GetTranslation();
    const cv::Mat R2w = pKF2->GetRotation();
    const cv::Mat t2w = pKF2->GetTranslation();

    // Set Sim3 vertex
    // 步骤2.1 添加Sim3顶点
    g2o::VertexSim3Expmap * vSim3 = new g2o::VertexSim3Expmap();    
    vSim3->_fix_scale=bFixScale;
    vSim3->setEstimate(g2oS12);//设置Sim3位姿变换初值：从pKF2到pKF1的Sim3变换
    vSim3->setId(0);
    vSim3->setFixed(false);// 优化Sim3顶点
    //设置Sim3顶点中两帧的相机内参
    vSim3->_principle_point1[0] = K1.at<float>(0,2); // 光心横坐标cx
    vSim3->_principle_point1[1] = K1.at<float>(1,2); // 光心纵坐标cy
    vSim3->_focal_length1[0] = K1.at<float>(0,0); // 焦距 fx
    vSim3->_focal_length1[1] = K1.at<float>(1,1); // 焦距 fy
    vSim3->_principle_point2[0] = K2.at<float>(0,2);
    vSim3->_principle_point2[1] = K2.at<float>(1,2);
    vSim3->_focal_length2[0] = K2.at<float>(0,0);
    vSim3->_focal_length2[1] = K2.at<float>(1,1);
    optimizer.addVertex(vSim3);

    // Set MapPoint vertices
    const int N = vpMatches1.size();
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();

    vector<g2o::EdgeSim3ProjectXYZ*> vpEdges12; //pKF2对应的MapPoints到pKF1的投影边
    vector<g2o::EdgeInverseSim3ProjectXYZ*> vpEdges21; //pKF1对应的MapPoints到pKF2的投影边
    vector<size_t> vnIndexEdge;

    vnIndexEdge.reserve(2*N);
    vpEdges12.reserve(2*N);
    vpEdges21.reserve(2*N);

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;

    for(int i=0; i<N; i++)//遍历两KF中匹配的地图点
    {
        if(!vpMatches1[i])
            continue;

        // pMP1和pMP2是匹配的MapPoints（为什么会是匹配的地图点会有两种不同的表达）：或许是因为相似变换会改变地图点的坐标以及其描述子，所以就有可能产生不同表达
        MapPoint* pMP1 = vpMapPoints1[i];
        MapPoint* pMP2 = vpMatches1[i];

        const int id1 = 2*i+1;//为了让两个匹配地图点构造的顶点的编号是连续的
        const int id2 = 2*(i+1);

        const int i2 = pMP2->GetIndexInKeyFrame(pKF2);//获得在pKF2中的索引

        if(pMP1 && pMP2)
        {
            if(!pMP1->isBad() && !pMP2->isBad() && i2>=0)//将两个匹配的地图点设置成优化图中的顶点
            {
                // 步骤2.2 添加PointXYZ顶点
                g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D1w = pMP1->GetWorldPos();
                cv::Mat P3D1c = R1w*P3D1w + t1w;//在pKF1下的相机坐标
                vPoint1->setEstimate(Converter::toVector3d(P3D1c));
                vPoint1->setId(id1);
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D2w = pMP2->GetWorldPos();
                cv::Mat P3D2c = R2w*P3D2w + t2w;
                vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
                continue;
        }
        else
            continue;

        nCorrespondences++;

        //添加边
        // Set edge x1 = S12*X2--pKF2中的地图点到pKF1中的变换
        Eigen::Matrix<double,2,1> obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;//这个边连接的是pKF2到pKF1，因此边的测量值应该是pKF2对应地图点在pKF1中的像素坐标
        //故此边是通过最小化重投影误差优化相似变换位姿和pKF2地图点的坐标

        // 步骤2.3 添加两个顶点（3D点）到相机投影的边
        g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e12->setMeasurement(obs1);
        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);

        //设置核函数
        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        //类似上面
        // Set edge x2 = S21*X1
        Eigen::Matrix<double,2,1> obs2;
        const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];
        obs2 << kpUn2.pt.x, kpUn2.pt.y;

        g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();//因为是上面的逆相似变换，所以这个边是Inverse（修改了计算误差的方式）

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);//保存从2->1的边
        vpEdges21.push_back(e21);//相反
        vnIndexEdge.push_back(i);//上述两边是由第i个匹配地图点所创造的
    }

    // Optimize!
    // 步骤3：g2o开始优化，先迭代5次
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // 步骤4：剔除一些误差大的边
    // Check inliers
    // 进行卡方检验，大于阈值的边剔除
    int nBad=0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)//误差较大时剔除。匹配地图点产生的两个边只要有一个误差较大，就视这个匹配是错误的
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);//剔除pKF1地图点与pKF2地图点的匹配关系
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i]=static_cast<g2o::EdgeSim3ProjectXYZ*>(NULL);//剔除idx对应的两个匹配地图点所对应的两个边
            vpEdges21[i]=static_cast<g2o::EdgeInverseSim3ProjectXYZ*>(NULL);
            nBad++;
        }
    }

    int nMoreIterations;
    if(nBad>0)
        nMoreIterations=10;
    else
        nMoreIterations=5;

    if(nCorrespondences-nBad<10)
        return 0;

    // Optimize again only with inliers使用上次留下的内点进行再次优化
    // 步骤5：再次g2o优化剔除后剩下的边
    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn = 0;//统计这次优化后的内点数目
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
        }
        else
            nIn++;
    }

    // Recover optimized Sim3
    // 步骤6：得到优化后的结果
    g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    g2oS12= vSim3_recov->estimate();

    return nIn;//返回内点数
}


} //namespace ORB_SLAM
