/**
* This file is part of ORB-SLAM2.
* This file is a modified version of EPnP <http://cvlab.epfl.ch/EPnP/index.php>, see FreeBSD license below.
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

/**
* Copyright (c) 2009, V. Lepetit, EPFL
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
*    list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* The views and conclusions contained in the software and documentation are those
* of the authors and should not be interpreted as representing official policies,
*   either expressed or implied, of the FreeBSD Project
*/

#ifndef PNPSOLVER_H
#define PNPSOLVER_H

#include <opencv2/core/core.hpp>
#include "MapPoint.h"
#include "Frame.h"

namespace ORB_SLAM2
{

class PnPsolver {
 public:
  PnPsolver(const Frame &F, const vector<MapPoint*> &vpMapPointMatches);

  ~PnPsolver();

  void SetRansacParameters(double probability = 0.99, int minInliers = 8 , int maxIterations = 300, int minSet = 4, float epsilon = 0.4,
                           float th2 = 5.991);//设置RANSAC方法的参数

  cv::Mat find(vector<bool> &vbInliers, int &nInliers);//

  cv::Mat iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers);//进行迭代求解，并标志那些点对是内点以及内点的数量

 private:

  void CheckInliers();
  bool Refine();//

  // Functions from the original EPnP code
  void set_maximum_number_of_correspondences(const int n);
  void reset_correspondences(void);
  void add_correspondence(const double X, const double Y, const double Z,
              const double u, const double v);//

  double compute_pose(double R[3][3], double T[3]);

  void relative_error(double & rot_err, double & transl_err,
              const double Rtrue[3][3], const double ttrue[3],
              const double Rest[3][3],  const double test[3]);//

  void print_pose(const double R[3][3], const double t[3]);
  double reprojection_error(const double R[3][3], const double t[3]);

  void choose_control_points(void);//
  void compute_barycentric_coordinates(void);//计算重心坐标
  void fill_M(CvMat * M, const int row, const double * alphas, const double u, const double v);//
  void compute_ccs(const double * betas, const double * ut);//
  void compute_pcs(void);

  void solve_for_sign(void);

  void find_betas_approx_1(const CvMat * L_6x10, const CvMat * Rho, double * betas);
  void find_betas_approx_2(const CvMat * L_6x10, const CvMat * Rho, double * betas);
  void find_betas_approx_3(const CvMat * L_6x10, const CvMat * Rho, double * betas);
  void qr_solve(CvMat * A, CvMat * b, CvMat * X);

  double dot(const double * v1, const double * v2);
  double dist2(const double * p1, const double * p2);

  void compute_rho(double * rho);
  void compute_L_6x10(const double * ut, double * l_6x10);

  void gauss_newton(const CvMat * L_6x10, const CvMat * Rho, double current_betas[4]);
  void compute_A_and_b_gauss_newton(const double * l_6x10, const double * rho,
				    double cb[4], CvMat * A, CvMat * b);

  double compute_R_and_t(const double * ut, const double * betas,
			 double R[3][3], double t[3]);

  void estimate_R_and_t(double R[3][3], double t[3]);

  void copy_R_and_t(const double R_dst[3][3], const double t_dst[3],
		    double R_src[3][3], double t_src[3]);

  void mat_to_quat(const double R[3][3], double q[4]);


  double uc, vc, fu, fv;

  double * pws, * us, * alphas, * pcs;//pws：（求解EPNP时所用的匹配点中的）地图点在世界坐标系的坐标；us：特征点坐标；alphas：用四个控制点表示地图点坐标时的四个系数；pcs：地图点相机坐标系坐标
  //这些点都是通过指针去搜索，所以要注意地址的计算
  int maximum_number_of_correspondences;//最大匹配点对数量（保证上面的指针指向的空间足够存放所有的匹配点对）
  int number_of_correspondences;//匹配点对数量

  double cws[4][3], ccs[4][3];//cws是四个控制点在世界坐标系下的坐标。ccs是这些控制点在相机坐标系下的坐标
  double cws_determinant;

  vector<MapPoint*> mvpMapPointMatches;//地图点与特征点的匹配点对

  // 2D Points
  vector<cv::Point2f> mvP2D;
  vector<float> mvSigma2;//不同尺度的特征点对应不同的最大偏差。容器中的值影响最大偏差的大小

  // 3D Points世界坐标系
  vector<cv::Point3f> mvP3Dw;

  // Index in Frame帧中特征点索引
  vector<size_t> mvKeyPointIndices;

  // Current Estimation
  double mRi[3][3];
  double mti[3];
  cv::Mat mTcwi;
  vector<bool> mvbInliersi;//同下。只是这里记录的是那些点是内点
  int mnInliersi;//每次使用CheckInliers时都会提前归零。保存某个位姿估计下内点总数

  // Current Ransac State当前RANSAC状态：总共迭代了多少次+最好的内点状态+内点数量+求解出的最好位姿
  int mnIterations;//总迭代次数
  vector<bool> mvbBestInliers;//最好状态下内点情况
  int mnBestInliers;//最好时内点有多少
  cv::Mat mBestTcw;

  // Refined：提炼后的位姿+提炼后的内点状态+内点数量
  cv::Mat mRefinedTcw;
  vector<bool> mvbRefinedInliers;
  int mnRefinedInliers;

  // Number of Correspondences
  int N;//匹配点对的总数

  // Indices for random selection [0 .. N-1]
  vector<size_t> mvAllIndices;//

  // RANSAC probability
  double mRansacProb;

  // RANSAC min inliers
  int mRansacMinInliers;

  // RANSAC max iterations
  int mRansacMaxIts;

  // RANSAC expected inliers/total ratio
  float mRansacEpsilon;//内点/总点数量的比率

  // RANSAC Threshold inlier/outlier. Max error e = dist(P1,T_12*P2)^2（重投影误差）
  float mRansacTh;//区别内外点的阈值

  // RANSAC Minimun Set used at each iteration
  int mRansacMinSet;//

  // Max square error associated with scale level. Max error = th*th*sigma(level)*sigma(level)
  vector<float> mvMaxError;//根据特征点尺度的不同，对应匹配点对的重投影最大误差的要求也不同

};

} //namespace ORB_SLAM

#endif //PNPSOLVER_H
