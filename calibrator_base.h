#pragma once
#include <iostream>
#include <vector>
#include "structures.h"
#include "ceres/solver.h"
#include "ceres/numeric_diff_cost_function.h"


class Calibrator
{
public:

	virtual bool calibrate() = 0;

	virtual void addFeatures(const std::vector<std::pair<Feature,Feature>>& feature_pairs) = 0;

	virtual Eigen::Matrix3d getIntrinsicMatrix() = 0;

	virtual void setIntrinsicMatrix(const Eigen::Matrix3d& K) = 0;

	static std::vector<std::vector<std::pair<Feature, Feature>>> undistortFeaturePairs(double k_1, const Eigen::Matrix3d& K, const std::vector<std::vector<std::pair<Feature, Feature>>>& input_framepairs_set)
	{
		std::vector<std::vector<std::pair<Feature, Feature>>> ret;
		cv::Mat cameraMatrix(3, 3, CV_16F);
		cv::eigen2cv(K, cameraMatrix);
		cv::Mat distCoeffs = (cv::Mat_<float>(5, 1) << k_1, 0, 0, 0, 0);
		for (const auto& current_feature_pairs : input_framepairs_set)
		{
			std::vector<cv::Point2f> inputDistortedPoints1;
			std::vector<cv::Point2f> inputDistortedPoints2;
			std::vector<cv::Point2f> outputUndistortedPoints1;
			std::vector<cv::Point2f> outputUndistortedPoints2;
			std::vector<std::pair<Feature, Feature>> undistorted_imagepairs;
			for (const auto& feature_pair : current_feature_pairs)
			{
				Feature one;
				one.image_id = feature_pair.first.image_id;
				one.M = feature_pair.first.M;
				Feature two;
				two.image_id = feature_pair.second.image_id;
				two.M = feature_pair.second.M;
				undistorted_imagepairs.emplace_back(one, two);
				//std::cout << "Distorted second point is: " << two.M << std::endl;
				//std::cout << feature_pair.second.feature_point << std::endl;
				inputDistortedPoints1.emplace_back(feature_pair.first.feature_point);
				inputDistortedPoints2.emplace_back(feature_pair.second.feature_point);
			}
			if (inputDistortedPoints1.empty() || inputDistortedPoints2.empty())
			{
				continue;
			}
			cv::undistortImagePoints(inputDistortedPoints1, outputUndistortedPoints1, cameraMatrix, distCoeffs);
			cv::undistortImagePoints(inputDistortedPoints2, outputUndistortedPoints2, cameraMatrix, distCoeffs);
			if (undistorted_imagepairs.size() != outputUndistortedPoints1.size())
			{
				return ret;
			}
			for (size_t i = 0; i < undistorted_imagepairs.size(); i++)
			{
				//std::cout << "Undistorted second point is: " << undistorted_imagepairs[i].second.M << std::endl;
				//std::cout << outputUndistortedPoints2[i] << std::endl;
				undistorted_imagepairs[i].first.feature_point = outputUndistortedPoints1[i];
				undistorted_imagepairs[i].second.feature_point = outputUndistortedPoints2[i];
			}
			ret.emplace_back(undistorted_imagepairs);
		}
		return ret;
	};


protected:
	Eigen::Matrix3d K_ = Eigen::Matrix3d::Identity();
	DistrotionParams D_;

};

