#pragma once

#include "calibrator_base.h"
#include "opencv2/core/eigen.hpp"

class RadialDistortionCalibrator : Calibrator
{
public:

	RadialDistortionCalibrator(const cv::Rect& roi, int limit_in_a_bin, int min_number_of_bin, int max_number_of_bin, double minimum_feature_distance);

	struct RadialDistortionResidual
	{
		RadialDistortionResidual(const std::vector<std::vector<std::pair<Feature, Feature>>>& set_of_featurepairs, const Eigen::Matrix3d& K)
			: feature_pairs_(set_of_featurepairs), K(K) {}

		template <typename T>
		bool operator()(const double* const k_1, T* residual) const {
			//For math and equations you should read imatest camera projection
			cv::Mat cameraMatrix(3, 3, CV_16F);
			cv::eigen2cv(K, cameraMatrix);
			cv::Mat distCoeffs = (cv::Mat_<float>(5, 1) << *k_1, 0, 0, 0, 0);
			residual[0] = calculateDistortionResidualLoss(feature_pairs_, cameraMatrix, distCoeffs);
			return true;
		}
	private:
		const std::vector<std::vector<std::pair<Feature, Feature>>> feature_pairs_;
		const Eigen::Matrix3d K;
	};

	struct RadialDistortionWithOdometry
	{
		RadialDistortionWithOdometry(const std::vector<std::vector<std::pair<Feature, Feature>>>& set_of_featurepairs, const Eigen::Matrix3d& K)
			: feature_pairs_(set_of_featurepairs), K(K){}
		template <typename T>
		bool operator()(const double* const k_1, const double* const k_2, T* residual) const {

			double final_d = 0;
			cv::Mat cameraMatrix(3, 3, CV_16F);
			cv::eigen2cv(K, cameraMatrix);
			cv::Mat distCoeffs = (cv::Mat_<float>(5, 1) << *k_1, *k_2, 0, 0, 0);

			final_d = calculateDistortionOdometryResidualLoss(feature_pairs_, K, distCoeffs);
			residual[0] = final_d;
			return true;
		}
	private:
		const std::vector<std::vector<std::pair<Feature, Feature>>> feature_pairs_;
		const Eigen::Matrix3d K;
	};


	struct RadialDistortionWithOdometryK2
	{
		RadialDistortionWithOdometryK2(const std::vector<std::vector<std::pair<Feature, Feature>>>& set_of_featurepairs, const Eigen::Matrix3d& K, double k1)
			: feature_pairs_(set_of_featurepairs), K(K), k1(k1) {}
		template <typename T>
		bool operator()(const double* const k_2, T* residual) const {

			double final_d = 0;
			cv::Mat cameraMatrix(3, 3, CV_16F);
			cv::eigen2cv(K, cameraMatrix);
			cv::Mat distCoeffs = (cv::Mat_<float>(5, 1) << k1,*k_2, 0, 0, 0);

			final_d = calculateDistortionOdometryResidualLoss(feature_pairs_, K, distCoeffs);
			residual[0] = final_d;
			return true;
		}
	private:
		const std::vector<std::vector<std::pair<Feature, Feature>>> feature_pairs_;
		const Eigen::Matrix3d K;
		const double k1;
	};

	bool calibrate() override;

	bool calibrateWithOdometry();

	bool calibrateWithOdometryK2();

	void addFeatures(const std::vector<std::pair<Feature, Feature>>& feature_pairs) override;

	Eigen::Matrix3d getIntrinsicMatrix() override;

	void setIntrinsicMatrix(const Eigen::Matrix3d& K);

	void calculateDistortionProblemSpace();

	double optimizeDistortionRange(bool optimize_k1, bool optimize_k2);

	void drawBinCapacity(cv::Mat image, const std::string& sequence_number);

	void drawFeatureLines(cv::Mat image);

private:
	static double calculateDistortionResidualLoss(const std::vector<std::vector<std::pair<Feature, Feature>>>& feature_paires, const cv::Mat& K, const cv::Mat& distortion_coeffs);

	static double calculateDistortionOdometryResidualLoss(const std::vector<std::vector<std::pair<Feature, Feature>>>& feature_paires, const Eigen::Matrix3d& K, const cv::Mat& distortion_coeffs);

	double calculateLinePointDistance(const cv::Point2d& direction_vector, const cv::Point2d& point_on_the_line, const cv::Point2d& middle_point);


private:
	std::vector<std::vector<std::pair<Feature, Feature>>> list_of_feature_pairs_;
	Bins bins_;
	size_t number_of_featurepairs = 0;
	double minimum_feature_distance_;
};