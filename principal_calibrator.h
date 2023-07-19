#pragma once
#include "calibrator_base.h"
class PrincipalCalibrator : Calibrator
{
public:

	PrincipalCalibrator(const cv::Rect& roi, int limit_in_a_bin, int min_number_of_bin, int max_number_of_bin, double ratio_threshold, double principal_boundary);

	struct PrincipalResidual 
	{
		PrincipalResidual(const std::vector<std::vector<std::pair<Feature, Feature>>>& set_of_featurepairs, double f_x, double f_y)
			: feature_pairs_(set_of_featurepairs), f_x_(f_x), f_y_(f_y) {}

		template <typename T>
		bool operator()(const double* const c_x, const double* const c_y, T* residual) const {
			//For math and equations you should read imatest camera projection

			Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
			K(0, 0) = f_x_;
			K(1, 1) = f_y_;;
			K(0, 2) = *c_x;
			K(1, 2) = *c_y;
			double final_d = 0;
			for (size_t set_number = 0; set_number < feature_pairs_.size(); set_number++)
			{
				for (size_t f_p = 0; f_p < feature_pairs_[set_number].size(); f_p++)
				{
					Eigen::Matrix3d rotation_1 = feature_pairs_[set_number][f_p].second.M.block<3, 3>(0, 0);
					Eigen::Vector3d translation_1 = feature_pairs_[set_number][f_p].second.M.block<3, 1>(0, 3);

					Eigen::Matrix3d rotation_0 = Eigen::Matrix3d::Identity();
					Eigen::Vector3d translation_0 = Eigen::Vector3d::Zero();

					Eigen::Vector3d P_i = { feature_pairs_[set_number][f_p].first.feature_point.x, feature_pairs_[set_number][f_p].first.feature_point.y,1.0 };
					Eigen::Vector3d P_i_star = { feature_pairs_[set_number][f_p].second.feature_point.x, feature_pairs_[set_number][f_p].second.feature_point.y,1.0 };

					Eigen::Vector3d ray_direction0 = (rotation_0 * K.inverse() * P_i);
					Eigen::Vector3d ray_direction1 = (rotation_1 * K.inverse() * P_i_star);
					Eigen::Vector3d N = ray_direction0.cross(ray_direction1);

					Eigen::Vector3d PQ = translation_1 - translation_0; //trans_0 -> trans_1 vector

					Eigen::Vector3d P1 = (ray_direction0)+translation_0;
					Eigen::Vector3d P2 = (ray_direction1)+translation_1;
					Eigen::Vector3d V = P1.cross(P2);

					double d = std::abs(PQ.dot(V)) / V.norm();
					final_d += d;
				}
			}
			residual[0] = final_d;
			return true;

		}
	private:
		const double f_x_;
		const double f_y_;
		const std::vector<std::vector<std::pair<Feature, Feature>>> feature_pairs_;
	};

	struct PrincipalResidualWithFundamental
	{
		PrincipalResidualWithFundamental(const std::vector<std::vector<std::pair<Feature, Feature>>>& set_of_featurepairs, double f_x, double f_y)
			: feature_pairs_(set_of_featurepairs), f_x_(f_x), f_y_(f_y) {}

		template <typename T>
		bool operator()(const double* const c_x, const double* const c_y, T* residual) const {
			//For math and equations you should read imatest camera projection

			Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
			K(0, 0) = f_x_;
			K(1, 1) = f_y_;;
			K(0, 2) = *c_x;
			K(1, 2) = *c_y;
			cv::Mat cameraMatrix(3, 3, CV_32F);
			cv::eigen2cv(K, cameraMatrix);
			double final_d = 0;
			for (size_t set_number = 0; set_number < feature_pairs_.size(); set_number++)
			{
				if (feature_pairs_[set_number].size() < 8)
				{
					continue;
				}
				std::vector<cv::Point2f> inputPoints1;
				std::vector<cv::Point2f> inputPoints2;
				for (size_t f_p = 0; f_p < feature_pairs_[set_number].size(); f_p++)
				{
					inputPoints1.emplace_back(feature_pairs_[set_number][f_p].first.feature_point);
					inputPoints2.emplace_back(feature_pairs_[set_number][f_p].second.feature_point);
					
				}
				cv::Mat fundamental_matrix = cv::findFundamentalMat(inputPoints1, inputPoints2,cv::FM_8POINT,0.99,1);

				std::vector<Eigen::Vector3f> input1;
				input1.reserve(inputPoints1.size());
				std::vector<Eigen::Vector3f> input2;
				input2.reserve(inputPoints2.size());
				Eigen::Matrix3f fundamental;

				for (size_t i = 0; i < inputPoints2.size(); i++)
				{
					input1.emplace_back(Eigen::Vector3f(inputPoints1[i].x, inputPoints1[i].y, 1));
					input2.emplace_back(Eigen::Vector3f(inputPoints2[i].x, inputPoints2[i].y, 1));
				}
				//std::cout << fundamental_matrix << std::endl;
				cv::cv2eigen(fundamental_matrix, fundamental);
				//Eigen::Matrix3d K_inv = K.inverse();
				for (size_t i = 0; i < input2.size(); i++)
				{
					double ret = abs(input2[i].transpose() * (fundamental) * input1[i]);
					//std::cout << ret << std::endl;
					//std::cout << outputUndistortedPoints1[i] << ", " << outputUndistortedPoints2[i] << std::endl;
					final_d += ret;
				}
			}
			residual[0] = final_d;
			return true;

		}
	private:
		const double f_x_;
		const double f_y_;
		const std::vector<std::vector<std::pair<Feature, Feature>>> feature_pairs_;
	};


	bool calibrate() override;

	void addFeatures(const std::vector<std::pair<Feature, Feature>>& feature_pairs) override;

	void exportShapeOfProblemSpace(const std::string& sequence_number, const std::string& sequence_length);

	static double calculatePrincipalResidualLoss(const std::vector<std::vector<std::pair<Feature, Feature>>>& feature_paires, const Eigen::Matrix3d& K);

	Eigen::Matrix3d getIntrinsicMatrix() override;

	void setIntrinsicMatrix(const Eigen::Matrix3d& K);

private:
	std::vector<std::vector<std::pair<Feature, Feature>>> list_of_feature_pairs_;
	Bins bins_;
	double minimum_feature_distance_;
	double principal_boundary_;
	size_t number_of_featurepairs = 0;
};