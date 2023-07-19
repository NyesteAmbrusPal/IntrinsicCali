#pragma once
#include "calibrator_base.h"
class FocalCalibrator : Calibrator
{
public:

	FocalCalibrator(const cv::Rect& roi, int limit_in_a_bin, int min_number_of_bin, int max_number_of_bin, double minimum_feature_distance);

	struct FocalResiudal 
	{
		FocalResiudal(const std::vector<std::vector<std::pair<Feature, Feature>>>& set_of_featurepairs, double c_x, double c_y)
			: feature_pairs_(set_of_featurepairs), c_x(c_x) , c_y(c_y) {}

		template <typename T>
		bool operator()(const double* const f_x, T* residual) const {
			//For math and equations you should read imatest camera projection bool operator()(const double* const f_x, const double* const f_y, T* residual) const {
			// const double* const c_x, const double* const c_y,
			Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
			K(0, 0) = *f_x;
			K(1, 1) = *f_x;
			K(0, 2) = c_x;
			K(1, 2) = c_y;

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
					//Eigen::Vector3d N = ray_direction0.cross(ray_direction1);

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
		const std::vector<std::vector<std::pair<Feature, Feature>>> feature_pairs_;
		const double c_x;
		const double c_y;
	};

	struct IntrinsicResidual
	{
		IntrinsicResidual(const std::vector<std::vector<std::pair<Feature, Feature>>>& set_of_featurepairs)
			: feature_pairs_(set_of_featurepairs){}

		template <typename T>
		bool operator()(const double* const f_x, const double* const f_y, const double* const c_x, const double* const c_y, T* residual) const {
			//For math and equations you should read imatest camera projection bool operator()(const double* const f_x, const double* const f_y, T* residual) const {
			// const double* const c_x, const double* const c_y,
			Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
			K(0, 0) = *f_x;
			K(1, 1) = *f_y;
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
					//Eigen::Vector3d N = ray_direction0.cross(ray_direction1);

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
		const std::vector<std::vector<std::pair<Feature, Feature>>> feature_pairs_;
	};
	

	bool calibrate() override;

	void addFeatures(const std::vector<std::pair<Feature, Feature>>& feature_pairs) override;

	Eigen::Matrix3d getIntrinsicMatrix() override;

	void setIntrinsicMatrix(const Eigen::Matrix3d& K) override;

	void exportShapeOfProblemSpace(const std::string& sequence_number, const std::string& sequence_length);

private:

	static double calculateFocalResidualLoss(const std::vector<std::vector<std::pair<Feature, Feature>>>& feature_paires, const Eigen::Matrix3d& K);

	std::vector<std::vector<std::pair<Feature, Feature>>> list_of_feature_pairs_;
	Bins bins_;
	double minimum_feature_distance_;
	size_t number_of_featurepairs = 0;


};
