#include "principal_calibrator.h"
#include <iostream>
#include <fstream>
#include <string>
#define ENABLE_LOG 1

PrincipalCalibrator::PrincipalCalibrator(const cv::Rect& roi, int limit_in_a_bin, int min_number_of_bin, int max_number_of_bin, double minimum_feature_distance, 
	double principal_boundary) :
	bins_(roi, limit_in_a_bin, min_number_of_bin, max_number_of_bin), minimum_feature_distance_(minimum_feature_distance)
{
	principal_boundary_ = principal_boundary;
}

bool PrincipalCalibrator::calibrate()
{
	std::cout << "Found feature pairs: " << number_of_featurepairs << std::endl;
	double f_x = K_(0, 0);
	double f_y = K_(1, 1);
	double c_x = K_(0, 2);
	double c_y = K_(1, 2);
	ceres::Problem problem;
	ceres::CostFunction* cost_function = new ceres::NumericDiffCostFunction<PrincipalResidual, ceres::CENTRAL, 1, 1, 1>(
		new PrincipalResidual(list_of_feature_pairs_, f_x, f_y));

	problem.AddResidualBlock(cost_function, nullptr, &c_x, &c_y);
	problem.SetParameterLowerBound(&c_x, 0, 696 - principal_boundary_);
	problem.SetParameterLowerBound(&c_y, 0, 256 - principal_boundary_);
	problem.SetParameterUpperBound(&c_x, 0, 696 + principal_boundary_);
	problem.SetParameterUpperBound(&c_y, 0, 256 + principal_boundary_);
	//problem.SetParameterLowerBound(&c_x, 0, 1348);
	//problem.SetParameterLowerBound(&c_y, 0, 700);
	//problem.SetParameterUpperBound(&c_x, 0, 1548);
	//problem.SetParameterUpperBound(&c_y, 0, 1100);
	ceres::Solver::Options options;
	options.max_num_iterations = 500;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = false;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//std::cout << summary.FullReport() << "\n";
	K_(0, 2) = c_x;
	K_(1, 2) = c_y;
	if ((summary.termination_type == ceres::TerminationType::CONVERGENCE))
	{
#if ENABLE_LOG
		std::cout << std::endl;
		std::cout << "Final cost: " << summary.final_cost << std::endl;
		std::cout << "Initial cost" << summary.initial_cost << std::endl;
		std::cout << "Termination type: " << summary.termination_type << std::endl;
		std::cout << std::endl;
		std::cout << K_ << std::endl;
#endif
		K_(0, 2) = c_x;
		K_(1, 2) = c_y;

		std::cout << K_ << std::endl;
		return true;
	}
	else
	{
		std::cout << std::endl;
		std::cout << "We did not converge during principal optimatization" << std::endl;
		std::cout << "F: " << f_x << std::endl;
		std::cout << "Principals: " << c_x << ", " << c_y << std::endl;
		std::cout << std::endl;
		return false;
	}
	return false;
}
void PrincipalCalibrator::addFeatures(const std::vector<std::pair<Feature, Feature>>& feature_pairs)
{

	std::vector<std::pair<Feature, Feature>> usable_feature_pairs;
	for (const auto& feature_pair : feature_pairs)
	{
		auto vec = feature_pair.first.feature_point - feature_pair.second.feature_point;
		if (abs(vec.x) < 2.0)
		{
			continue;
		}
		if (abs(vec.y) < 1.0)
		{
			continue;
		}
		auto point_distance = cv::norm(feature_pair.first.feature_point - feature_pair.second.feature_point);
		//std::cout << "The distance is: " << point_distance << std::endl;
		if (point_distance < minimum_feature_distance_)
		{
			continue;
		}
		if (bins_.doesFeaturePairFitInBins(feature_pair))
		{
			usable_feature_pairs.emplace_back(feature_pair);
		}
	}
	number_of_featurepairs += usable_feature_pairs.size();
	list_of_feature_pairs_.emplace_back(usable_feature_pairs);
	bins_.printCapacityOfBins();
	
}
double PrincipalCalibrator::calculatePrincipalResidualLoss(const std::vector<std::vector<std::pair<Feature, Feature>>>& feature_paires, const Eigen::Matrix3d& K)
{
	double final_d = 0;
	for (size_t set_number = 0; set_number < feature_paires.size(); set_number++)
	{
		for (size_t f_p = 0; f_p < feature_paires[set_number].size(); f_p++)
		{
			Eigen::Matrix3d rotation_1 = feature_paires[set_number][f_p].second.M.block<3, 3>(0, 0);
			Eigen::Vector3d translation_1 = feature_paires[set_number][f_p].second.M.block<3, 1>(0, 3);

			Eigen::Matrix3d rotation_0 = Eigen::Matrix3d::Identity();
			Eigen::Vector3d translation_0 = Eigen::Vector3d::Zero();

			Eigen::Vector3d P_i = { feature_paires[set_number][f_p].first.feature_point.x, feature_paires[set_number][f_p].first.feature_point.y,1.0 };
			Eigen::Vector3d P_i_star = { feature_paires[set_number][f_p].second.feature_point.x, feature_paires[set_number][f_p].second.feature_point.y,1.0 };

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
	return final_d;
}

void PrincipalCalibrator::exportShapeOfProblemSpace(const std::string& sequence_number, const std::string& sequence_length)
{
	std::ofstream datafile;
	datafile.open("principal_cost" + sequence_number + "_" + sequence_length + ".py");
	datafile << "res = [";
	Eigen::Matrix3d K_build = Eigen::Matrix3d::Identity();
	K_build(0, 0) = K_(0, 0);
	K_build(1, 1) = K_(1, 1);
	K_build(0, 2) = 0;
	K_build(1, 2) = 0;
	int principal_x = 0;
	int principal_y = 0;
	while (principal_x < 900)
	{

		while (principal_y < 900)
		{
			datafile << "[";
			K_build(0, 2) = principal_x;
			K_build(1, 2) = principal_y;
			double final_res = 0;
			final_res = calculatePrincipalResidualLoss(list_of_feature_pairs_, K_build);
			datafile << principal_x << "," << principal_y << "," << final_res;
			if (principal_x < 900 - 4 || principal_y < 900 - 4)
			{
				datafile << "],";
			}
			else
			{
				datafile << "]";
			}
			principal_y += 5;
		}
		principal_x += 5;
		principal_y = 0;
	}
	datafile << "]";
}

Eigen::Matrix3d PrincipalCalibrator::getIntrinsicMatrix()
{
	return K_;
}

void PrincipalCalibrator::setIntrinsicMatrix(const Eigen::Matrix3d& K)
{
	K_(0, 0) = K(0,0);
	K_(1, 1) = K(1,1);
	K_(0, 2) = K(0,2);
	K_(1, 2) = K(1, 2);
}
