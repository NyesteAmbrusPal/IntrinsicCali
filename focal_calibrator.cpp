#pragma once
#include "focal_calibrator.h"
#include <iostream>
#include <fstream>
#include <string>
#define ENABLE_LOG 1

FocalCalibrator::FocalCalibrator(const cv::Rect& roi, int limit_in_a_bin, int min_number_of_bin, int max_number_of_bin, double minimum_feature_distance) :
	bins_(roi, limit_in_a_bin, min_number_of_bin, max_number_of_bin), minimum_feature_distance_(minimum_feature_distance)
{

}

bool FocalCalibrator::calibrate()
{
	std::cout << "Found feature pairs: " << number_of_featurepairs << std::endl;
	double f_x = K_(0, 0);
	double f_y = K_(1, 1);
	double c_x = K_(0, 2);
	double c_y = K_(1, 2);
	ceres::Problem problem;
	ceres::CostFunction* cost_function = new ceres::NumericDiffCostFunction<FocalResiudal, ceres::CENTRAL, 1, 1>(
		new FocalResiudal(list_of_feature_pairs_, c_x, c_y));

	problem.AddResidualBlock(cost_function, nullptr, &f_x); //&f_x, &f_y , &c_x, &c_y
	//problem.SetParameterLowerBound(&f_x, 0, 100);
	problem.SetParameterLowerBound(&f_x, 0, 100);
	//problem.SetParameterUpperBound(&f_x, 0, 4750);
	problem.SetParameterUpperBound(&f_x, 0, 1750);

	ceres::Solver::Options options;
	options.max_num_iterations = 500;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = false;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	if ((summary.termination_type == ceres::TerminationType::CONVERGENCE) && f_x < 4750 && f_x > 500)
	{
#if ENABLE_LOG
		std::cout << std::endl;
		std::cout << "Final cost: " << summary.final_cost << std::endl;
		std::cout << "Initial cost" << summary.initial_cost << std::endl;
#endif
		K_(0, 0) = f_x;
		K_(1, 1) = f_x;
		K_(0, 2) = c_x;
		K_(1, 2) = c_y;

		return true;
	}
	else
	{
#if ENABLE_LOG
		std::cout << std::endl;
		std::cout << "We did not converge during focal optimatization" << std::endl;
		std::cout << "F: " << f_x << std::endl;
		std::cout << "Principals: " << c_x << ", " << c_y << std::endl;
		std::cout << std::endl;
#endif
		return false;
	}
	return false;
}

void FocalCalibrator::addFeatures(const std::vector<std::pair<Feature, Feature>>& feature_pairs)
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
		auto dist = cv::norm(feature_pair.first.feature_point - feature_pair.second.feature_point);
		if (dist < minimum_feature_distance_)
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

Eigen::Matrix3d FocalCalibrator::getIntrinsicMatrix()
{
	return K_;
}

void FocalCalibrator::setIntrinsicMatrix(const Eigen::Matrix3d& K)
{
	K_(0, 0) = K(0, 0);
	K_(1, 1) = K(1, 1);
	K_(0, 2) = K(0, 2);
	K_(1, 2) = K(1, 2);
}

double FocalCalibrator::calculateFocalResidualLoss(const std::vector<std::vector<std::pair<Feature, Feature>>>& feature_paires, const Eigen::Matrix3d& K)
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

void FocalCalibrator::exportShapeOfProblemSpace(const std::string& sequence_number, const std::string& sequence_length)
{
	std::ofstream datafile;
	datafile.open("focal_cost_" + sequence_number + "_" + sequence_length + ".py");
	datafile << "res = [";
	Eigen::Matrix3d K_build = Eigen::Matrix3d::Identity();
	K_build(0, 0) = 0;
	K_build(1, 1) = 0;
	K_build(0, 2) = K_(0, 2);
	K_build(1, 2) = K_(1, 2);
	int focal_x = 307;
	int focal_y = 307;
	while (focal_x < 1107)
	{
		
		while (focal_y < 1107)
		{
			datafile << "[";
			K_build(0, 0) = focal_x;
			K_build(1, 1) = focal_y;
			double final_res = 0;
			final_res = calculateFocalResidualLoss(list_of_feature_pairs_, K_build);
			datafile << focal_x << "," << focal_y << "," << final_res;
			if (focal_x < 1107-4 || focal_y < 1107 -4)
			{
				datafile << "],";
			}
			else
			{
				datafile << "]";
			}
			focal_y+=4;
		}
		focal_x+=4;
		focal_y = 307;
	}
	datafile << "]";
}
