#pragma once
#include "distortion_calibrator.h"
#include <iostream>
#include <fstream>
#include <string>

RadialDistortionCalibrator::RadialDistortionCalibrator(const cv::Rect& roi, int limit_in_a_bin, int min_number_of_bin, int max_number_of_bin, double minimum_feature_distance) :
	bins_(roi, limit_in_a_bin, min_number_of_bin, max_number_of_bin), minimum_feature_distance_(minimum_feature_distance)
{
}

bool RadialDistortionCalibrator::calibrate()
{
	if (number_of_featurepairs < 10)
	{
		return false;
	}
	double f_x = K_(0, 0);
	double f_y = K_(1, 1);
	double c_x = K_(0, 2);
	double c_y = K_(1, 2);

	cv::Mat cameraMatrix(3, 3, CV_32F);
	cv::eigen2cv(K_, cameraMatrix);
	std::map<double, double> residual_coefficient_map_k1;
	std::map<double, double> residual_coefficient_map_k2;
	for (size_t i = 0; i < 2; i++)
	{
		double k1 = -1.0;
		while (k1 < 1.0)
		{
			double final_d = 0;
			cv::Mat distCoeffs = (cv::Mat_<float>(5, 1) << k1, D_.k2, 0, 0, 0);
			final_d = calculateDistortionResidualLoss(list_of_feature_pairs_, cameraMatrix, distCoeffs);
			residual_coefficient_map_k1.emplace(final_d, k1);
			k1 += 0.01;
		}
		D_.k1 = (*residual_coefficient_map_k1.begin()).second;
		std::cout << "K1: " << D_.k1 << std::endl;
		double k2 = -0.5;
		while (k2 < 0.5)
		{
			double final_d = 0;
			cv::Mat distCoeffs = (cv::Mat_<float>(5, 1) << D_.k1, k2, 0, 0, 0);
			final_d = calculateDistortionResidualLoss(list_of_feature_pairs_, cameraMatrix, distCoeffs);
			residual_coefficient_map_k2.emplace(final_d, k2);
			k2 += 0.01;
		}
		D_.k2 = (*residual_coefficient_map_k2.begin()).second;
		std::cout << "K2: " << D_.k2 << std::endl;
		residual_coefficient_map_k1.clear();
		residual_coefficient_map_k2.clear();
	}
	double k_1 = D_.k1;
	ceres::Problem problem;
	ceres::CostFunction* cost_function = new ceres::NumericDiffCostFunction<RadialDistortionResidual, ceres::CENTRAL, 1, 1>(
		new RadialDistortionResidual(list_of_feature_pairs_, K_));

	problem.AddResidualBlock(cost_function, nullptr, &k_1);
	problem.SetParameterUpperBound(&k_1, 0, 1.0);
	problem.SetParameterLowerBound(&k_1, 0, -1.0);
	ceres::Solver::Options options;
	options.max_num_iterations = 150;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = false;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	if ((summary.termination_type == ceres::TerminationType::CONVERGENCE))
	{
		std::cout << std::endl;
		std::cout << "Final cost: " << summary.final_cost << std::endl;
		std::cout << "Initial cost" << summary.initial_cost << std::endl;
		std::cout << "Termination type: " << summary.termination_type << std::endl;
		std::cout << std::endl;
	}
	else
	{
		std::cout << "We did not converge during optimatization" << std::endl;
		std::cout << "Final cost: " << summary.final_cost << std::endl;
		std::cout << "Initial cost" << summary.initial_cost << std::endl;
	}
	D_.k1 = k_1;
	D_.printDistortionParams();
	return false;
}

bool RadialDistortionCalibrator::calibrateWithOdometry()
{
	/*double average_dist = 0;
	for (const auto& pairs : list_of_feature_pairs_)
	{
		double current_average = 0;
		for (const auto& pair : pairs)
		{
			auto direction_vec = pair.second.feature_point - pair.first.feature_point;
			auto dist = cv::norm(direction_vec);
			current_average += dist;
		}
		current_average /= pairs.size();
		std::cout << "The current average is: " << current_average << std::endl;
		average_dist += current_average;
	}
	average_dist /= list_of_feature_pairs_.size();
	std::cout << "The complete average feature distance for all feature pairs is: " << average_dist << std::endl;*/
	double boundary = optimizeDistortionRange(true, false);
	double boundary2 = optimizeDistortionRange(false, true);
	double k1 = D_.k1;
	double k2 = D_.k2;
	std::cout << "The preoptimized distortion parameters" << std::endl;
	D_.printDistortionParams();
	double upper_boundary = k1 + boundary;
	double lower_boundary = k1 - boundary;
	double upper_boundary2 = k2 + boundary2;
	double lower_boundary2 = k2 - boundary2;
	std::cout << "The k1 boundary" << std::endl;
	std::cout << upper_boundary << ", " << lower_boundary << std::endl;
	std::cout << "The k2 boundary" << std::endl;
	//std::cout << upper_boundary2 << ", " << lower_boundary2 << std::endl;
	ceres::Problem problem;
	ceres::CostFunction* cost_function = new ceres::NumericDiffCostFunction<RadialDistortionWithOdometry, ceres::CENTRAL, 1, 1, 1>(
		new RadialDistortionWithOdometry(list_of_feature_pairs_, K_));

	problem.AddResidualBlock(cost_function, nullptr, &k1, &k2); //&f_x, &f_y , &c_x, &c_y
	problem.SetParameterLowerBound(&k1, 0, lower_boundary);
	problem.SetParameterUpperBound(&k1, 0, upper_boundary);
	problem.SetParameterLowerBound(&k2, 0, 0);
	problem.SetParameterUpperBound(&k2, 0, std::abs(D_.k1));

	ceres::Solver::Options options;
	//options.use_nonmonotonic_steps = true;
	//options.max_num_iterations = 250;
	//options.preconditioner_type = ceres::SCHUR_JACOBI;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	options.minimizer_progress_to_stdout = true;
	/*options.use_inner_iterations = true;
	options.max_num_iterations = 100;*/
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	if ((summary.termination_type == ceres::TerminationType::CONVERGENCE))
	{
#if ENABLE_LOG
		std::cout << std::endl;
		std::cout << "Final cost: " << summary.final_cost << std::endl;
		std::cout << "Initial cost" << summary.initial_cost << std::endl;
#endif
		D_.k1 = k1;
		D_.k2 = k2;
		D_.printDistortionParams();
		return true;
	}
	else
	{
#if ENABLE_LOG
		std::cout << std::endl;
		std::cout << "We did not converge during optimatization" << std::endl;
		std::cout << "K1: " << k1 << std::endl;
		std::cout << std::endl;
#endif
		return false;
	}
	return false;
}

bool RadialDistortionCalibrator::calibrateWithOdometryK2()
{
	
	if (number_of_featurepairs < 500)
	{
		return false;
	}
	double boundary = optimizeDistortionRange(false,true);
	double k2 = D_.k2;
	double upper_boundary = k2 + boundary;
	double lower_boundary = k2 - boundary;
	std::cout << upper_boundary << ", " << lower_boundary << std::endl;
	ceres::Problem problem;
	ceres::CostFunction* cost_function = new ceres::NumericDiffCostFunction<RadialDistortionWithOdometryK2, ceres::CENTRAL, 1, 1>(
		new RadialDistortionWithOdometryK2(list_of_feature_pairs_, K_, D_.k1));

	problem.AddResidualBlock(cost_function, nullptr, &k2); //&f_x, &f_y , &c_x, &c_y
	problem.SetParameterLowerBound(&k2, 0, lower_boundary);
	problem.SetParameterUpperBound(&k2, 0, upper_boundary);

	ceres::Solver::Options options;
	options.max_num_iterations = 250;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	if ((summary.termination_type == ceres::TerminationType::CONVERGENCE))
	{
#if ENABLE_LOG
		std::cout << std::endl;
		std::cout << "Final cost: " << summary.final_cost << std::endl;
		std::cout << "Initial cost" << summary.initial_cost << std::endl;
#endif
		D_.k2 = k2;
		D_.printDistortionParams();
		return true;
	}
	else
	{
#if ENABLE_LOG
		std::cout << std::endl;
		std::cout << "We did not converge during optimatization" << std::endl;
		std::cout << "K1: " << k1 << std::endl;
		std::cout << std::endl;
#endif
		return false;
	}
	return false;
}

void RadialDistortionCalibrator::addFeatures(const std::vector<std::pair<Feature, Feature>>& feature_pairs)
{
	std::vector<std::pair<Feature, Feature>> usable_feature_pairs;
	for (const auto& feature_pair : feature_pairs)
	{
		auto direction_vec = feature_pair.second.feature_point - feature_pair.first.feature_point;
		if (abs(direction_vec.x) < 3.0)
		{
			continue;
		}
		if (abs(direction_vec.y) < 3.0)
		{
			continue;
		}

		//auto distance_to_center = calculateLinePointDistance(direction_vec, feature_pair.second.feature_point, cv::Point2d(K_(0, 2), K_(1, 2)));
		//if (distance_to_center < 10)
		//{
		//	continue;
		//}
		auto dist = cv::norm(direction_vec);
 		//std::cout << "feature line distance to the distortion center: " << distance_to_center << ", line length: " << dist << std::endl;
		
		if (dist < minimum_feature_distance_)
		{
			continue;
		}
		/*if (dist > 80)
		{
			continue;
		}*/
		//if (abs(cv::norm(feature_pair.first.feature_point - cv::Point2d(K_(0, 2), K_(1, 2)))) < 100)
		//{
		//	continue;
		//}
		/*auto derivative = abs(direction_vec.x / direction_vec.y);
		if (derivative < 0.9 || derivative > 1.1)
		{
			continue;
		}*/
		
		
		if (bins_.doesFeaturePairFitInBins(feature_pair))
		{
			usable_feature_pairs.emplace_back(feature_pair);
		}


	}
	number_of_featurepairs += usable_feature_pairs.size();
	list_of_feature_pairs_.emplace_back(usable_feature_pairs);
}



Eigen::Matrix3d RadialDistortionCalibrator::getIntrinsicMatrix()
{
	return K_;
}

void RadialDistortionCalibrator::setIntrinsicMatrix(const Eigen::Matrix3d& K)
{
	K_(0, 0) = K(0, 0);
	K_(1, 1) = K(1, 1);
	K_(0, 2) = K(0, 2);
	K_(1, 2) = K(1, 2);
}

void RadialDistortionCalibrator::calculateDistortionProblemSpace()
{
	std::ofstream distortionfile;
	distortionfile.open("odometry_distortion_cost_05_150.py");
	distortionfile << "res = [";
	double k = -1.0;
	while (k < 1.0)
	{
		double final_d = 0;
		cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) << -0.37, k, 0, 0, 0);
		final_d = calculateDistortionOdometryResidualLoss(this->list_of_feature_pairs_, K_, distCoeffs);
		distortionfile << "[" << k << "," << final_d << "],";
		k += 0.01;
	}
	distortionfile << "]";
	distortionfile.close();

}

double RadialDistortionCalibrator::optimizeDistortionRange(bool optimize_k1, bool optimize_k2)
{
	if (optimize_k1)
	{
		std::map<double, double> residual_coefficient_map_k1;
		double starting_value = (D_.k1 < 0.0) ? 2*D_.k1 : (- 2 * D_.k1);
		double k_1 = (D_.k1 == 0.0) ? -1.0 : 2*starting_value;
		while (k_1 < 1.0)
		{
			double final_d = 0;
			cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) << k_1, D_.k2, 0, 0, 0);
			final_d = calculateDistortionOdometryResidualLoss(this->list_of_feature_pairs_, K_, distCoeffs);

			residual_coefficient_map_k1.emplace(final_d, k_1);
			k_1 += 0.01;
		}
		D_.k1 = (*residual_coefficient_map_k1.begin()).second;
		return (abs(D_.k1) / 100) * 75;
	}
	if (optimize_k2)
	{
		if (D_.k1 == 0)
		{
			return 0.0;
		}
		std::map<double, double> residual_coefficient_map_k2;
		double k_2 = (D_.k1 < 0.0) ? 0.0 : -D_.k1;
		while (k_2 < abs(D_.k1))
		{
			double final_d = 0;
			cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) << D_.k1, k_2, 0, 0, 0);
			final_d = calculateDistortionOdometryResidualLoss(this->list_of_feature_pairs_, K_, distCoeffs);

			residual_coefficient_map_k2.emplace(final_d, k_2);
			k_2 += 0.01;
		}
		D_.k2 = (*residual_coefficient_map_k2.begin()).second;
		return (abs(D_.k2) / 100) * 25;
	}
	
}

void RadialDistortionCalibrator::drawBinCapacity(cv::Mat image, const std::string& sequence_number)
{
	bins_.drawBins(image, sequence_number);
}



double RadialDistortionCalibrator::calculateDistortionResidualLoss(const std::vector<std::vector<std::pair<Feature, Feature>>>& feature_paires, const cv::Mat& K, const cv::Mat& distortion_coeffs)
{
	double loss = 0;
	for (size_t set_number = 0; set_number < feature_paires.size(); set_number++)
	{
		std::vector<cv::Point2f> inputDistortedPoints1;
		std::vector<cv::Point2f> inputDistortedPoints2;
		std::vector<cv::Point2f> outputUndistortedPoints1;
		std::vector<cv::Point2f> outputUndistortedPoints2;
		for (size_t f_p = 0; f_p < feature_paires[set_number].size(); f_p++)
		{
			inputDistortedPoints1.emplace_back(feature_paires[set_number][f_p].first.feature_point);
			inputDistortedPoints2.emplace_back(feature_paires[set_number][f_p].second.feature_point);
		}
		if (inputDistortedPoints1.empty() || inputDistortedPoints2.empty())
		{
			continue;
		}
		cv::undistortImagePoints(inputDistortedPoints1, outputUndistortedPoints1, K, distortion_coeffs);
		cv::undistortImagePoints(inputDistortedPoints2, outputUndistortedPoints2, K, distortion_coeffs);

		bool usable_point = true;
		std::vector<cv::Point2f> outputUndistortedPoints11;
		std::vector<cv::Point2f> outputUndistortedPoints22;
		for (size_t i = 0; i < outputUndistortedPoints1.size(); i++)
		{
			if (outputUndistortedPoints1[i].x < 0 || outputUndistortedPoints1[i].y < 0 || outputUndistortedPoints1[i].x > 1392 || outputUndistortedPoints1[i].y>512)
			{
				usable_point = false;
			}
			else if (outputUndistortedPoints2[i].x < 0 || outputUndistortedPoints2[i].y < 0 || outputUndistortedPoints2[i].x > 1392 || outputUndistortedPoints2[i].y>512)
			{
				usable_point = false;
			}
			else if (!usable_point)
			{
				outputUndistortedPoints1.erase(outputUndistortedPoints1.begin() + i);
				outputUndistortedPoints2.erase(outputUndistortedPoints2.begin() + i);
				usable_point = true;
			}
		}
		for (size_t i = 0; i < outputUndistortedPoints1.size(); i+=1)
		{
			outputUndistortedPoints11.emplace_back(outputUndistortedPoints1[i]);
			outputUndistortedPoints22.emplace_back(outputUndistortedPoints2[i]);
		}
		if (outputUndistortedPoints11.size() < 9)
		{
			continue;
		}
		cv::Mat fundamental_matrix = cv::findFundamentalMat(outputUndistortedPoints11, outputUndistortedPoints22, cv::FM_8POINT);
		if (fundamental_matrix.empty() || fundamental_matrix.cols != 3 || fundamental_matrix.rows != 3)
		{
			continue;
		}
		std::vector<Eigen::Vector3f> input1;
		input1.reserve(outputUndistortedPoints1.size());
		std::vector<Eigen::Vector3f> input2;
		input2.reserve(outputUndistortedPoints2.size());
		Eigen::Matrix3f fundamental;

		for (size_t i = 0; i < outputUndistortedPoints1.size(); i++)
		{
			input1.emplace_back(Eigen::Vector3f(outputUndistortedPoints1[i].x, outputUndistortedPoints1[i].y, 1));
			input2.emplace_back(Eigen::Vector3f(outputUndistortedPoints2[i].x, outputUndistortedPoints2[i].y, 1));
		}
		//std::cout << fundamental_matrix << std::endl;
		cv::cv2eigen(fundamental_matrix, fundamental);
		for (size_t i = 0; i < input2.size(); i++)
		{
			auto ret = abs(input2[i].transpose() * fundamental * input1[i]);
			//std::cout << ret << std::endl;
			//std::cout << outputUndistortedPoints1[i] << ", " << outputUndistortedPoints2[i] << std::endl;
			loss += abs(input2[i].transpose() * fundamental * input1[i]);
		}
	}
	return loss;
}

double RadialDistortionCalibrator::calculateDistortionOdometryResidualLoss(const std::vector<std::vector<std::pair<Feature, Feature>>>& feature_paires, const Eigen::Matrix3d& K, const cv::Mat& distortion_coeffs)
{
	double loss = 0;
	cv::Mat cameraMatrix(3, 3, CV_16F);
	cv::eigen2cv(K, cameraMatrix);
	for (size_t set_number = 0; set_number < feature_paires.size(); set_number++)
	{
		for (size_t f_p = 0; f_p < feature_paires[set_number].size(); f_p++)
		{
			Eigen::Matrix3d rotation_1 = feature_paires[set_number][f_p].second.M.block<3, 3>(0, 0);
			Eigen::Vector3d translation_1 = feature_paires[set_number][f_p].second.M.block<3, 1>(0, 3);

			Eigen::Matrix3d rotation_0 = Eigen::Matrix3d::Identity();
			Eigen::Vector3d translation_0 = Eigen::Vector3d::Zero();

			std::vector<cv::Point2f> inputDistortedPoints1;
			std::vector<cv::Point2f> inputDistortedPoints2;
			std::vector<cv::Point2f> outputUndistortedPoints1;
			std::vector<cv::Point2f> outputUndistortedPoints2;

			inputDistortedPoints1.emplace_back(feature_paires[set_number][f_p].first.feature_point);
			inputDistortedPoints2.emplace_back(feature_paires[set_number][f_p].second.feature_point);

			//cv::undistortImagePoints(inputDistortedPoints1, outputUndistortedPoints1, cameraMatrix, distCoeffs);
			//cv::undistortImagePoints(inputDistortedPoints2, outputUndistortedPoints2, cameraMatrix, distCoeffs);
			cv::undistortPoints(inputDistortedPoints1, outputUndistortedPoints1, cameraMatrix, distortion_coeffs, cv::noArray(), cameraMatrix);
			cv::undistortPoints(inputDistortedPoints2, outputUndistortedPoints2, cameraMatrix, distortion_coeffs, cv::noArray(), cameraMatrix);

			if (outputUndistortedPoints1.front().x < 0 || outputUndistortedPoints1.front().y < 0 || outputUndistortedPoints1.front().x > 1391 || outputUndistortedPoints1.front().y > 512)
			{
				continue;
			}
			if (outputUndistortedPoints2.front().x < 0 || outputUndistortedPoints2.front().y < 0 || outputUndistortedPoints2.front().x > 1391 || outputUndistortedPoints2.front().y > 512)
			{
				continue;
			}
			Eigen::Vector3d P_i = { outputUndistortedPoints1.front().x, outputUndistortedPoints1.front().y,1.0 };
			Eigen::Vector3d P_i_star = { outputUndistortedPoints2.front().x, outputUndistortedPoints2.front().y,1.0 };

			Eigen::Vector3d ray_direction0 = (rotation_0 * K.inverse() * P_i);
			Eigen::Vector3d ray_direction1 = (rotation_1 * K.inverse() * P_i_star);
			Eigen::Vector3d N = ray_direction0.cross(ray_direction1);

			Eigen::Vector3d PQ = translation_1 - translation_0; //trans_0 -> trans_1 vector

			Eigen::Vector3d P1 = (ray_direction0)+translation_0;
			Eigen::Vector3d P2 = (ray_direction1)+translation_1;
			Eigen::Vector3d V = P1.cross(P2);
			double d = std::abs(PQ.dot(V)) / V.norm();
			loss += d;
		}
	}
	return loss;
}

double RadialDistortionCalibrator::calculateLinePointDistance(const cv::Point2d& direction_vector, const cv::Point2d& point_on_the_line, const cv::Point2d& middle_point)
{
	auto n = cv::Point2d(direction_vector.y, -direction_vector.x);
	auto PQ = middle_point - point_on_the_line;
	auto distance = abs(PQ.dot(n) / cv::norm(n));
	return distance;

}

void RadialDistortionCalibrator::drawFeatureLines(cv::Mat image)
{
	for (size_t set_number = 0; set_number < list_of_feature_pairs_.size(); set_number++)
	{
		for (size_t f_p = 0; f_p < list_of_feature_pairs_[set_number].size(); f_p++)
		{
			auto first_point = list_of_feature_pairs_[set_number][f_p].first.feature_point;
			auto second_point = list_of_feature_pairs_[set_number][f_p].second.feature_point;
			cv::line(image, first_point, second_point, cv::Scalar(0, 250, 0), 1);
		}
	}
	cv::imwrite("Feature_lines.png", image);
}
