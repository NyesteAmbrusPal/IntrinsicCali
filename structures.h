#pragma once
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Dense>
#include <deque>
#include <vector>
#include "functional_impl.h"
#include <opencv2/core/eigen.hpp>


struct Feature
{
	cv::Point2d feature_point;
	size_t image_id = 0;
	Eigen::Matrix4d M = Eigen::Matrix4d::Identity();

	bool operator==(const Feature& other)
	{
		if ((other.feature_point == this->feature_point))
		{
			return true;
		}
		return false;
	}
	bool operator!=(const Feature& other)
	{
		return !(*this == other);
	}
};

struct Bin
{
	cv::Point2i top_left = { 0,0 };
	cv::Point2i bottom_right = { 0,0 };
	int num_of_features = 0;
	std::deque<size_t> id;
	//std::vector< std::pair<Feature, Feature>> feature_paires;

};

struct Bins
{
public:
	Bins(cv::Rect roi, int limit_in_a_bin, int min_number, int max_number) :
		roi_(roi), limit_in_bins_(limit_in_a_bin)
	{
		cv::Point2i top_left = roi_.tl();
		cv::Point2i bottom_rigth = roi_.br();
		int divisor_x = find_divisor(bottom_rigth.x, top_left.x, min_number, max_number);//15 20
		int divisor_y = find_divisor(bottom_rigth.y, top_left.y, min_number, max_number);//8 13
		int width_of_bin = (bottom_rigth.x - top_left.x) / divisor_x;
		int height_of_bin = (bottom_rigth.y - top_left.y) / divisor_y;
		for (int i = top_left.x; i < bottom_rigth.x; i += width_of_bin)
		{
			for (int j = top_left.y; j < bottom_rigth.y; j += height_of_bin)
			{
				Bin current_bin;
				current_bin.top_left = { i,j };
				current_bin.bottom_right = { i + width_of_bin,j + height_of_bin };
				bins_.push_back(current_bin);
			}
		}
	}
	bool doesFeaturePairFitInBins(const std::pair<Feature, Feature>& potentional_feature)
	{
		if (!pointIsInArea(potentional_feature))
		{
			return false;
		}
		for (int j = 0; j < bins_.size(); j++)
		{
			bool isIn = pointIsInBin(bins_[j], potentional_feature);
			if (((bins_[j].num_of_features <= limit_in_bins_) && isIn))
			{
				/*if (std::find(bins_[j].id.begin(), bins_[j].id.end(), potentional_feature.second.image_id) != bins_[j].id.end())
				{
					continue;
				}*/
				//std::cout << "The current point which is stored in the " << j << "-th bin has the coords: " << potentional_feature.first.feature_point << ", " <<
				//	potentional_feature.second.feature_point << std::endl;
				bins_[j].num_of_features++;
				bins_[j].id.emplace_back(potentional_feature.second.image_id);
				return true;
			}
			else
			{
				continue;
			}
		}
		return false;
	}
	void printCapacityOfBins()
	{
		size_t sum_of_fulls = 0;
		size_t sum_of_all_feature = 0;
		for (size_t i = 0; i < bins_.size(); i++)
		{
			if (limit_in_bins_ < bins_[i].num_of_features)
			{
				sum_of_fulls++;
				
			}
			sum_of_all_feature += (bins_[i].num_of_features);
		}
		double fullnes = (sum_of_fulls / double(bins_.size()));
		std::cout << "Currently we have found " << sum_of_all_feature << " feature pairs" << std::endl;
		std::cout << "Currently " << fullnes * 100 << "% of the bins are filled" << std::endl;
	}

	void drawBins(cv::Mat image, const std::string& sequence_number)
	{
		for (int i = 0; i < bins_.size(); i++)
		{
			auto middle = (bins_[i].top_left);
			if (bins_[i].num_of_features >= limit_in_bins_)
			{
				cv::rectangle(image, bins_[i].top_left, bins_[i].bottom_right, cv::Scalar(0, 255, 0), 2, 8, 0);
				cv::putText(image, std::to_string(i), middle, cv::FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2);
			}
			else
			{
				cv::rectangle(image, bins_[i].top_left, bins_[i].bottom_right, cv::Scalar(0, 0, 255), 2, 8, 0);
				cv::putText(image, std::to_string(i), middle, cv::FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2);
			}
		}
		cv::imshow("Distribution", image);
		cv::waitKey(1);
		cv::imwrite("Distribution" + sequence_number + ".png", image);
	}

	double getCapacityOfBins()
	{
		size_t sum_of_fulls = 0;
		size_t sum_of_all_feature = 0;
		for (size_t i = 0; i < bins_.size(); i++)
		{
			if (limit_in_bins_ < bins_[i].num_of_features)
			{
				sum_of_fulls++;
				sum_of_all_feature += (bins_[i].num_of_features);
			}
		}
		return (sum_of_fulls / double(bins_.size()));
	}


private:
	bool pointIsInBin(const Bin& current_bin, const std::pair<Feature, Feature>& feature_pair)
	{
		if ((current_bin.top_left.x <= feature_pair.second.feature_point.x) && (current_bin.top_left.y <= feature_pair.second.feature_point.y)
			&& (current_bin.bottom_right.x >= feature_pair.second.feature_point.x) && (current_bin.bottom_right.y >= feature_pair.second.feature_point.y))
		{
			return true;
		}
		return false;
	}
	bool pointIsInArea(const std::pair<Feature, Feature>& feature_pair)
	{
		int max_width = roi_.br().x;
		int max_height = roi_.br().y;
		int min_width = roi_.tl().x;
		int min_height = roi_.tl().y;
		if (feature_pair.first.feature_point.x > max_width || feature_pair.first.feature_point.x < min_width)
		{
			return false;
		}
		if (feature_pair.first.feature_point.y > max_height || feature_pair.first.feature_point.y < min_height)
		{
			return false;
		}
		if (feature_pair.second.feature_point.x > max_width || feature_pair.second.feature_point.x < min_width)
		{
			return false;
		}
		if (feature_pair.second.feature_point.y > max_height || feature_pair.second.feature_point.y < min_height)
		{
			return false;
		}
		return true;
	}
	cv::Rect roi_;
	int limit_in_bins_;
	std::vector<Bin> bins_;
};

struct DistrotionParams
{
	DistrotionParams() = default;

	double k1 = 0;
	double k2 = 0;
	double k3 = 0;
	double p1 = 0;
	double p2 = 0;
	void printDistortionParams()
	{
		std::cout << " K1: " << k1 << " K2: " << k2 << " K3: " << k3 << " P1: " << p1 << " P2: " << p2 << std::endl;
	}
	std::vector<double> getDistortionsInVector()
	{
		std::vector<double> ret;
		ret.emplace_back(k1);
		ret.emplace_back(k2);
		ret.emplace_back(p1);
		ret.emplace_back(p2);
		ret.emplace_back(k3);
		return ret;
	}
};

struct DistortionLine
{
	void addStarterPoints(const std::pair<Feature, Feature>& current_feature)
	{
		double slope = (current_feature.second.feature_point.y - current_feature.first.feature_point.y) / (double)(current_feature.second.feature_point.x - current_feature.first.feature_point.x);
		if (current_points.empty())
		{
			current_points.emplace_back(current_feature.first);
			current_points.emplace_back(current_feature.second);
		}
	}
	bool checkPointIsConsequtive(const Feature& current_feature)
	{
		if (current_points.empty())
		{
			return false;
		}
		auto diff = current_feature.feature_point - current_points.front().feature_point;
		cv::norm(diff);
		if (cv::norm(diff) < 0.001)
		{
			return true;
		}
		return false;
	}
	bool isOutOfIDRange(size_t current_id)
	{
		if (current_points.empty())
		{
			return false;
		}
		if (current_points.back().image_id < current_id)
		{
			return true;
		}
		return false;
	}
	bool isItEmpty()
	{
		return current_points.empty();
	}
	void addToEnd(const Feature& current_feature)
	{
		if (std::find(current_points.begin(), current_points.end(), current_feature) != current_points.end())
		{
			return;
		}
		current_points.emplace_back(current_feature);
	}

	double getlinelength() const
	{
		if (!current_points.empty())
		{
			return cv::norm(current_points.front().feature_point - current_points.back().feature_point);
		}
		return 0.0;
	}
	std::vector<cv::Point2d> getPoints()
	{
		std::vector<cv::Point2d> ret;
		for (const auto& point : current_points)
		{
			ret.push_back(point.feature_point);
		}
		return ret;
	}
	std::vector<Feature> current_points;
};




