#include "parser.h"
#include <stdio.h>
#include <string.h>

void ImageSequence::loadPoses(const std::string& file_name, size_t length) {
    std::string line;
    std::ifstream file(file_name);
    size_t line_counter = 0;
    while (getline(file, line))
    {
        if (line_counter >= length)
        {
            break;
        }
        std::stringstream iss(line);
        std::string item;
        int i = 0;
        int j = 0;
        Pose Position(0,Eigen::Matrix4d::Identity());
        Position.id = line_counter;
        while (iss >> item)
        {
            double d;
            if (std::istringstream(item) >> d)
            {
                Position.pose_of_0camera(j, i) = d;
                if (i % 3 == 0 && i != 0)
                {
                    i = 0;
                    j++;
                }
                else
                {
                    i++;
                }
                
            }
        }
        poses.emplace_back(Position);
        j = 0;
        //std::cout << ""R | t matrix " << line_counter << std::endl;
        //std::cout << Position.pose_of_0camera << std::endl;
        line_counter++;
    }
}

void ImageSequence::loadAiDrivePoses(const std::string& file_name, size_t length) {
    std::string line;
    static Eigen::Matrix4d gross_pose = Eigen::Matrix4d::Identity();
    std::ifstream file(file_name);
    size_t line_counter = 0;
    while (getline(file, line))
    {
        if (line_counter >= length)
        {
            break;
        }
        std::stringstream iss(line);
        std::string item;
        int i = 0;
        int j = 0;
        Pose pose(0, Eigen::Matrix4d::Identity());
        pose.id = line_counter;
        while (getline(iss, item,','))
        {
            double d;
            if (std::istringstream(item) >> d)
            {
                if (d > 2.0)
                { 
                    //TODO: remove this kind of parsing and replace with line counting.
                    /*pose.id = d;*/
                    continue;
                }
                pose.pose_of_0camera(j, i) = d;
                if (i % 3 == 0 && i != 0)
                {
                    i = 0;
                    j++;
                }
                else
                {
                    i++;
                }

            }
        }
        
        Eigen::Matrix4d axis_swap;
        axis_swap << 0, 0, 1, 0,
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 0, 1;
            
        gross_pose = gross_pose * pose.pose_of_0camera;
        pose.pose_of_0camera = (axis_swap.transpose()) * gross_pose;
        
        poses.emplace_back(pose);
        j = 0;
        //std::cout << ""R | t matrix " << line_counter << std::endl;
        //std::cout << Position.pose_of_0camera << std::endl;
        line_counter++;
    }
}

void ImageSequence::loadCameraMatrix(const std::string& path_to_calibfile, bool projectionMatrixNeeded = true)
{
    std::string line;
    std::ifstream file(path_to_calibfile);
    size_t matrix_length = projectionMatrixNeeded ? 3 : 2;
    while (getline(file, line))
    {
        std::stringstream iss(line);
        std::string item;
        std::string matrix_name;
        int i = 0;
        int j = 0;
        Eigen::Matrix4d P = Eigen::Matrix4d::Identity();
        while (iss >> item)
        {
            if (item.at(0) == 'P' || item.at(0) == 'T')
            {
                matrix_name = item;
            }
            double d;
            if (std::istringstream(item) >> d)
            {
                P(j, i) = d;
                if (i % matrix_length == 0 && i != 0)
                {
                    i = 0;
                    j++;
                }
                else
                {
                    i++;
                }
            }
        }
        /*std::cout << "The currently loaded matrix is " << matrix_name << std::endl;
        std::cout << P << std::endl;*/
        cameraMatrix.emplace( matrix_name,P );
    }
}

ImageSequence::ImageSequence(const std::string& path_to_images, const std::string& path_to_poses, const std::string& path_to_calibfile, size_t length)
{
    this->loadImageSeq(path_to_images,length);
    this->loadPoses(path_to_poses,length);
    //this->loadAiDrivePoses(path_to_poses, length);
    this->loadCameraMatrix(path_to_calibfile);
}

Eigen::Matrix4d& ImageSequence::getCameraMatrix(const std::string& camera_id)
{
    if (this->cameraMatrix.empty())
    {
        throw std::invalid_argument("The container of the camera matricies are empty. Firstly load calib.txt!");
    }
    if (this->cameraMatrix.find(camera_id) == this->cameraMatrix.end())
    {
        std::cout << "Cant find the camera matrix with the id of " << camera_id << "Therefore we load the first matrix with the id: " << this->cameraMatrix.begin()->first << std::endl;
        return this->cameraMatrix.begin()->second;
    }
    return this->cameraMatrix.at(camera_id);
}

Eigen::Matrix3d ImageSequence::getIntrinsicMatrix(const std::string& camera_id) const
{
    Eigen::Matrix3d ret = Eigen::Matrix3d::Identity();
    if (this->cameraMatrix.empty())
    {
        throw std::invalid_argument("The container of the camera matricies are empty. Firstly load calib.txt!");
    }
    if (this->cameraMatrix.find(camera_id) == this->cameraMatrix.end())
    {
        std::cout << "Cant find the camera matrix with the id of " << camera_id << "Therefore we load the first matrix with the id: " << this->cameraMatrix.begin()->first << std::endl;
        ret = this->cameraMatrix.begin()->second(Eigen::seq(0, 2), Eigen::seq(0, 2));
        return ret;
    }
    ret = this->cameraMatrix.at(camera_id)(Eigen::seq(0, 2), Eigen::seq(0, 2));
    //std::cout << ret << std::endl;
    return ret;
}

const std::vector<Pose>& ImageSequence::getPoses() const
{
    if (this->poses.empty())
    {
        throw std::invalid_argument("Poses are empty, Please load odometry_poses from the Kitti dataset");
    }
    return this->poses;
}

const cv::Size ImageSequence::getImageSize() const
{
    return image_size;
}

void ImageSequence::loadImageSeq(const std::string& path, size_t length = 10)
{
    size_t i = 0;
    std::map<int, std::string> sorted_paths;
    for (auto& entry : std::filesystem::directory_iterator(path))
    {
        if (entry.path().extension().string() != ".png")
        {
            continue;
        }
        std::string word = entry.path().stem().string();
        int cid = std::stoi(word);
        
        sorted_paths.emplace(cid, entry.path().string());
    }
    
    for (const auto &entry : sorted_paths)
    {
        if (i >= length)
        {
            break;
        }
        //std::experimental::filesystem::path sim_path{ entry };
        std::string filename = entry.second;//sim_path.u8string();
        Frame frame;
        frame.image = cv::imread(filename);
        frame.id = i;
        frame.path_of_image = filename;
        image_sequence.emplace(i, frame);
        i++;
        if (!does_image_size_set)
        {
            image_size.width = frame.image.cols;
            image_size.height = frame.image.rows;
            does_image_size_set = true;
        }
    }
}

void ImageSequence::loadAiDriveImageSeq(const std::string& path, size_t length)
{
    size_t i = 0;
    std::map<int, std::string> sorted_paths;
    for (auto& entry : std::filesystem::directory_iterator(path))
    {
        if (entry.path().extension().string() != ".png")
        {
            continue;
        }
        std::string word = entry.path().stem().string();
        int cid = std::stoi(word);

        sorted_paths.emplace(cid, entry.path().string());
    }

    for (const auto& entry : sorted_paths)
    {
        if (i >= length)
        {
            break;
        }
        //std::experimental::filesystem::path sim_path{ entry };
        std::string filename = entry.second;//sim_path.u8string();
        Frame frame;
        frame.image = cv::imread(filename);
        frame.id = i;
        frame.path_of_image = filename;
        image_sequence.emplace(i, frame);
        i++;
        if (!does_image_size_set)
        {
            image_size.width = frame.image.cols;
            image_size.height = frame.image.rows;
            does_image_size_set = true;
        }
    }
}

std::map<size_t, Frame> ImageSequence::getImageSeq() const
{
    return this->image_sequence;
}


cv::Mat ImageSequence::getImageById(size_t id) const
{
    if (this->image_sequence.empty())
    {
        //throw std::invalid_argument("Empty image sequence. Please load it from the dataset!");
        std::cout << "Empty image sequence.Please load it from the dataset!" << std::endl;
        return cv::Mat();
    }
    for (const auto& elem : this->image_sequence)
    {
        if (elem.second.id == id)
        {
            //std::cout <<"PICTURE:      " << elem.first << std::endl;
            return elem.second.image;
        }
    }
    return cv::Mat();
}

std::vector<std::pair<Frame, Pose>> ImageSequence::makeFramePosePairs() const
{
    std::vector<std::pair<Frame, Pose>> frame_pose_paires;
    if (poses.empty() || image_sequence.empty())
    {
        std::cout << "Empty Image sequence or empty poses!" << std::endl;
        return std::vector<std::pair<Frame, Pose>>();
    }
    for (size_t i = 0; i < poses.size(); i++)
    {
        if (image_sequence.find(i) == image_sequence.end())
        {
            continue;
        }
        frame_pose_paires.push_back({ image_sequence.at(i),poses[i] });
    }
    return frame_pose_paires;
    
}
