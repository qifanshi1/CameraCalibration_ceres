#pragma once
#include<string>
#include<vector>
#include<Eigen/Dense>
#include<fstream>
#include<sstream>
#include<iostream>
bool readObjectpoint(std::string file_name, std::vector<Eigen::Vector2d>& object_points)
{
	object_points.clear();

	std::ifstream in(file_name);
	if (!in.is_open())
	{
		std::cerr << "Error: the file " << file_name << " open failed !" << std::endl;
		return false;
	}
	std::string str;
	float  x, y, z;
	while (std::getline(in, str))
	{
		std::istringstream istr(str);
		istr >> x >> y >> z;
		object_points.push_back(Eigen::Vector2d(x, y));
	}
	//std::cout << " image points size : " << image_points.size() << std::endl;
	return true;
}
bool readImagepoint(std::string file_name, std::vector<Eigen::Vector2d>& image_points)
{
	image_points.clear();

	std::ifstream in(file_name);
	if (!in.is_open())
	{
		std::cerr << "Error: the file " << file_name << " open failed !" << std::endl;
		return false;
	}
	std::string str;
	float  num, x, y;
	while (std::getline(in, str))
	{
		std::istringstream istr(str);
		istr >> num >> x >> y;
		image_points.push_back(Eigen::Vector2d(x, y));
	}
	//std::cout << " image points size : " << image_points.size() << std::endl;
	return true;
}