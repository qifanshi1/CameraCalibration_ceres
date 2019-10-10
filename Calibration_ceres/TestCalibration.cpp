
#include"io.h"
#include"calibrate.h"
#include"optimize.h"

int main()
{
	std::vector<std::vector<Eigen::Vector2d>> object_points;
	std::vector<std::vector<Eigen::Vector2d>> image_points_left;
	std::vector<std::vector<Eigen::Vector2d>> image_points_right;
	//read object points in the target 
	std::string file_name_object = "data_point3d.txt";
	std::vector<Eigen::Vector2d> temp_pt;
	bool read_flag = readObjectpoint(file_name_object, temp_pt);
	if (!read_flag)
		return 0;
	//read image points
	for (int i = 0; i < 5; i++)
	{
		std::string file_name_left = "left_cam\\pos" + std::to_string(i + 1) + ".txt";
		std::string file_name_right = "right_cam\\pos" + std::to_string(i + 1) + ".txt";
		bool read_flag_left = false, read_flag_right = false;
		std::vector<Eigen::Vector2d> temp_pt_left, temp_pt_right;
		read_flag_left = readImagepoint(file_name_left, temp_pt_left);
		read_flag_right = readImagepoint(file_name_right, temp_pt_right);
		if (!read_flag_left || !read_flag_right)
			return 0;
		if ((temp_pt.size() == temp_pt_left.size()) && (temp_pt_left.size() == temp_pt_right.size()))
		{
			object_points.push_back(temp_pt);
			image_points_left.push_back(temp_pt_left);
			image_points_right.push_back(temp_pt_right);
		}
	}
	std::vector<Eigen::Matrix3d> homo_list_left;
	findHomoList(object_points, image_points_left, homo_list_left);

	Eigen::Matrix3d K_left;
	findInitialK(homo_list_left, K_left);

	std::vector<Eigen::Matrix3d> R_list_left;
	std::vector<Eigen::Vector3d> T_list_left;
	findInitialRT(homo_list_left, K_left, R_list_left, T_list_left);
	/*double error = 0.0;
	for (int i = 0; i < object_points.size(); i++)
	{
		double error_lone = 0.0;
		for (int j = 0; j < object_points[i].size(); j++)
		{
			double error_temp = computeError(object_points[i][j], image_points_left[i][j],
				InitialK_left, R_list_left[i], T_list_left[i]);
			error_lone += error_temp;
			error += error_temp;
		}
		std::cout << " error_lone : " << error_lone / object_points[i].size() << std::endl;
	}
	std::cout << " error : " << error / (object_points[0].size()* object_points.size()) << std::endl;*/


	optimizeKRT(object_points, image_points_left, K_left, R_list_left, T_list_left);

	/*error = 0.0;
	for (int i = 0; i < object_points.size(); i++)
	{
		double error_lone = 0.0;
		for (int j = 0; j < object_points[i].size(); j++)
		{
			double error_temp = computeError(object_points[i][j], image_points_left[i][j],
				InitialK_left, R_list_left[i], T_list_left[i]);
			error_lone += error_temp;
			error += error_temp;
		}
		std::cout << " error_lone : " << error_lone/ object_points[i].size() << std::endl;
	}
	std::cout << " error : " << error / (object_points[0].size()* object_points.size()) << std::endl;*/
	Eigen::Matrix<double, 5, 1> Distort_coeffs;
	findDistort(object_points, image_points_left, K_left, R_list_left, T_list_left, Distort_coeffs);

	optimizeKDRT(object_points, image_points_left, K_left, R_list_left, T_list_left, Distort_coeffs);

	return 0;
}