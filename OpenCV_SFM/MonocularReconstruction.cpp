#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;

#define width 3072	// 图片宽度
#define height 2048	// 图片长度

void extract_features(
	vector<string>& image_names,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<Mat>& descriptor_for_all,
	vector <vector<Vec3b>>& colors_for_all
);

void match_features(Mat& query, Mat& train, vector<DMatch>& matches);

bool find_transform(Mat& K, vector<Point2f>& p1, vector<Point2f>& p2, Mat& R, Mat& T, Mat& mask);

void reconstruct(Mat& K, Mat& R, Mat& T, vector<Point2f>& p1, vector<Point2f>& p2, Mat& structure);

void get_matched_points(
	vector<KeyPoint>& p1,
	vector<KeyPoint>& p2,
	vector<DMatch> matches,
	vector<Point2f>& out_p1,
	vector<Point2f>& out_p2
);

void get_matched_colors(
	vector<Vec3b>& c1,
	vector<Vec3b>& c2,
	vector<DMatch> matches,
	vector<Vec3b>& out_c1,
	vector<Vec3b>& out_c2
);

void maskout_points(vector<Point2f>& p1, Mat& mask);

void maskout_colors(vector<Vec3b>& p1, Mat& mask);

void save_structure(string file_name, vector<Mat>& rotations, vector<Mat>& motions, Mat& structure, vector<Vec3b>& colors);

//int main(int argc, char** argv)
//{
//	string img1 = "E:\\VS_code\\OpenCV_SFM\\image\\0004.png";
//	string img2 = "E:\\VS_code\\OpenCV_SFM\\image\\0006.png";
//	vector<string> img_names = { img1, img2 };
//
//	vector<vector<KeyPoint>> key_points_for_all;
//	vector<Mat> descriptor_for_all;
//	vector<vector<Vec3b>> colors_for_all;
//	vector<DMatch> matches;
//
//	// 本征矩阵
//	Mat K(Matx33d(
//		2759.48, 0, 1520.69,
//		0, 2764.16, 1006.81,
//		0, 0, 1
//	));
//
//	// 提取特征
//	extract_features(img_names, key_points_for_all, descriptor_for_all, colors_for_all);
//
//	// 特征匹配
//	match_features(descriptor_for_all[0], descriptor_for_all[1], matches);
//
//	// 计算变换矩阵
//	vector<Point2f> p1, p2;
//	vector<Vec3b> c1, c2;
//	Mat R, T;	// 旋转矩阵和平移向量
//	Mat mask;	// mask中大于零的点代表匹配点，等于零的点代表失配点
//	get_matched_points(key_points_for_all[0], key_points_for_all[1], matches, p1, p2);
//	get_matched_colors(colors_for_all[0], colors_for_all[1], matches, c1, c2);
//	find_transform(K, p1, p2, R, T, mask);
//
//	// 三维重建
//	Mat structure;	// 4行N列的矩阵，每一列代表空间中的一个点（齐次坐标）
//	maskout_points(p1, mask);
//	maskout_points(p2, mask);
//	reconstruct(K, R, T, p1, p2, structure);
//
//
//	// 保存并显示
//	vector<Mat> rotations = { Mat::eye(3, 3, CV_64FC1), R };
//	vector<Mat> motions = { Mat::zeros(3, 1, CV_64FC1), T };
//	maskout_colors(c1, mask);
//	save_structure("E:\\VS_code\\OpenCV_SFM\\Viewer\\structure.yml", rotations, motions, structure, c1);
//	cout << "successful!!!" << endl;
//	getchar();
//	return 0;
//}

void extract_features(
	vector<string>& image_names,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<Mat>& descriptor_for_all,
	vector <vector<Vec3b>>& colors_for_all
)
{
	key_points_for_all.clear();
	descriptor_for_all.clear();
	Mat image;

	// 读取图像，获取图像特征点并保存
	Ptr<Feature2D> sift = xfeatures2d::SIFT::create(0, 3, 0.04, 10);
	for (auto it = image_names.begin(); it != image_names.end(); ++it)
	{
		image = imread(*it);
		if (image.empty())
		{
			continue;
		}

		vector<KeyPoint> key_points;
		Mat descriptor;
		// 偶尔出现内存分配失败的错误  Detects keypoints and computes the descriptors
		// sift->detectAndCompute(image, noArray(), key_points, descriptor);
		sift->detect(image, key_points);
		sift->compute(image, key_points, descriptor);


		// 特征点过少，则排除该图像
		if (key_points.size() <= 10)
		{
			continue;
		}

		key_points_for_all.push_back(key_points);
		descriptor_for_all.push_back(descriptor);

		vector<Vec3b> colors(key_points.size());	// 三通道 存放该位置三通道颜色
		for (int i = 0; i < key_points.size(); ++i)
		{
			Point2f& p = key_points[i].pt;
			/*cout << p.x << ", " << p.y << endl;
			if (i == 2653)
			{
				cout << p.x << ", " << p.y << endl;
				cout << image.rows << ", " << image.cols << endl;
			}*/
			if (p.x <= image.rows && p.y <= image.cols)
				colors[i] = image.at<Vec3b>(p.x, p.y);
		}

		colors_for_all.push_back(colors);
	}
}

void match_features(Mat& query, Mat& train, vector<DMatch>& matches)
{
	vector<vector<DMatch>> knn_matches;
	BFMatcher matcher(NORM_L2);
	matcher.knnMatch(query, train, knn_matches, 2);

	// 获取满足Ratio Test的最小匹配的距离
	float min_dist = FLT_MAX;
	for (int r = 0; r < knn_matches.size(); ++r)
	{
		// Rotio Test
		if (knn_matches[r][0].distance > 0.6 * knn_matches[r][1].distance)
		{
			continue;
		}

		float dist = knn_matches[r][0].distance;
		if (dist < min_dist)
		{
			min_dist = dist;
		}
	}

	matches.clear();
	for (size_t r = 0; r < knn_matches.size(); ++r)
	{
		// 排除不满足Ratio Test的点和匹配距离过大的点
		if (
			knn_matches[r][0].distance > 0.6 * knn_matches[r][1].distance ||
			knn_matches[r][0].distance > 5 * max(min_dist, 10.0f)
			)
		{
			continue;
		}

		// 保存匹配点
		matches.push_back(knn_matches[r][0]);
	}
}

bool find_transform(Mat& K, vector<Point2f>& p1, vector<Point2f>& p2, Mat& R, Mat& T, Mat& mask)
{
	// 根据内参数矩阵获取相机的焦距和光心坐标（主点坐标）
	double focal_length = 0.5 * (K.at<double>(0) + K.at<double>(4));
	Point2d principle_point(K.at<double>(2), K.at<double>(5));

	// 根据匹配点求取本征矩阵，使用RANSAC，进一步排除失配点
	Mat E = findEssentialMat(p1, p2, focal_length, principle_point, RANSAC, 0.999, 1.0, mask);
	if (E.empty())
	{
		return false;
	}

	double feasible_count = countNonZero(mask);	// 得到非零元素，即数组中的有效点
	// cout << (int)feasible_count << " - in - " << p1.size() << endl;

	// 对于RANSAC而言，outlier数量大于50%时，结果是不可靠的
	if (feasible_count <= 15 || (feasible_count / p1.size()) < 0.6)
	{
		return false;
	}

	// 分解本征矩阵，获取相对变换
	int pass_count = recoverPose(E, p1, p2, R, T, focal_length, principle_point, mask);

	// cout << "pass_count = " << pass_count << endl;

	// 同时位于两个相机前方的点的数量要足够大
	if (((double)pass_count) / feasible_count < 0.7)
	{
		return false;
	}
	return true;
}

void reconstruct(Mat& K, Mat& R, Mat& T, vector<Point2f>& p1, vector<Point2f>& p2, Mat& structure)
{
	// 两个相机的投影矩阵[R T], triangulatePoints 只支持float型
	Mat proj1(3, 4, CV_32FC1);
	Mat proj2(3, 4, CV_32FC1);

	proj1(Range(0, 3), Range(0, 3)) = Mat::eye(3, 3, CV_32FC1);	// 对角矩阵 为1
	proj1.col(3) = Mat::zeros(3, 1, CV_32FC1);

	R.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
	T.convertTo(proj2.col(3), CV_32FC1);

	Mat fK;
	K.convertTo(fK, CV_32FC1);
	proj1 = fK * proj1;
	proj2 = fK * proj2;

	// 三角化重建
	triangulatePoints(proj1, proj2, p1, p2, structure);
}

void get_matched_points(
	vector<KeyPoint>& p1,
	vector<KeyPoint>& p2,
	vector<DMatch> matches,
	vector<Point2f>& out_p1,
	vector<Point2f>& out_p2
)
{
	out_p1.clear();
	out_p2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		/*Point2f tmp1 = p1[matches[i].queryIdx].pt;
		Point2f tmp2 = p2[matches[i].trainIdx].pt;
		if (tmp1.x <= width && tmp1.y < height)*/
		out_p1.push_back(p1[matches[i].queryIdx].pt);
		//if (tmp2.x <= width && tmp2.y < height)
		out_p2.push_back(p2[matches[i].trainIdx].pt);
	}
}

void get_matched_colors(
	vector<Vec3b>& c1,
	vector<Vec3b>& c2,
	vector<DMatch> matches,
	vector<Vec3b>& out_c1,
	vector<Vec3b>& out_c2
)
{
	out_c1.clear();
	out_c2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_c1.push_back(c1[matches[i].queryIdx]);
		out_c2.push_back(c2[matches[i].trainIdx]);
	}
}

void maskout_points(vector<Point2f>& p1, Mat& mask)
{
	vector<Point2f> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
		{
			p1.push_back(p1_copy[i]);
		}
	}
}

void maskout_colors(vector<Vec3b>& p1, Mat& mask)
{
	vector<Vec3b> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
		{
			p1.push_back(p1_copy[i]);
		}
	}
}

void save_structure(string file_name, vector<Mat>& rotations, vector<Mat>& motions, Mat& structure, vector<Vec3b>& colors)
{
	int n = (int)rotations.size();

	FileStorage fs(file_name, FileStorage::WRITE);
	fs << "Camera Count" << n;
	fs << "Point Count" << structure.cols;

	fs << "Rotations" << "[";
	for (size_t i = 0; i < n; ++i)
	{
		fs << rotations[i];
	}
	fs << "]";

	fs << "Motions" << "[";
	for (size_t i = 0; i < n; i++)
	{
		fs << motions[i];
	}
	fs << "]";

	fs << "Points" << "[";
	for (size_t i = 0; i < structure.cols; ++i)
	{
		Mat_<float> c = structure.col(i);
		c /= c(3);	//齐次坐标，需要除以最后一个元素才是真正的坐标值
		fs << Point3f(c(0), c(1), c(2));
	}
	fs << "]";

	fs << "Colors" << "[";
	for (size_t i = 0; i < colors.size(); ++i)
	{
		fs << colors[i];
	}
	fs << "]";

	fs.release();

}