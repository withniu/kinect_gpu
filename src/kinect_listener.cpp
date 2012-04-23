#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>        // GPU structures and methods

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

#include <pcl/correspondence.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/registration/correspondence_rejection_sample_consensus.h>

#define USE_IMAGE_TRANSPORT_SUBSCRIBER_FILTER 1
#define DRAW 1
#define GPU 0


#if USE_IMAGE_TRANSPORT_SUBSCRIBER_FILTER
#  include <image_transport/subscriber_filter.h>
#else
#  include <sensor_msgs/Image.h>
#  include <message_filters/subscriber.h>
#endif

const float fx = 525.0;
const float fy = 525.0;
const float cx = 319.5;
const float cy = 239.5;

class KinectListener {
// Members
private:


	int index_;		// Count for callback to switch ping-pong
	ros::Time time_;	// Timing for callback
	ros::NodeHandle nh_;
	image_transport::ImageTransport it_;
	ros::Publisher pub_;

	pcl::PointCloud<pcl::PointXYZRGB> cloud1_,cloud2_;	// 3D RGB cloudpoints
	pcl::PointCloud<pcl::PointXYZ>::Ptr feature_cloud_ptr1_,feature_cloud_ptr2_;
	

	cv::Mat img1_,img2_;
	cv::Mat img1_depth_,img2_depth_;
	cv::Mat img1_features_,img2_features_;
	cv::vector<cv::KeyPoint> keypoints1_, keypoints2_;
	cv::Mat descriptors1_, descriptors2_;
	cv::vector<cv::Point3f> keypoints3d1_,keypoints3d2_;
	
	
#if GPU	
	cv::gpu::GpuMat keypoints1_dev_, descriptors1_dev_;
	cv::gpu::GpuMat keypoints2_dev_, descriptors2_dev_;
	cv::gpu::GpuMat img_dev_, mask_dev_;
	cv::gpu::SURF_GPU surf_;
#endif


#if USE_IMAGE_TRANSPORT_SUBSCRIBER_FILTER
  typedef image_transport::SubscriberFilter ImageSubscriber;
#else
  typedef message_filters::Subscriber< sensor_msgs::Image > ImageSubscriber;
#endif
	ImageSubscriber rgb_image_sub_;
	ImageSubscriber depth_image_sub_;
	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
	message_filters::Synchronizer< MySyncPolicy > sync;

// Methods
public:
	KinectListener() :	// Constructor
		feature_cloud_ptr1_ (new pcl::PointCloud<pcl::PointXYZ>),
		feature_cloud_ptr2_ (new pcl::PointCloud<pcl::PointXYZ>),		
		it_(nh_),
#if USE_IMAGE_TRANSPORT_SUBSCRIBER_FILTER
	    	rgb_image_sub_( it_, "/camera/rgb/image_color", 1 ),
		depth_image_sub_( it_, "/camera/depth/image", 1 ),
#else
		rgb_image_sub_( nh_, "image/rgb", 1 ),
		depth_image_sub_( nh_, "image/depth", 1 ),
#endif
		sync( MySyncPolicy( 10 ), rgb_image_sub_, depth_image_sub_ )
		{
			pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB> > ("keypoints3d", 1);
			index_ = 0;
			time_ = ros::Time::now();
			img1_ = cv::Mat::ones(480,640,CV_8UC3);
			img1_depth_ = cv::Mat::ones(480,640,CV_32FC1);

			cloud1_.header.frame_id = "/openni_camera";
			cloud1_.height = 480;
			cloud1_.width = 640;
			cloud2_.header.frame_id = "/openni_camera";
			cloud2_.height = 480;
			cloud2_.width = 640;
			
			feature_cloud_ptr1_->height = 1;
			feature_cloud_ptr2_->height = 1;

#if GPU
			cv::Mat mask_host = cv::Mat::ones(480,640,CV_8UC1);
			mask_dev_.upload(mask_host);
			cv::Mat src_host(480,640,CV_8UC3);
			cv::gpu::GpuMat dst_device, src_device;
			src_device.upload(src_host);
			cv::gpu::cvtColor(src_device,dst_device,CV_BGR2GRAY);
			cv::Mat result_host;
			dst_device.download(result_host);
			ROS_INFO("GPU initialization done...");
#endif

#if DRAW
			cv::namedWindow("RGB");			
			cv::namedWindow("Depth");
			cv::namedWindow("Feature");
			cv::namedWindow("Matches");				
#endif
			sync.registerCallback( boost::bind( &KinectListener::callback, this, _1, _2 ) );
			
		}

	void callback(const sensor_msgs::ImageConstPtr& rgb_msg,
				  const sensor_msgs::ImageConstPtr& depth_msg) {
		ros::Time begin,end;
		begin = ros::Time::now();
		
		// Convert image msg to cv::Mat
		cv_bridge::CvImagePtr cv_rgb_ptr;	// cv_bridge for rgb image
		cv_bridge::CvImagePtr cv_depth_ptr;	// cv_bridge for depth image




		try
		{
			cv_rgb_ptr = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
			cv_depth_ptr = cv_bridge::toCvCopy(depth_msg);		
		}
		catch (cv_bridge::Exception& e)
		{
			ROS_ERROR("cv_bridge exception: %s", e.what());
			return;
		}


#if GPU		// GPU Version
		vector<DMatch> matches;
		try
		{
			if (number_ % 2)
				img1_ = cv_ptr->image;
			else
				img2_ = cv_ptr->image;
		
			cv::gpu::GpuMat src_dev;
			src_dev.upload(cv_ptr->image);
			cv::gpu::cvtColor(src_dev,img_dev_,CV_BGR2GRAY);
			
			// SURF GPU	
			if (number_ % 2)
				surf(img_dev_,mask_dev_,keypoints1_dev_, descriptors1_dev_);
			else
				surf(img_dev_,mask_dev_,keypoints2_dev_, descriptors2_dev_);
			
			
			cv::gpu::BruteForceMatcher_GPU<cv::L2<float> > matcher;
			
			if (!(number_ % 2))
				matcher.match(descriptors1_dev_,descriptors2_dev_,matches);
			else
				matcher.match(descriptors2_dev_,descriptors1_dev_,matches);
			
			surf.downloadKeypoints(keypoints1_dev_, keypoints1_);		
			surf.downloadKeypoints(keypoints2_dev_, keypoints2_);
#if DRAW
			cv::Mat img_matches;
			if (!(number_ % 2))
				drawMatches(img1_, keypoints1_host, img2_, keypoints2_host, matches, img_matches);
			else
				drawMatches(img2_, keypoints2_host, img1_, keypoints1_host, matches, img_matches);
			
			char filename_gpu[40];
			sprintf(filename_gpu,"gpu_kinect_rgb_matches_%03d.jpg",index_);
		    	cv::imwrite(filename_gpu,img_matches);    
#endif
		}
		catch(const cv::Exception& ex)
		{
			std::cout << "Error: " << ex.what() << std::endl;
		}

#else	// CPU version


		
		// SURF	Detection and Matching	
		cv::SurfFeatureDetector detector(400);	
		cv::BruteForceMatcher<cv::L2<float> > matcher;
		cv::vector<cv::DMatch> matches;
		cv::SurfDescriptorExtractor extractor;
		cv::Mat img_matches;
		if (index_ % 2) {
			img1_ = cv_rgb_ptr->image;
			img1_depth_ = cv_depth_ptr->image;
			detector.detect(img1_, keypoints1_);
			extractor.compute(img1_, keypoints1_, descriptors1_);
			matcher.match(descriptors2_, descriptors1_, matches);
#if DRAW
			cv::drawKeypoints(img1_,keypoints1_,img1_features_);
//			cv::imshow("Feature", img1_features_);
			cv::drawMatches(img2_, keypoints2_, img1_, keypoints1_, matches, img_matches);
#endif
		} else {
			img2_ = cv_rgb_ptr->image;
			img2_depth_ = cv_depth_ptr->image;
			detector.detect(img2_, keypoints2_);
			extractor.compute(img2_, keypoints2_, descriptors2_);
			matcher.match(descriptors1_, descriptors2_, matches);
#if DRAW			
			cv::drawKeypoints(img2_,keypoints2_,img2_features_);
//			cv::imshow("Feature", img2_features_);			
			cv::drawMatches(img1_, keypoints1_, img2_, keypoints2_, matches, img_matches);
#endif		
		}

#endif

#if GPU

#else

		// Calculate 3D points
		if (index_ % 2) 
			cloud1_.clear();		
		else
			cloud2_.clear();

		pcl::PointXYZRGB point;
		for (int r = 0;r < 480;r++)
			for (int c = 0;c < 640;c++) {
				float z = cv_depth_ptr->image.at<float>(r,c);
				point.x = (c - cx) * z / fx;
				point.y = (r - cy) * z / fy;
				point.z = z;
				cv::Point xy;
				cv::Vec3b bgr = cv_rgb_ptr->image.at<cv::Vec3b>(r,c);
				uint8_t r = bgr[2];
				uint8_t g = bgr[1];
				uint8_t b = bgr[0];
				uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
	      					static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
     				point.rgb = *reinterpret_cast<float*>(&rgb);
				if (index_ % 2) 
					cloud1_.push_back(point);		
				else
					cloud2_.push_back(point);		
		}
#endif
		
		// 3D Features Extraction
		pcl::Correspondences correspondences,correspondences_inlier;
		pcl::Correspondence correspondence;		
		for (unsigned int i = 0;i < (keypoints1_.size() < keypoints2_.size() ? keypoints1_.size() : keypoints2_.size());i++) {		
			correspondence.index_query = matches[i].queryIdx;
			correspondence.index_match = matches[i].trainIdx;
			float z1 = img1_depth_.at<float>(keypoints1_[correspondence.index_query].pt.y,keypoints1_[correspondence.index_query].pt.x);
			float z2 = img2_depth_.at<float>(keypoints2_[correspondence.index_match].pt.y,keypoints2_[correspondence.index_match].pt.x);
			if(z1 > 0.5 && z1 < 6.0 && z2 > 0.5 && z2 < 6.0)	// Only use correspondences with reasonable depth
				correspondences.push_back(correspondence);
		}		

		if (index_ % 2) {
			feature_cloud_ptr1_->clear();
			for (unsigned int i = 0;i < keypoints1_.size();i++) {
				unsigned int x1 = keypoints1_[i].pt.x;
				unsigned int y1 = keypoints1_[i].pt.y;
				pcl::PointXYZ point1;
				float z1 = img1_depth_.at<float>(y1,x1);
				z1 = (z1 > 0.5 && z1 < 6.0) ? z1 : 0.01;		
				point1.x = (x1 - cx) * z1 / fx;
				point1.y = (y1 - cy) * z1 / fy;
				point1.z = z1;	
				feature_cloud_ptr1_->points.push_back(point1);
			}
			feature_cloud_ptr1_->width = (int) keypoints1_.size();
		
		} else{
			feature_cloud_ptr2_->clear();
			for (unsigned int i = 0;i < keypoints2_.size();i++) {
				unsigned int x2 = keypoints2_[i].pt.x;
				unsigned int y2 = keypoints2_[i].pt.y;
				pcl::PointXYZ point2;
				float z2 = img2_depth_.at<float>(y2,x2);
				z2 = (z2 > 0.5 && z2 < 6.0) ? z2 : 0.01;				
				point2.x = (x2 - cx) * z2 / fx;
				point2.y = (y2 - cy) * z2 / fy;
				point2.z = z2;
				feature_cloud_ptr2_->points.push_back(point2);
			}
			feature_cloud_ptr2_->width = (int) keypoints2_.size();
		}
		
		if (matches.size() == 0) {
			std::cout << "Matches = 0!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
			index_++;
			return;
		}

		// RANSAC Pose Estimation
		pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ> ransac;
		if (index_ % 2) {
			ransac.setInputCloud (feature_cloud_ptr2_);
			ransac.setTargetCloud (feature_cloud_ptr1_);		
		} else {		
			ransac.setInputCloud (feature_cloud_ptr1_);
			ransac.setTargetCloud (feature_cloud_ptr2_);
		}		
		ransac.setMaxIterations (5000);
		ransac.setInlierThreshold (.1);

		ransac.getRemainingCorrespondences (correspondences,correspondences_inlier);
		Eigen::Matrix4f tf = ransac.getBestTransformation ();







		
//		std::cout << "##size1 = " << feature_cloud_ptr1_->size() << "##size2 = " << feature_cloud_ptr2_->size() << std::endl;		
		std::cout << "Threshold = " << ransac.getInlierThreshold () << "m\tMax Iteration = " << ransac.getMaxIterations () << std::endl;
		std::cout << tf << std::endl; 
		std::cout << "#correspondences = " << correspondences.size() << "\t#inlier = " << correspondences_inlier.size() << std::endl;
//		pub_.publish (cloud1_);

#if DRAW
		cv::imshow("Matches", img_matches);
//		cv::imshow("RGB", cv_rgb_ptr->image);
//		cv::imshow("Depth", cv_depth_ptr->image);
		cv::waitKey(3);
#endif
		index_++;
		
//		ros::Duration(1.0).sleep();
		ros::Time time_now = ros::Time::now();
		ROS_INFO("%fs, %.2fHz",time_now.toSec() - begin.toSec(),1 / (time_now.toSec() - time_.toSec()));
		time_ = time_now;
	}



};




int main(int argc, char** argv) {
  ros::init( argc, argv, "my_node" );
  KinectListener mc;

  while( ros::ok() ){
    ros::spin();
  }

  return EXIT_SUCCESS;
}
 
