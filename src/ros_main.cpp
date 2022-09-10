///////////////////////////////////////////////////////////////////////////////
//  SORT: A Simple, Online and Realtime Tracker
//  
//  This is a ROS + C++ reimplementation of the open source tracker in
//  https://github.com/abewley/sort
//  Based on the work of Alex Bewley, alex@dynamicdetection.com, 2016
//
//
//  ROS implemented by EungChang Mason Lee, eungchang_mason@kaist.ac.kr, 2020
//
//  It receives bounding box data from Object Detector (here used YOLO v3 ROS version)
//  and outputs tracked bounding boxes and the image to show 
//
///////////////////////////////////////////////////////////////////////////////


#include <ros/ros.h>
#include "darknet_ros_msgs/BoundingBoxes.h" // from YOLO ROS version,

#include <iostream>
#include <iomanip> // to format image names using setw() and setfill()
#include <unistd.h>
#include <string>
#include <set>

#include "Hungarian.h"
#include "KalmanTracker.h"
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include "opencv2/video/tracking.hpp"
#include <std_msgs/UInt8MultiArray.h>

#define StateType Rect_<float>

using namespace std;
using namespace cv;

typedef struct TrackingBox
{
	int frame;
	int id;
	Rect_<float> box;
}TrackingBox;

// Computes IOU between two bounding boxes
double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
	float in = (bb_test & bb_gt).area();
	float un = bb_test.area() + bb_gt.area() - in;

	if (un < DBL_EPSILON)
		return 0;

	return (double)(in / un);
}
bool display = false;
// global variables for counting
#define CNUM 20
//darknet_ros_msgs::BoundingBox predBox;
//darknet_ros_msgs::BoundingBox detectBox;
darknet_ros_msgs::BoundingBoxes tracked_bboxes;
darknet_ros_msgs::BoundingBoxes detected_bboxes;
darknet_ros_msgs::BoundingBoxes predicted_bboxes;
cv::Mat frame;
cv::Mat frame2;
vector<TrackingBox> detData;
int maxFrame = 1;
// 0. randomly generate colors, only for display
RNG rng(0xFFFFFFFF);
Scalar_<int> randColor[CNUM];
vector<KalmanTracker> trackers;
int KalmanTracker::kf_count = 0; // tracking id relies on this, so we have to reset it in each seq.

void setup(){
    for (int i = 0; i < CNUM; i++)
	    rng.fill(randColor[i], RNG::UNIFORM, 0, 256);
}
// get_rect(float cx, float cy, float s, float r)
// {
// 	float w = sqrt(s * r);
// 	float h = s / w;
// 	float x = (cx - w / 2);
// 	float y = (cy - h / 2);

// 	if (x < 0 && cx > 0)
// 		x = 0;
// 	if (y < 0 && cy > 0)
// 		y = 0;

// 	return StateType(x, y, w, h);
// }
void SORT(const darknet_ros_msgs::BoundingBoxes::ConstPtr& msg)
{
	//ROS_INFO("current0");
    darknet_ros_msgs::BoundingBoxes bboxes;
    bboxes.bounding_boxes = msg->bounding_boxes;
	//detected_bboxes = bboxes.bounding_boxes
	//ROS_INFO("current0");
    // 1. read bounding boxes from object detector, here from YOLO v3 ROS version.
    ROS_INFO("%d",bboxes.bounding_boxes.size());
	for (int i=0; i< int(bboxes.bounding_boxes.size()) ; i++)
    {
        TrackingBox tb;
        tb.frame = 1;
        tb.id = bboxes.bounding_boxes[i].id;
		
        tb.box = Rect_<float>(Point_<float>(float(bboxes.bounding_boxes[i].xmin), float(bboxes.bounding_boxes[i].ymin)), Point_<float>(float(bboxes.bounding_boxes[i].xmax), float(bboxes.bounding_boxes[i].ymax)));
		cv::rectangle(frame, tb.box, Scalar_<int>(0,255,0), 2, 8, 0);
		detData.push_back(tb);
		//cout << tb.box.x << endl;
    }

	// 2. group detData by frame
	vector<vector<TrackingBox>> detFrameData;
    detFrameData.push_back(detData);

	// 3. update across frames
	int frame_count = 0;
	int max_age = 1;
	int min_hits = 3;
	double iouThreshold = 0.3;
	//vector<KalmanTracker> trackers;
	KalmanTracker::kf_count = 0; // tracking id relies on this, so we have to reset it in each seq.

	// variables used in the for-loop
	vector<Rect_<float>> predictedBoxes;
	vector<vector<double>> iouMatrix;
	vector<int> assignment;
	set<int> unmatchedDetections;
	set<int> unmatchedTrajectories;
	set<int> allItems;
	set<int> matchedItems;
	vector<cv::Point> matchedPairs;
	vector<TrackingBox> frameTrackingResult;
	unsigned int trkNum = 0;
	unsigned int detNum = 0;

	double cycle_time = 0.0;
	int64 start_time = 0;
	set<int> manageId;
		
	//////////////////////////////////////////////
	// main loop
	for (int fi = 0; fi < maxFrame; fi++)
	{
		//ROS_INFO("current4");
		frame_count++;
		// I used to count running time using clock(), but found it seems to conflict with cv::cvWaitkey(),
		// when they both exists, clock() can not get right result. Now I use cv::getTickCount() instead.
		start_time = getTickCount();
		ROS_INFO("trackes size: %d",trackers.size());
		if (trackers.size() == 0) // the first frame met
		{
			// initialize kalman trackers using first detections.
			for (unsigned int i = 0; i < detFrameData[fi].size(); i++)
			{
				
				manageId.insert(KalmanTracker::kf_count);
				KalmanTracker trk = KalmanTracker(detFrameData[fi][i].box);
				ROS_INFO("m_id %d",trk.m_id);
				//ROS_INFO("detframedata: %d",detFrameData[fi].size());
				
				trackers.push_back(trk);
			}
			// output the first frame detections
			for (unsigned int id = 0; id < detFrameData[fi].size(); id++)
			{
				TrackingBox tb = detFrameData[fi][id];
				//cv::rectangle(frame, tb.box, Scalar_<int>(0,255,0), 2, 8, 0);
				//ROS_INFO("tb id %d ", tb.id);
			}
			
			ROS_INFO("trackes size2: %d",trackers.size());
			continue;
		}
		ROS_INFO("trackes size3: %d",trackers.size());
		//ROS_INFO("tb id2 ");
		///////////////////////////////////////
		// 3.1. get predicted locations from existing trackers.
		predictedBoxes.clear(); 
		darknet_ros_msgs::BoundingBox predBox;
		for (auto it = trackers.begin(); it != trackers.end();)
		{
			//KalmanTracker::kf_count++;
			Rect_<float> pBox = (*it).predict();
			if (pBox.x >= 0 && pBox.y >= 0)
			{
				// boundingBox.id = ;
				// boundingBox.xmin = res.box.x;
				// boundingBox.ymin = res.box.y;
				// boundingBox.xmax = res.box.x + res.box.width;
				// boundingBox.ymax = res.box.y + res.box.height;
				// tracked_bboxes.bounding_boxes.push_back(boundingBox);
				cv::rectangle(frame, pBox,  Scalar_<int>(255,0,0), 2, 8, 0);
				predictedBoxes.push_back(pBox);
				
				//ROS_INFO("tb id ");
				it++;
			}
			else
			{
				//ROS_INFO("tb id 2");
				//manageId.erase((*it).m_id);
				it = trackers.erase(it);
				//cerrdetFrameData
			}
		}

		///////////////////////////////////////
		// 3.2. associate detections to tracked object (both represented as bounding boxes)
		// dets : detFrameData[fi]
		trkNum = predictedBoxes.size();
		detNum = detFrameData[fi].size();

		iouMatrix.clear();
		iouMatrix.resize(trkNum, vector<double>(detNum, 0));
		
		for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
		{
			for (unsigned int j = 0; j < detNum; j++)
			{
				// use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
				iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detFrameData[fi][j].box);
			}
		}

		// solve the assignment problem using hungarian algorithm.
		// the resulting assignment is [track(prediction) : detection], with len=preNum
		HungarianAlgorithm HungAlgo;
		assignment.clear();
		HungAlgo.Solve(iouMatrix, assignment);
		
		// find matches, unmatched_detections and unmatched_predictions
		unmatchedTrajectories.clear();
		unmatchedDetections.clear();
		allItems.clear();
		matchedItems.clear();
		ROS_INFO("det num: %d",detNum);
		ROS_INFO("trk num: %d",trkNum);
		if (detNum > trkNum) //	there are unmatched detections
		{
			for (unsigned int n = 0; n < detNum; n++)
				allItems.insert(n);

			for (unsigned int i = 0; i < assignment.size(); ++i)
				matchedItems.insert(assignment[i]);

			set_difference(allItems.begin(), allItems.end(),
				matchedItems.begin(), matchedItems.end(),
				insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
		}
		else if (detNum < trkNum) // there are unmatched trajectory/predictions
		{
			for (unsigned int i = 0; i < trkNum; ++i)
				if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
					{
						manageId.insert(trackers[i].m_id);
						
						unmatchedTrajectories.insert(i);
						ROS_INFO("-------------------------");
						ROS_INFO("UNMATCHED tracker , %d",i);
						ROS_INFO("UNMATCHED ID** , %d",trackers[i].m_id);
					}
					
		}
		else
			;

		// filter out matched with low IOU
		matchedPairs.clear();
		for (unsigned int i = 0; i < trkNum; ++i)
		{
			if (assignment[i] == -1) // pass over invalid values
				continue;
			if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
			{
				unmatchedTrajectories.insert(i);
				unmatchedDetections.insert(assignment[i]);
			}
			else
				matchedPairs.push_back(cv::Point(i, assignment[i]));
		}

		///////////////////////////////////////
		// 3.3. updating trackers

		// update matched trackers with assigned detections.
		// each prediction is corresponding to a tracker
		int detIdx, trkIdx;
		//set<int> manageId;
		
		for (unsigned int i = 0; i < matchedPairs.size(); i++)
		{
			ROS_INFO("matched pair %d",matchedPairs.size());
			trkIdx = matchedPairs[i].x;
			detIdx = matchedPairs[i].y;
			trackers[trkIdx].update(detFrameData[fi][detIdx].box);
			//trackers[trkIdx].m_id = trkIdx;
			ROS_INFO("trkidx id %d",trkIdx);
			ROS_INFO("detidx id %d",detIdx);
			ROS_INFO("Matched m_id %d --",trackers[trkIdx].m_id);
			manageId.insert(trackers[trkIdx].m_id);
			// if((KalmanTracker::kf_count) == trackers[trkIdx].m_id)
			// {
			// 	ROS_INFO("kfcount:%d",KalmanTracker::kf_count);
			// 	KalmanTracker::kf_count++;
			// }
		}
		
		
		// create and initialise new trackers for unmatched detections
		
		for (auto umd : unmatchedDetections)
		{
			
			while(1)
			{
				set<int>::iterator Iter = manageId.find(KalmanTracker::kf_count);
				if(Iter != manageId.end())
				{
					ROS_INFO("find %d, %d",KalmanTracker::kf_count,(*Iter));
					KalmanTracker::kf_count++;
				}
				else
				{
					ROS_INFO("not find %d, %d",KalmanTracker::kf_count,(*Iter));
					break;
				}
			}

			manageId.insert(KalmanTracker::kf_count);
			KalmanTracker tracker = KalmanTracker(detFrameData[fi][umd].box);
			ROS_INFO("UM m_id %d **",tracker.m_id);
			
			trackers.push_back(tracker);
			
			ROS_INFO("Add New Detection");
			
		}
		
		ROS_INFO("Display %d UMT : %d",display,unmatchedTrajectories.size());
		//Delete unmatchedTrajectories 
		for (auto it = trackers.begin(); it!=trackers.end();)
		{

				if(it != trackers.end() && (*it).m_time_since_update > max_age)
				{
				manageId.erase((*it).m_id);
				trackers.erase(it);
				it--;
				ROS_INFO("ERASE");
				}
				it++;
		}
		
		// get trackers' output
		detData.clear();
		detFrameData.clear();
		frameTrackingResult.clear();
		for (auto it = trackers.begin(); it != trackers.end();)
		{
			//ROS_INFO("tb id ");
			if (((*it).m_time_since_update < 1) &&
				((*it).m_hit_streak >= min_hits || frame_count <= min_hits))
			{
				darknet_ros_msgs::BoundingBox boundingBox;
				TrackingBox res;
				res.box = (*it).get_state();
				res.id = (*it).m_id + 1;
				res.frame = frame_count;
				if(display){
				//cv::rectangle(frame, res.box, randColor[res.id % CNUM], 2, 8, 0);
				cv::rectangle(frame, res.box, Scalar_<int>(0,0,255), 2, 8, 0);
				cv::putText(frame,std::to_string(res.id),Point_<int>(int(res.box.x),int(res.box.y)),cv::FONT_ITALIC,1,randColor[res.id % CNUM],2);
				cv::imshow("view", frame);
				cv::waitKey(1);
				}
				frameTrackingResult.push_back(res);
				
				 boundingBox.id = res.id;
				 boundingBox.xmin = res.box.x;
				 boundingBox.ymin = res.box.y;
				 boundingBox.xmax = res.box.x + res.box.width;
				 boundingBox.ymax = res.box.y + res.box.height;
				 tracked_bboxes.bounding_boxes.push_back(boundingBox);
				it++;
				
			}
			else
				it++;
			// remove dead tracklet
			//ROS_INFO("time %d",(*it).m_time_since_update);
			//if (it != trackers.end() && (*it).m_time_since_update > max_age)
			//{
			//	manageId.erase((*it).m_id); // filter id remove
			//	ROS_INFO("m_time: %d ERASE",(*it).m_time_since_update);
			//	it = trackers.erase(it);
				
			//}
			// for (auto it = trackers.begin(); it != trackers.end();)
			// {
			// 	darknet_ros_msgs::BoundingBox boundingBox;
			// 	boundingBox.id = (*it).m_id;
			// 	boundingBox.xmin = (*it).get_state().x;
			// 	boundingBox.ymin = (*it).get_state().y;
			// 	boundingBox.xmax = (*it).get_state().x + (*it).get_state().width;
			// 	boundingBox.ymax = (*it).get_state().y + (*it).get_state().height;
			// 	tracked_bboxes.bounding_boxes.push_back(boundingBox);
			// }
			//ROS_INFO("tb id ");
//		    if (display) // read image, draw results and show them
//		    {
//			    cv::rectangle(img, tb.box, randColor[tb.id % CNUM], 2, 8, 0);
//		    }
			
		}
		

		cycle_time = (double)(getTickCount() - start_time);
        double fps = (1.0/cycle_time)*getTickFrequency();

        ROS_INFO("current : %.1f", fps);
	}
	//ROS_INFO("current2");
	
}



void imgCall(const sensor_msgs::ImageConstPtr& msg)
{
	try
  {
    // Decode image in msg
    frame = cv_bridge::toCvShare(msg, "bgr8")->image;
    display = true;
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cannot decode image");
  }
}

int main(int argc, char **argv)
{
    setup();
    ros::init(argc, argv, "SORT");
    ros::NodeHandle n;
	cv::namedWindow("view");
  	cv::startWindowThread();
    ros::Subscriber sub = n.subscribe("/darknet_ros/bounding_boxes", 50, &SORT);
	ros::Subscriber img_sub = n.subscribe("/zed2/zed_node/left_raw/image_raw_color", 1, &imgCall);
    ros::Publisher result_boxes = n.advertise<darknet_ros_msgs::BoundingBoxes>("tracked_boxes",50);
    ros::Rate loop_rate(30);
    while (ros::ok())
    {
		//ROS_INFO("current3");
		result_boxes.publish(tracked_bboxes);
		tracked_bboxes.bounding_boxes.clear();
        ros::spinOnce();
        loop_rate.sleep();
    }
	cv::destroyWindow("view");
}
