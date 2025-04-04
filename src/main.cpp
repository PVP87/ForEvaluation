//
// SDD object detection using OpenCV
//   - Using SSD MobileNet v2 COCO data with TensorFlow
//
// configration file (.pbtxt) downloaded from below:
// https://github.com/opencv/opencv_extra/tree/master/testdata/dnn
//
// SDD MobileNet model file (.pb) downloaded from below:
// https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
//
// Sample source:
// https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.cpp
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <chrono>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/core/utils/filesystem.hpp>

#if CV_VERSION_MAJOR < 4
#pragma message( "OpenCV version < 4" )
#endif

#include "SSDModel.h"
#include "Graphic.h"


int main()
{
	float conf_threshold = 0.5f; //����� �������
	float nms_threshold = 0.5f; //����������� ����� �������

	// �������� ������������ ���������� �������� �����
	std::string img_file = "C:/Users/Vasil/source/repos/AutoTargetWin/AutoTargetWin/Images/boatV.mp4";
	if (img_file == "")
	{
		std::cout << "Input file is not specified.\n";
		return 0;
	}
	if (!cv::utils::fs::exists(img_file))
	{
		std::cout << "Input file (" << img_file << ") not found.\n";
		return 0;
	}

	const std::string window_name = "Object Detection";
	cv::namedWindow(window_name, cv::WINDOW_NORMAL);

	// �������� �������� ��� �������� ����������� �� ������� � �� ��������������
	// ��������� ����� ��������� image_queue � detection_queue, ������� ��������� ��������� ���� 
	// MessageQueue<cv::Mat>.
	// ��� ������� ��������� ����������� � �������������� ��������� new.
	// ����� MessageQueue ������������ ��� �������� �������� ���� cv::Mat(����������� OpenCV).
	std::shared_ptr<MessageQueue<cv::Mat>> image_queue(new MessageQueue<cv::Mat>);
	std::shared_ptr<MessageQueue<cv::Mat>> detection_queue(new MessageQueue<cv::Mat>);

	// ������� SSD MobileNet ������
	SSDModel ssd_model = SSDModel(conf_threshold, nms_threshold);

	// Create Graphic model which handles images 
	Graphic input = Graphic(img_file, ssd_model.getClassNumber());

	// Set shared pointers of queues into objects
	input.setImageQueue(image_queue);
	input.setDetectionQueue(detection_queue);
	ssd_model.setDetectionQueue(detection_queue);

	cv::resizeWindow(window_name, input.getWindowSize());

	// Launch the readinig thread and the detecting thread
	input.thread_for_read(); //��������� ����� � ������� ������� ������ � ��������� detection_queue
	ssd_model.thread_for_detection();


	std::vector<int> classIds;
	std::vector<std::string> classNames;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	cv::Mat current_image;

	const int duration = (int)(1000 / input.getFps());
	int count = 0;
	std::this_thread::sleep_for(std::chrono::seconds(1));

	while (cv::waitKey(duration) < 0)
	{

		if (image_queue->getTotal() > 0 && count >= image_queue->getTotal())
		{
			break;
		}

		//std::this_thread::sleep_for(std::chrono::milliseconds(1));
		current_image = image_queue->receive();

		// Execute the detection once per counts specified by getDetectFreq()
		if (count % (input.getDetectFreq()) == 0)
		{
			ssd_model.getNextDetection(classIds, classNames, confidences, boxes);
		}

		// Plot the result and show the image on window
		input.drawResult(current_image, classIds, classNames, confidences, boxes);
		cv::imshow(window_name, current_image);

		++count;
	}
	std::cout << " --- Object detection finished. Press Enter key to quit.---\n";
	cv::waitKey(0);
	return 0;
}
