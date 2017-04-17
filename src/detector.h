/*************************************************************************
    > File Name: detector.h
    > Author: ma6174
    > Mail: ma6174@163.com 
    > Created Time: Thu 02 Mar 2017 10:23:47 AM CST
 ************************************************************************/

#include<iostream>
#include "opencv2/opencv.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
//#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
//#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "text_proposal_layer.h"
#include "nms.h"
#include "connect.h"
#include <map>

using namespace std;
using namespace cv;

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;

#define TEXT_PROPOSALS_MIN_SCORE 0.7
#define TEXT_PROPOSALS_NMS_THRESH 0.3
#define LINE_MIN_SCORE 0.7

class TextProposalDetector{
	public:
		TextProposalDetector(boost::shared_ptr< Net<float> > net){
			m_caffenet = net;
		}
		bool detect(Mat img, vector< vector<float> >& rois, vector<float>& scores){
			img.convertTo(img, CV_32FC3);
			Mat mean(img.rows,img.cols,CV_32FC3,cv::Scalar(102.9801, 115.9465, 122.7717));
			img -= mean;
			int shorter = img.rows<img.cols?img.rows:img.cols;
			float ff = 600.0/shorter;
			int num=1,channel=3,height=img.rows*ff,width=img.cols*ff;
			m_caffenet->input_blobs()[0]->Reshape(num, channel, height, width);
			resize(img,img,cv::Size(width,height));
/*
			boost::shared_ptr<caffe::Blob<float> > info_blob=Net->blob_by_name("im_info");
			int * info_data = info_blob->mutable_cpu_data();
			info_data[0] = height;
			info_data[1] = width;
*/			
			boost::shared_ptr<caffe::Blob<float> > input_blob = m_caffenet->blob_by_name("data");
			float * input_data = input_blob->mutable_cpu_data();
			vector<Mat> mat_vec;
			cv::split(img, mat_vec);//(224,224,CV_32FC3,input_data);
			float * data_ptr = new float[input_blob->count()];		
  			for (int i = 0; i < 3; ++i) {
	  			float* src_ptr = reinterpret_cast<float *>(mat_vec[i].data);
				caffe::caffe_copy(img.total(), src_ptr, data_ptr+img.total()*i);
				//memcpy(data_ptr+img.total()*i,src_ptr,img.total());
			}
			caffe::caffe_copy(input_blob->count(),data_ptr,input_data);
			delete data_ptr;
			
			std::vector<caffe::Blob<float>* > input_vec;
			m_caffenet->Forward(input_vec);
			boost::shared_ptr<caffe::Blob<float> > rpn_cls_blob = m_caffenet->blob_by_name("rpn_cls_prob_reshape");
			//const float * rpn_cls = rpn_cls_blob->cpu_data();
			//for(int i=0;i<10;i++)
			//	cout<<*(rpn_cls+i)<<endl;
			//cout<<rpn_cls_blob->shape_string()<<endl;
			boost::shared_ptr<caffe::Blob<float> > rpn_bbox_blob = m_caffenet->blob_by_name("rpn_bbox_pred");
			//const float * rpn_bbox = rpn_bbox_blob->cpu_data();
			//cout<<rpn_cls_blob->shape_string()<<endl;
			
			//vector<float> scores;
			//vector< vector<int> > rois;
			ProposalLayerForward(img.rows, img.cols, rpn_cls_blob, rpn_bbox_blob, rois, scores, TEXT_PROPOSALS_MIN_SCORE);
		
			return true;
		}
		~TextProposalDetector(){};
	private:
		boost::shared_ptr< Net<float> > m_caffenet;
};

vector< vector<int> > detect_tline(Mat img)
{
	string deployfile = "models/deploy.prototxt";
	string modelfile = "models/ctpn_trained_model.caffemodel";
	boost::shared_ptr< Net<float> > net(new Net<float>(deployfile,caffe::TEST));
	net->CopyTrainedLayersFrom(modelfile);
	TextProposalDetector tpd(net);

	vector< vector<int> > tline;
	vector< vector<float> > rois;
	vector<float> scores;

	if(!tpd.detect(img,rois,scores))
		return tline;

	map<float, int> scores_index;
	for(int i=0; i<scores.size(); i++)
		scores_index.insert(pair<float,int>(scores[i],i));
	
	vector< vector<float> > roi = rois;
	int i=scores.size()-1;
	for(auto it=scores_index.begin(); it!=scores_index.end(); it++)
	{
		scores[i] = it->first;
		rois[i] = roi[it->second];
		i--;
	}
		//cout<<i++<<','<<it->second<<endl;
	
	// nms for text proposals
	vector<int> keep_idx;
	keep_idx = nms(rois,scores,TEXT_PROPOSALS_NMS_THRESH);
	for(int i=0; i<keep_idx.size(); i++)
	{
		scores[i] = scores[keep_idx[i]];
		rois[i] = rois[keep_idx[i]];
	}
	rois.resize(keep_idx.size());
	scores.resize(keep_idx.size());

	//normalize
	int maxidx=0,minidx=0;
	for(int i=0; i<scores.size(); i++)
	{
		if(scores[i] > scores[maxidx])
			maxidx = i;
		if(scores[i] < scores[minidx])
			minidx = i;
	}
	if(scores[maxidx] == scores[minidx])
	{
		for(int i=0; i<scores.size(); i++)
			scores[i] -= scores[minidx];
	}
	else
	{
		for(int i=0; i<scores.size(); i++)
			scores[i]=(scores[i]-scores[minidx])/(scores[maxidx]-scores[minidx]);
	}
	//connect
	vector< vector<float> >	tlines;
	tlines = get_text_lines(rois,scores,img.rows,img.cols);
	//nms for text lines
	
	return tline;
}
