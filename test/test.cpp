/*************************************************************************
    > File Name: test.cpp
    > Author: ma6174
    > Mail: ma6174@163.com 
    > Created Time: Thu 02 Mar 2017 09:47:58 AM CST
 ************************************************************************/

#include<iostream>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
//#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
//#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"

#include "detector.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;

using namespace cv;
using namespace std;


int main(){
	
	Mat img = imread("test.png");
	vector< vector<int> > tline = detect_tline(img);
	return 0;
}
