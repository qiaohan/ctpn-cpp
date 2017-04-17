/*************************************************************************
    > File Name: text_proposal_layer.cpp
    > Author: ma6174
    > Mail: ma6174@163.com 
    > Created Time: Fri 03 Mar 2017 02:22:58 PM CST
 ************************************************************************/

#include<iostream>
#include <vector>
using namespace std;

inline float threshold(float x, int min_, int max_)
{
	float t = x>max_?max_:x;
	return t>min_?t:min_;
}

bool apply_deltas_to_anchors(vector< vector<float> >& res, vector<float>& bb_deltas, vector<float>& scores, int stride, int height, int width, int imgh, int imgw, float min_score )
{
	vector<int> heights={11, 16, 23, 33, 48, 68, 97, 139, 198, 283};
	vector<int> widths={16};
	vector<float> score = scores;
	scores.clear();
	int base_size = 16;
	int num_anchors = heights.size()*widths.size();
	//vector< vector<float> > res;
	//locate_anchors: 10x[4,] per feature pixel
	for(int hh=0; hh<height; hh++)
		for(int ww=0; ww<width; ww++)
		{
			// 10 anchors
			int x_ = ww*stride;
			int y_ = hh*stride;
			for(int h=0; h<heights.size(); h++)
				for(int w=0; w<widths.size(); w++)
				{
					int x1,y1,x2,y2;
					float x_ctr = base_size*0.5;
					float y_ctr = base_size*0.5;
					x1 = x_ctr - widths[w]/2 + x_;
					x2 = x_ctr + widths[w]/2 -1 + x_;
					y1 = y_ctr - heights[h]/2 + y_;
					y2 = y_ctr + heights[h]/2 -1 + y_;
					
					int anchor_y_ctr = (y1+y2)/2;
					int anchor_h = heights[h]+1;
					/*
					float delta0 = bb_deltas[ww*height*num_anchors+hh*num_anchors+h+w];
					float delta1 = bb_deltas[ww*height*num_anchors+hh*num_anchors+h+w+1];
					*/
					float delta0 = bb_deltas[2*hh*width*num_anchors+2*ww*num_anchors+2*(h+w)];
					float delta1 = bb_deltas[2*hh*width*num_anchors+2*ww*num_anchors+2*(h+w)+1];
					float global_coords1 = anchor_h*exp(delta1);
					float global_coords0 = delta0*anchor_h+anchor_y_ctr-global_coords1/2;
					if(score[hh*width*num_anchors+ww*num_anchors+(h+w)]<min_score)
						continue;
					//cout<<hh*width*num_anchors+ww*num_anchors+(h+w)<<','<<scores[hh*width*num_anchors+ww*num_anchors+(h+w)]<<endl;
					scores.push_back(score[hh*width*num_anchors+ww*num_anchors+(h+w)]);
					vector<float> rect;
					rect.push_back(threshold(x1,0,imgw-1));
					rect.push_back(threshold(global_coords0,0,imgh-1));
					rect.push_back(threshold(x2,0,imgw-1));
					rect.push_back(threshold(global_coords0+global_coords1,0,imgh-1));
					res.push_back(rect);
					//cout<<delta0<<','<<delta1<<endl;
					//cout<<rect[0]<<','<<rect[1]<<','<<rect[2]<<','<<rect[3]<<endl;	
					//cout<<x1<<','<<y1<<','<<x2<<','<<y2<<endl;	
				}
		}

	return true;
}

bool ProposalLayerForward(int originimgh, int originimgw,
		boost::shared_ptr<caffe::Blob<float> > rpn_cls_prob,
		boost::shared_ptr<caffe::Blob<float> > rpn_bbox_pred,
		vector< vector<float> >& rois,
		vector<float>& scores, float min_score)
{
	int _feat_stride = 16;
	int _num_anchors = 10;
	if( rpn_cls_prob->shape(0) != 1 )
		return false;
	
	const float* cls = rpn_cls_prob->cpu_data();
	const float* bbox = rpn_bbox_pred->cpu_data();
	const int imgw = rpn_cls_prob->shape(3);
	const int imgh = rpn_cls_prob->shape(2);

	const int ss = rpn_cls_prob->shape(1)-_num_anchors;	
	scores.resize( ss*imgh*imgw );
	for(int k=0; k<ss; k++)
		for(int i=0; i<imgh; i++)	
			for(int j=0; j<imgw; j++)
	{
		*( scores.data()+i*ss*imgw+j*ss+k ) = *( cls+(k+_num_anchors)*imgh*imgw+i*imgw+j );
	}
/*
	for(int i=0; i<10; i++)
		cout<<*(cls+imgh*imgw*_num_anchors+i)<<';';
	cout<<endl;
	for(int i=0; i<scores.size(); i++)
		cout<<scores[i]<<',';
*/
	cout<<endl;
	const int sb = rpn_bbox_pred->shape(1);
	vector<float> bb;
	bb.resize( sb*imgh*imgw );
	for(int k=0; k<sb; k++)
		for(int i=0; i<imgh; i++)	
			for(int j=0; j<imgw; j++)
	{
		*( bb.data()+i*sb*imgw+j*sb+k ) = *( bbox+k*imgh*imgw+i*imgw+j );
	}
	
	
	//for(int i=0; i<bb.size(); i++)
		//cout<<bb[i*2]<<','<<bb[i*2+1]<<endl;
		//cout<<bbox[i]<<endl;
		//cout<<bb[i]<<endl;
	
	
	return apply_deltas_to_anchors(rois,bb,scores,_feat_stride,imgh,imgw,originimgh,originimgw,min_score);
	//cout<<roi.size()<<endl;
/*
	for(int i=0; i<10; i++)
	{
		for(int j=0; j<roi[0].size(); j++)
			cout<<roi[i][j]<<',';
		cout<<endl;
		
	}
*/
	//return true;
}
