/*************************************************************************
    > File Name: ../src/nms.h
    > Author: ma6174
    > Mail: ma6174@163.com 
    > Created Time: Fri 07 Apr 2017 03:56:33 PM CST
 ************************************************************************/

#include<iostream>
using namespace std;

vector<int> nms(vector< vector<float> >& rois, vector<float>& scores, float thresh)
{
	vector<int> keep;
	vector<bool> suppressed;
	suppressed.resize(rois.size());
	for(int i=0; i<rois.size(); i++)
		suppressed[i] = false;
	for(int i=0; i<rois.size(); i++)
	{
		if(suppressed[i])
			continue;
		keep.push_back(i);
		float x1 = rois[i][0];
		float y1 = rois[i][1];
		float x2 = rois[i][2];
		float y2 = rois[i][3];
		float area = (x2 - x1 + 1) * (y2 - y1 + 1);
		for(int j=i+1; j<rois.size(); j++)
		{
			if(suppressed[j])
				continue;
			float xx1 = max(rois[j][0],x1);
			float yy1 = max(rois[j][1],y1);
			float xx2 = min(rois[j][2],x2);
			float yy2 = min(rois[j][3],y2);
			float areaj = (xx2 - xx1 + 1) * (yy2 - yy1 + 1);
			float w = xx2-xx1+1;
			w = w>0?w:0;
			float h = yy2-yy1+1;
			h = h>0?h:0;
			float inter = w*h;
			float ovr = inter / (area + areaj - inter);
			if(ovr>=thresh)
				suppressed[j]=true;
		}
	}
	//	cout<<scores[i]<<endl;
	return keep;
}
