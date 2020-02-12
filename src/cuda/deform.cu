#include "deform.cuh"
#include "kernels.cuh"
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <iostream>
using namespace std;

deform::deform(size_t ntheta, size_t nz, size_t n) : ntheta(ntheta), nz(nz), n(n)
{
	
}

// destructor, memory deallocation
deform::~deform()
{
	free();
}

void deform::free()
{
	if (!is_free)
	{
		
		is_free = true;
	}
}

void deform::remap(size_t res, size_t f, size_t x, size_t y)
{
	cv::Mat mres(n,nz, CV_32F,(float*)res);
	cv::Mat mf(n,nz, CV_32F,(float*)f);
	cv::Mat mx(n,nz, CV_32F,(float*)x);
	cv::Mat my(n,nz, CV_32F,(float*)y);
    cv::cuda::GpuMat gres(n,nz, CV_32F);
    cv::cuda::GpuMat gf(n,nz, CV_32F);
    cv::cuda::GpuMat gx(n,nz, CV_32F);
    cv::cuda::GpuMat gy(n,nz, CV_32F);		
	gres.upload(mres);
	gf.upload(mf);
	gx.upload(mx);
	gy.upload(my);
	cv::cuda::remap(gf,gres,gx,gy,2);//0 nn, 1 linear, 2 cubic
	gres.download(mres);
	memcpy((float*)res,mres.ptr<float>(0),n*nz*sizeof(float));
}

void deform::registration(size_t res_, size_t f_, size_t g_)
{
	float *res = (float *)res_;
	float *f = (float *)f_;
	float *g = (float *)g_;
	auto algo = cv::cuda::FarnebackOpticalFlow::create();

	cv::Mat mflowx(n,nz, CV_32F);
	cv::Mat mflowy(n,nz, CV_32F);
	cv::Mat mf(n,nz, CV_32F,f);
	cv::Mat mg(n,nz, CV_32F,g);
	
	cv::cuda::GpuMat gflow;
	cv::cuda::GpuMat gf(n,nz, CV_32F);    	
	cv::cuda::GpuMat gg(n,nz, CV_32F);    	
	
	gf.upload(mf);
	gg.upload(mg);


	algo->calc(gf, gg, gflow);

	cv::cuda::GpuMat planes[2];
	cv::cuda::split(gflow, planes);
	planes[0].download(mflowx);
	planes[1].download(mflowy);			
	cout<<gflow.size()<<endl;
	cout<<mflowx.at<float>(0,0)<<mflowy.at<float>(50,50)<<endl;
	memcpy(&res[0],mflowx.ptr<float>(0),n*nz*sizeof(float));
	memcpy(&res[n*nz],mflowy.ptr<float>(0),n*nz*sizeof(float));
}