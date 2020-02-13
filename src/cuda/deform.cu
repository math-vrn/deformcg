#include "deform.cuh"
#include "kernels.cuh"
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <iostream>
using namespace std;

deform::deform(size_t ntheta, size_t nz, size_t n) 
: ntheta(ntheta), nz(nz), n(n)
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

void deform::remap(size_t res_, size_t f_, size_t x_, size_t y_)
{
	float *res = (float *)res_;
	float *f = (float *)f_;
	float *x = (float *)x_;
	float *y = (float *)y_;

	cv::cuda::GpuMat res_gpu(n,nz, CV_32F,(float*)res);
	cv::cuda::GpuMat f_gpu(n,nz, CV_32F,(float*)f);
	cv::cuda::GpuMat x_gpu(n,nz, CV_32F,(float*)x);
	cv::cuda::GpuMat y_gpu(n,nz, CV_32F,(float*)y);
	cv::cuda::remap(f_gpu,res_gpu,x_gpu,y_gpu,2);//0 nn, 1 linear, 2 cubic	
}

void deform::registration(size_t flow_, size_t f_, size_t g_, int numLevels, double pyrScale, bool fastPyramids, int winSize, int numIters, int polyN, double polySigma, int flags)
{
	float *f = (float *)f_;
	float *g = (float *)g_;
	float *flow = (float *)flow_;
	
	cv::cuda::GpuMat flowxy_gpu[2];	
	cv::cuda::GpuMat f_gpu(n,nz, CV_32F, f);    	
	cv::cuda::GpuMat g_gpu(n,nz, CV_32F, g);    		
	cv::cuda::GpuMat flow_gpu(n,nz, CV_32FC2, flow);    		
	auto algo = cv::cuda::FarnebackOpticalFlow::create(numLevels, pyrScale, fastPyramids, winSize,numIters, polyN, polySigma, flags);
	algo->calc(f_gpu, g_gpu, flow_gpu);	
}