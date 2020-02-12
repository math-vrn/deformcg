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

	cv::cuda::GpuMat flow_gpu;
	cv::cuda::GpuMat planes_gpu[2];
	cv::cuda::GpuMat f_gpu(n,nz, CV_32F, f);    	
	cv::cuda::GpuMat g_gpu(n,nz, CV_32F, g);    		

	auto algo = cv::cuda::FarnebackOpticalFlow::create();
	algo->calc(f_gpu, g_gpu, flow_gpu);	
	cv::cuda::split(flow_gpu, planes_gpu);	
	cudaMemcpy(&res[0],planes_gpu[0].ptr<float>(0),n*nz*sizeof(float),cudaMemcpyDefault);
	cudaMemcpy(&res[n*nz],planes_gpu[1].ptr<float>(0),n*nz*sizeof(float),cudaMemcpyDefault);
}