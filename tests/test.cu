#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

int main (int argc, char* argv[])
{
    int n=128;
    int nz=128;
    //cv::Mat mres(n,nz, CV_32F,(float*)res);
	//cv::Mat mf(n,nz, CV_32F,(float*)f);
	//cv::Mat mx(n,nz, CV_32F,(float*)x);
	//cv::Mat my(n,nz, CV_32F,(float*)y);
    cv::cuda::GpuMat gres(n,nz, CV_32F);
    cv::cuda::GpuMat gf(n,nz, CV_32F);
    cv::cuda::GpuMat gx(n,nz, CV_32F);
    cv::cuda::GpuMat gy(n,nz, CV_32F);	
	//gres.upload(mres);
	//gf.upload(mf);
	//gx.upload(mx);
	//gy.upload(my);
	cv::cuda::remap(gf,gres,gx,gy,0,0);
	//gf.download(mres);
	//memcpy((float*)res,mres.ptr<float>(0),n*nz*sizeof(float));
    return 0;
}