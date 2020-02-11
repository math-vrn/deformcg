# deformcg
# deformcg

To install opecv+cuda from source:

conda create -n opencv python=3.7
conda install -c conda-forge scikit-build, ninja

git clone https://github.com/opencv/opencv
git clone https://github.com/opencv/opencv_contrib

(or release /opencv-4.2.0/)

cd opencv; mkdir build; cd build



cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/home/beams/VNIKITIN/sw/opencv \
    -DBUILD_PNG=ON \
    -DBUILD_TIFF=ON \
    -DBUILD_opencv_hdf=OFF \
    -DBUILD_TBB=ON \
    -DBUILD_JPEG=ON \
    -DBUILD_JASPER=ON \
    -DBUILD_ZLIB=ON \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_JAVA=OFF \
    -DBUILD_opencv_python2=OFF \
    -DBUILD_opencv_python3=ON \
    -DWITH_OPENCL=OFF \
    -DWITH_OPENMP=OFF \
    -DWITH_FFMPEG=OFF \
    -DWITH_GSTREAMER=OFF \
    -DWITH_GSTREAMER_0_10=OFF \
    -DWITH_CUDA=ON \
    -DWITH_GTK=OFF \
    -DWITH_VTK=OFF \
    -DWITH_TBB=OFF \
    -DWITH_1394=OFF \
    -DWITH_OPENEXR=OFF \
    -DCUDA_TOOLKIT_ROOT_DIR=/home/beams/VNIKITIN/sw/cuda-10.0/ \
    -DCUDA_BIN_PATH=/home/beams/VNIKITIN/sw/cuda-10.0/bin \
    -DCUDA_INCLUDE_PATH=/home/beams/VNIKITIN/sw/cuda-10.0/include \
    -DCUDA_LIB_PATH=/home/beams/VNIKITIN/sw/cuda-10.0/lib64 \
    -DCUDA_ARCH_BIN=6.0,6.1,7.5 \
    -DCUDA_ARCH_PTX="" \
    -DINSTALL_C_EXAMPLES=OFF \
    -DINSTALL_TESTS=OFF \
    -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.2.0/modules \
    ../../opencv-4.2.0


    export CUDACXX=/home/beams/VNIKITIN/sw/cuda-10.0/bin/nvcc
    CMAKE_PREFIX_PATH=/home/beams/VNIKITIN/sw/opencv/lib64/cmake/opencv4/ python setup.py install
    LD_LIBRARY_PATH=/home/beams/VNIKITIN/sw/opencv/lib64/:$LD_LIBRARY_PATH python test_adjoint.py