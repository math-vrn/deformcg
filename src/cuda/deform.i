/*interface*/
%module deform

%{
#define SWIG_FILE_WITH_INIT
#include "deform.cuh"
%}

%include "numpy.i"

%init %{
import_array();
%}

class deform
{

public:
  %immutable;
  size_t n;
  size_t ntheta;
  size_t nz; 
  %mutable;
  deform(size_t ntheta, size_t nz, size_t n);
  ~deform();  
  void free();
  void remap(size_t res, size_t f, size_t x, size_t y);
  void registration(size_t flow, size_t f, size_t g, int numLevels, double pyrScale, bool fastPyramids, int winSize, int numIters, int polyN, double polySigma, int flags);
};
