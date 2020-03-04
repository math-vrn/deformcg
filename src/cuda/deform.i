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
};
