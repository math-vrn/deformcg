#include <cufft.h>
#include <npp.h>
class deform
{
  bool is_free = false;  
  NppStreamContext* nstreams;
  cudaStream_t* cstreams;

public:
  size_t n;
  size_t ntheta;
  size_t nz; 
  size_t ptheta;
  
  deform(size_t ntheta, size_t nz, size_t n, size_t ptheta);
  void remap(size_t g, size_t f, size_t flowx, size_t flow_y);

  ~deform();  
  void free();
};
