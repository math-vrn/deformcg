#include <cufft.h>

class deform
{
  bool is_free = false;    
public:
  size_t n;
  size_t ntheta;
  size_t nz; 
  
  deform(size_t ntheta, size_t nz, size_t n);
  ~deform();  
  void free();
  void remap(size_t res, size_t f, size_t x, size_t y);
  void registration(size_t flow, size_t f, size_t g, int numLevels, double pyrScale, bool fastPyramids, int winSize, int numIters, int polyN, double polySigma, int flags);
};
