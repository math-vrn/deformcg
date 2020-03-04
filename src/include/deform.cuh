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
};
