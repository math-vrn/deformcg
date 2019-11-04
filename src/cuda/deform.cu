#include "deform.cuh"
#include "kernels.cuh"
#include <stdio.h>

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

