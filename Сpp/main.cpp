#include <stdio.h>
#include <cstdlib>
#include "..\CudaUnman\CudaMathFuncs.h"

void func() {

}

void main() {
	MyCudaMathFuncs::Integrals *myclass = new MyCudaMathFuncs::Integrals();

	printf("Integral = %f\n", myclass->_Simpson_3_8(0, 10, 100000, func));
	system("pause");
}