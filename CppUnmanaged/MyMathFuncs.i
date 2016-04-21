%module CppUnmanaged

%{
#include "MathFuncs.h"
%}

%include <windows.i>
%include "MathFuncs.h"