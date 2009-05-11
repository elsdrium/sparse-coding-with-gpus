/* Author: Anand Madhavan */
#ifndef __UTILS_H__
#define __UTILS_H__

#include "cutil.h"
#include <string>
#include "coreutils.hh"

namespace cpu {
class CpuEventTimer {
  float& _time;
  unsigned int timer;
  public:
  CpuEventTimer(float& time):_time(time), timer(0) { 
    CUDA_SAFE_CALL( cutCreateTimer(&timer) );
    CUDA_SAFE_CALL( cutStartTimer(timer) );
  }
  ~CpuEventTimer() { 
    CUDA_SAFE_CALL( cutStopTimer(timer) );
    _time = cutGetTimerValue(timer);
    CUT_SAFE_CALL(cutDeleteTimer(timer));
  }
};
}

#endif
