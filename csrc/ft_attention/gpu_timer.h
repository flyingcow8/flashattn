#pragma once

#include <cuda_runtime.h>

namespace mmha {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

struct GpuTimer {

  cudaEvent_t events[2];

  //
  // Methods
  //

  GpuTimer();
  ~GpuTimer();

  /// Records a start event in the stream
  void start(cudaStream_t stream = nullptr);

  /// Records a stop event in the stream
  void stop(cudaStream_t stream = nullptr);

  /// Records a stop event in the stream and synchronizes on the stream
  void stop_and_wait(cudaStream_t stream = nullptr);

  /// Returns the duration in miliseconds
  double duration(int iterations = 1) const;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace mmha