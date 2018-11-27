#pragma once

__global__ void count_repeats(const int *transcripts,
                              const int *transcript_offsets,
                              const int *transcript_lengths,
                              int *num_repeats_per_transcript) {
  // bool label_is_repeated = false;
  // for(int label_idx = threadIdx.x; label_idx < labelLengths[transcription_idx] - 1;
  //     label_idx += blockDim.x) {
  //   label_is_repeated =
  //     transcript_this_block[label_idx] == transcript_this_block[label_idx + 1];
  // }
  const int transcript_idx = blockIdx.x;
  const int label_idx = threadIdx.x;
#if !defined(__APPLE__)
  assert(blockDim.x >= transcript_lengths[transcript_idx] - 1);
#endif
  
  const int *transcript_this_block =
    &transcripts[transcript_offsets[transcript_idx]];

  bool label_is_repeated;
  if (label_idx < transcript_lengths[transcription_idx] - 1) {
    // Think about the memory coallescing implications of this
    label_is_repeated =
      transcript_this_block[label_idx] == transcript_this_block[label_idx + 1];
  } else {
    label_is_repeated = false;
  }

  int num_repeats = __syncthreads_count(label_is_repeated);

  if (threadIdx.x == 0) {
    num_repeats_per_transcript[blockIdx.x] = num_repeats;
  }
}

// Addition of two log probabilities in probability space, only to
// take the logarithm again
template<typename T>
__device__ inline T log_add_kill_inf(const T& a, const T&b) {
  if (a == -std::numeric_limits<T>::infinity()) {
    return b;
  } else if (b == -std::numeric_limits<T>::infinity()) {
    return a;
  } else {
    return log1p(exp(-abs(a - b))) + std::max(a, b);
  }
}

template<typename T, int VT>
__global__ void compute_alpha(
  const T * __restrict__ probs,
  const int * __restrict__ input_lengths,
  const int probs_max_input_length_stride,
  const int probs_max_label_length_stride,
  const int * __restrict__ transcripts,
  const int * __restrict__ transcript_offsets,
  const int * __restrict__ transcript_lengths,
  const int * __restrict__ repeats,
  T * __restrict__ alpha,
  T * __restrict__ nll) {
  // repeats: batch_size
  // alpha: batch_size x time x probs
  const int utterance_idx = blockIdx.x;
  const int transcript_length = transcript_lengths[utterance_idx];
  const int interleaved_transcript_length = 2 * transcript_lengths[utterance_idx] + 1;
  const int input_length = input_lengths[utterance_idx];
  assert(transcript_length + repeats <= input_length);

  const int transcript = &transcripts[transcript_offsets[utterance_idx]];
  
  auto probs_idx = [probs, ](t, label) -> T& {
    TODO
  };
  auto alpha_idx = [alpha, max_input_length, interleaved_transcript_length](t, s) -> T& {
    const T *alpha_this_utterance =
    &alpha[max_input_length * max_interleaved_transcript_length * utterance_idx];
    return alpha_this_utterance[interleaved_transcript_length * s + t];
  };

  // Do t == 0 separately
  {
    if (transcript_length + repeats == input_length) {
      if (threadIdx.x == 0) {
        alpha_idx(0, threadIdx.x + 1) = log(probs_idx(0, transcript[threadIdx.x]));
      }
    } else {
      if (threadIdx.x == 0 && transcript_length + repeats < input_length) {
        alpha_idx(0, threadIdx.x) = log(probs_idx(0, blank_label));
      } else if (threadIdx.x == 1) {
        alpha_idx(0, threadIdx.x) = log(probs_idx(0, transcript[threadIdx.x / 2]));
      }
    }
  }

  // Make previous writes to alpha visible
  __syncthreads();

  // (1) Next: Try to have two shared memory buffers, one for current and
  // other for previous alpha time step.

  // (2) Observe: Shared memory for labels
  // See if labels should be interleaved with blanks, in order to reduce divergence.
  
  // (3) Disable L1 cache for probs
  for (int t = 1; t < input_lengths[utterance_idx]; t++) {
    for (int interleaved_label_idx = threadIdx.x;
        interleaved_label_idx < interleaved_label_length;
        interleaved_label_idx += blockDim.x) {

      if (interleaved_label_idx == 0) {
        alpha_idx(t, interleaved_label_idx) =
          alpha_idx(t - 1, interleaved_label_idx) + log(probs_idx(t, blank_label));
      }
      // Two branches may be of no benefit at this point: We're not
      // doing full bank accesses anyway.
      int label = transcript[interleaved_label_idx >> 1];
      bool is_repeat = label & (1 << 31);
      label = label & 0x7fffffff;
      T alpha_bar = log_add_kill_inf(alpha_idx(t - 1, interleaved_label_idx),
                            alpha_idx(t - 1, interleaved_label_idx - 1));
      if (is_repeat || interleaved_label_idx == 1) {
        alpha_idx(t, interleaved_label_idx) =
          alpha_bar + log(probs_idx(t, label));
      } else if (interleaved_label_idx & 1 == 0 /* label is blank */) {
        // Must be a blank label
        alpha_idx(t, interleaved_label_idx) =
          alpha_bar + log(probs_idx(t, blank_label));
      } else if (interleaved_label_idx & 1 == 1 /* label is not blank and not a repeat */) {
        alpha_idx(t, interleaved_label_idx) =
          log_add_kill_inf(alpha_bar, alpha_idx(t - 1, interleaved_label_idx - 2)) +
          log(probs_idx(t, blank_label));
      } else {
        assert(false);
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    log_likelihood =
      log_add_kill_inf(alpha_idx(input_length - 1, interleaved_transcript_length - 1),
                       alpha_idx(input_length - 1, interleaved_transcript_length - 2));
    nll_forward[blockIdx.x]= - log_likelihood;
  }
}

template<typename T>
__global__ void compute_beta_and_grad(
  const T* __restrict__ probs,
  const int * __restrict__ input_lengths,
  const int probs_max_input_length_stride,
  const int alpha_max_label_length_stride,
  const int * __restrict__ transcripts,
  const int * __restrict__ transcript_offsets,
  const int * __restrict__ transcript_lengths,
  const int * __restrict__ repeats,
  T * __restrict__ alpha,
  T * __restrict__ nll) {
  
}
