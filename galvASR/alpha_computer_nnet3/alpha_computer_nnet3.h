#pragma once

namespace galvASR {

class AlphaComputer {
public:
  using StateId = fst::VectorFst<fst::StdArc>::StateId;

  struct CommandSpotting {
    std::size_t num_frames_duration;
    float negative_log_likelihood;
  };

  AlphaComputer(const fst::ConstFst<fst::StdArc> &command_HCLG_graph,
                std::size_t min_command_length_output_frames,
                std::size_t max_command_length_output_frames);
  void ResetDecoder();
  void UpdateMForOutputFrame(const kaldi::Vector& acoustic_negative_log_likelihood_this_frame);
  // Get a reference to the M arrays. This will be modified if you
  // call UpdateMForOutputFrame() after calling GetM()! Also, this
  // reference will be invalid if the instance of this class is ever
  // destroyed.
  const std:deque<std::unique_ptr<kaldi::Matrix>>& GetM() const;
  bool CheckForCommand(float threshold) const;

private:

  void AdvanceMOneFrame();
  static StateId GetStartState();
  
  fst::VectorFst<fst::StdArc> reversed_command_fst_;
  // We can't just use reversed_phrase_fst_.Start() because that'd
  // actually be the end state, yikes!
  StateId fst_start_state_;
  StateId fst_final_state_;
  std::size_t min_command_length_output_frames_;
  std::size_t max_command_length_output_frames_;

  // Check with the threshold elsewhere...
  // float threshold;

  std::deque<std::unique_ptr<kaldi::Vector>> acoustic_negative_log_likelihood_buffer_;
  std::deque<std::unique_ptr<kaldi::Matrix>> M_queue_;
};

} // namespace galvASR
