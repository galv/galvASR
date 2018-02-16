#include <stdexcept>

namespace galvASR {

AlphaComputer::AlphaComputer(const fst::ConstFst<fst::StdArc> &command_HCLG_graph,
                             std::size_t min_command_length_output_frames,
                             std::size_t max_command_length_output_frames)
  : min_command_length_output_frames_(min_command_length_output_frames),
    max_command_length_output_frames_(max_command_length_output_frames),
    // reversed_phrase_fst_(),
    // fst_start_state_(command_HCLG_graph.StartState()),
    // fst_final_state_(),
    acoustic_likelihood_buffer_(),
    M_queue_() {
  GALVASR_ASSERT(min_command_length_output_frames_ >= 1);
  // Note: fst::Reverse does _not_ preserve state ID's! Thus, we
  // cannot get the start state from the input FST. We must get it
  // from the reversed FST.
  fst::Reverse(command_HCLG_graph, reversed_command_fst_);
  fst_start_state_ = AlphaDecoder::GetStartState(reversed_command_fst_);
  fst_final_state_ = reversed_command_fst_.StartState();

  for(std::size_t i = 0; i < max_command_length_output_frames_ + 1; ++i) {
    // one row for each time step. Every M matrix has the same number
    // of rows, even though only the one at the head of the queue will
    // use all of them. This is so that we don't have to append new
    // rows as matrices are advanced along the queue.
    int32 num_rows = max_command_length_output_frames_ + 1;
    int32 num_columns = reversed_command_fst_.NumStates();

    auto M = new kaldi::FloatMatrix(num_rows, num_columns, kUndefined);
    M->Set(std::numeric_limits<float>::infinity());
    M->Index(0, fst_start_state_) = 0.0f;
    M_queue_.emplace_back(std::make_unique, M);
  }
}

void AlphaComputer::ResetDecoder() {
  acoustic_likelihood_buffer_.clear();

  for(auto M_iter = M_queue_->begin(); M_iter != M_queue->end(); ++M_iter) {
    M_iter->Set(std::numeric_limits<float>::infinity());
    M_iter->Index(0, fst_start_state_) = 0.0f;
  }
}

void AlphaComputer::UpdateMForOutputFrame(const kaldi::Vector& next_acoustic_nll) {
  // TODO: Pop final element, push it to the front.
  auto last = M_queue_.back();
  M_queue_.pop_back();
  M_queue_.push_front(last);

  for(std::size_t i = 0; last_time < M_queue_.size(); ++i) {
    auto M_ptr = M_queue_[i];
    GALVASR_ASSERT(M_queue_.NumCols() == reversed_command_fst_.NumStates());
    std::size_t next_row_to_fill_for_this_M = i + 1;

    const float* previous_M_row = M_ptr->RowData(next_row_to_fill_for_this_M - 1);
    float* this_M_row = M_ptr->RowData(next_row_to_fill_for_this_M);
    for(std::size_t j = 0; j < M_ptr.NumCols(); ++j) {
      float min_negative_log_likelihood = std::numeric_limits<float>::infinity();
      for (fst::ArcIterator<fst::StdFst> aiter(fst, j); !aiter.Done(); aiter.Next()) {
        const Arc &arc = aiter.Value();
        // TODO: Figure out if we need acoustic scale here!
        min_negative_log_likelihood =
          std::min(acoustic_scale * next_acoustic_nll[arc.id] + arc.Weight() +
                   previous_M_row[arc.Destination()],
                   min_negative_log_likelihood);
      }
      this_M_row[j] = min_negative_log_likelihood;
    }
  }
}

void AlphaComputer::UpdateMForOutputFrame(const kaldi::Vector& next_acoustic_nll) {
  if (acoustic_likelihood_buffer_.size() < min_phrase_length_output_frames_) {
    kaldi::Vector *copy = std::make_unique(
      new kaldi::Vector(next_acoustic_nll));
    acoustic_likelihood_buffer_.push_back(copy);
    // Nothing to do yet...
    return;
  } else if (acoustic_likelihood_buffer_.size() < max_phrase_length_output_frames_) {
    kaldi::Vector *copy = std::make_unique(
      new kaldi::Vector(next_acoustic_nll));
    acoustic_likelihood_buffer_.push_back(copy);

    AdvanceMOneFrame();
  } else if (acoustic_likelihood_buffer_.size() == max_phrase_length_output_frames_) {
    kaldi::Vector *copy = std::make_unique(
      new kaldi::Vector(next_acoustic_nll));
    acoustic_likelihood_buffer_.push_back(copy);
    acoustic_likelihood_buffer_.pop_front();

    AdvanceMOneFrame();
  } else {
    throw std::logic_error();
  }
  // otherwise, we have enough data to at 
}

// May want to make this really just output the M matrices. That way,
// I can inspect them. Goody, goody. Because we need a mechanism to
// take these values and compute the right decision boundary
// threshold.
void AlphaComputer::CheckForCommand() {
  for(unsigned int num_frames_to_check = min_phrase_length_output_frames_;
      num_frames_to_check < acoustic_likelihood_buffer_.size();
      ++num_frames_to_check) {
    // initialization
    std::pair<MatrixIndexT, float> *first_sparse_row = M_.Data().Data();
    GALVASR_ASSERT(sparse_row->first == 0);
    // Everything else should be initialized to +inf. Whose
    // responsibility is that? Probably the responsibility of the code
    // that initializes the matrix appropriately with the right
    // non-sparse elements. Wow, such wow.
    sparse_row->second = 0.0f;
    // Problem: we actually have a fairly sparse vector of likelihoods
    // when you consider that most states won't show up in this
    // FST. What to do? CSC matrix would make the most sense.
  }
}

const std::deque<std::unique_ptr<kaldi::Matrix>>& AlphaDecoder::GetM() const {
  return M_queue_.get();
}

void AlphaDecoder::AdvanceMOneFrame() {
  // Does auto infer reference types?
  auto previous_max_num_frames_M = M_queue_.back();
  M_queue_.pop_back();
  M_queue_.push_front(previous_max_num_frames_M);

  // Handle first case by itself
  M_iter->Set(std::numeric_limits<float>::infinity());
  M_iter->Index(0, fst_start_state_) = 0.0f;
  
  for(auto M_iter = M_queue_.begin(), std::advance(M_iter, 1),
      std::size_t num_frames = min_phrase_length_output_frames_ + 1;
      M_iter != M_queue.end() && num_frames < max_phrase_length_output_frames_ + 1;
      ++M_iter, ++num_frames) {
    auto M_ref = *M_iter;

    float* previous_M_row = M_ref->RowData();
    float* this_M_row = M_ref->RowData();
    for (StateId state = 0; state < reversed_command_fst_.NumStates(); ++state) {
      float min_negative_log_likelihood = std::numeric_limits<float>::infinity();
      for (ArcIterator<StdFst> aiter(fst, i); !aiter.Done(); aiter.Next()) {
        const Arc &arc = aiter.Value();
        // TODO: Figure out if we need acoustic scale here!
        min_negative_log_likelihood =
          std::min(acoustic_scale * acoustic_negative_log_likelihood_buffer_[what_time_to_use_unsure][arc.id] + arc.Weight() + previous_M_row[arc.Destination()],
            min_negative_log_likelihood);
      }
      this_M_row[state] = min_negative_log_likelihood;
    }
  }
}

static StateId AlphaDecoder::GetStartState(
  const fst::VectorFst<fst::StdArc>& reversed_phrase_fst) {
  // Would be good to move this to a static method at some point...
  StateId start_state = -1;
  {
    for (fst::StateIterator<fst::VectorFst<fst::StdArc>> state_iter(reversed_phrase_fst);
         !state_iter.Done(); state_iter.Next()) {
      if (state_iter.Value().Final() != fst::ConstFst<fst::StdArc>::Weight::Zero()) {
        if (start_state != -1) {
          throw std::invalid_argument("More than one final state in the input FST! Not allowed");
        } else {
          start_state = state_iter.Value();
        }
      }
    }
    if (start_state == -1) {
      throw std::invalid_argument("No final state in input FST! Is that even possible?");
    } else {
      return start_state;
    }
  }
}

} // namespace galvASR
