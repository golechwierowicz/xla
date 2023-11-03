#include "torch_xla/csrc/runtime/thread_pool.h"

#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>

#include "torch_xla/csrc/runtime/metrics.h"
#include "torch_xla/csrc/runtime/tf_logging.h"
#include "tsl/platform/threadpool.h"

namespace torch_xla {
namespace runtime {
namespace env {
namespace {

tsl::thread::ThreadPool* GetThreadPool() {
  static size_t num_threads = sys_util::GetEnvInt(
      "XLA_THREAD_POOL_SIZE", std::thread::hardware_concurrency());
  static tsl::thread::ThreadPool pool(tsl::Env::Default(), "pytorchxla",
                                      num_threads);
  return &pool;
}

}  // namespace

class Completion::Data {
 public:
  void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return completed_; });
    if (exptr_ != nullptr) {
      std::rethrow_exception(exptr_);
    }
  }

  static std::function<void()> GetCompleter(std::shared_ptr<Data> data,
                                            std::function<void()> closure) {
    auto closure_wrapper = [closure = std::move(closure), data]() {
      std::exception_ptr exptr;
      try {
        closure();
      } catch (...) {
        exptr = std::current_exception();
      }
      data->Complete(exptr);
    };
    return closure_wrapper;
  }

 private:
  void Complete(std::exception_ptr exptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    exptr_ = std::move(exptr);
    completed_ = true;
    cv_.notify_all();
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  bool completed_ = false;
  std::exception_ptr exptr_;
};

Completion::Completion(std::shared_ptr<Data> data) : data_(std::move(data)) {}

Completion::~Completion() {}

void Completion::Wait() { data_->Wait(); }

void ScheduleClosure(std::function<void()> closure) {
  GetThreadPool()->Schedule(std::move(closure));
}

void ScheduleIoClosure(std::function<void()> closure) {
  GetThreadPool()->Schedule(std::move(closure));
}

Completion ScheduleClosureWithCompletion(std::function<void()> closure) {
  auto data = std::make_shared<Completion::Data>();
  GetThreadPool()->Schedule(
      Completion::Data::GetCompleter(data, std::move(closure)));
  return Completion(std::move(data));
}

Completion ScheduleIoClosureWithCompletion(std::function<void()> closure) {
  auto data = std::make_shared<Completion::Data>();
  GetThreadPool()->Schedule(
      Completion::Data::GetCompleter(data, std::move(closure)));
  return Completion(std::move(data));
}

}  // namespace env
}  // namespace runtime
}  // namespace torch_xla
