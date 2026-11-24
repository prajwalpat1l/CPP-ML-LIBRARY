#pragma once

// Per-op profiler, compiled in only when ML_PROFILE is defined. Otherwise the
// macros expand to nothing.
//
//   ML_PROFILE_SCOPE("Matrix::matmul");  times the rest of the scope
//   ML_PROFILE_FLOPS(2 * m * n * q);     adds flops to the innermost scope
//   ML_PROFILE_THREADS(t);               records that this call ran on t threads
//
// Each name keeps its call count, total time and self time (total minus
// nested profiled scopes). Single-threaded only.

#ifdef ML_PROFILE

#include <chrono>
#include <cstdint>
#include <deque>
#include <map>
#include <string>
#include <vector>

namespace profiler {

inline constexpr bool enabled = true;

struct Entry {
    std::string name;
    std::uint64_t calls = 0;
    std::int64_t total_ns = 0;
    std::int64_t self_ns = 0;
    std::uint64_t flops = 0;
    std::map<int, std::uint64_t> thread_calls; // thread count -> calls
};

// A deque, so references to entries stay valid as new names are added.
inline std::deque<Entry>& registry() {
    static std::deque<Entry> entries;
    return entries;
}

inline Entry& find_entry(const char* name) {
    for (Entry& e : registry()) {
        if (e.name == name) {
            return e;
        }
    }
    registry().push_back(Entry{name});

    return registry().back();
}

class Timer;

inline Timer*& current() {
    static Timer* top = nullptr;
    return top;
}

class Timer {
private:
    using clock = std::chrono::steady_clock;

    Entry& entry;
    Timer* parent;
    std::int64_t child_ns = 0;
    clock::time_point start;

public:
    explicit Timer(Entry& e) : entry(e), parent(current()) {
        current() = this;
        start = clock::now();
    }

    ~Timer() {
        std::int64_t elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - start).count();
        entry.calls++;
        entry.total_ns += elapsed;
        entry.self_ns += elapsed - child_ns;
        if (parent) {
            parent -> child_ns += elapsed;
        }
        current() = parent;
    }

    void add_flops(std::uint64_t n) {
        entry.flops += n;
    }

    void add_threads(int t) {
        entry.thread_calls[t]++;
    }

    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;
};

inline void reset() {
    for (Entry& e : registry()) {
        e.calls = 0;
        e.total_ns = 0;
        e.self_ns = 0;
        e.flops = 0;
        e.thread_calls.clear();
    }
}

inline void add_flops(std::uint64_t n) {
    if (current()) {
        current() -> add_flops(n);
    }
}

inline void add_threads(int t) {
    if (current()) {
        current() -> add_threads(t);
    }
}

inline std::vector<Entry> entries() {
    std::vector<Entry> result;
    for (const Entry& e : registry()) {
        if (e.calls > 0) {
            result.push_back(e);
        }
    }

    return result;
}

}

// Two levels, so __LINE__ is expanded before it is pasted.
#define ML_PROFILE_CAT_(a, b) a##b
#define ML_PROFILE_CAT(a, b) ML_PROFILE_CAT_(a, b)
#define ML_PROFILE_SCOPE(name) \
    static ::profiler::Entry& ML_PROFILE_CAT(profile_entry_, __LINE__) = ::profiler::find_entry(name); \
    ::profiler::Timer ML_PROFILE_CAT(profile_timer_, __LINE__)(ML_PROFILE_CAT(profile_entry_, __LINE__))
#define ML_PROFILE_FLOPS(n) ::profiler::add_flops(static_cast<std::uint64_t>(n))
#define ML_PROFILE_THREADS(t) ::profiler::add_threads(static_cast<int>(t))

#else

namespace profiler {
inline constexpr bool enabled = false;
}

#define ML_PROFILE_SCOPE(name) ((void)0)
#define ML_PROFILE_FLOPS(n) ((void)0)
#define ML_PROFILE_THREADS(t) ((void)0)

#endif
