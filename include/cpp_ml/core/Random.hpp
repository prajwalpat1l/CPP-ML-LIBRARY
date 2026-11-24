#pragma once

#include <random>
#include <cstdint>
#include <cstddef>
#include <utility>

// Random numbers built only on raw std::mt19937 output, which the standard
// fixes. std::uniform_real_distribution and std::shuffle are not fixed, so
// they give different numbers on libstdc++, libc++ and MSVC.
namespace rng {

// Uniform double in [0, 1) from 53 random bits (genrand_res53).
inline double uniform01(std::mt19937& gen) {
    std::uint32_t a = gen() >> 5, b = gen() >> 6;
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

inline double uniform(std::mt19937& gen, double low, double high) {
    return low + (high - low) * uniform01(gen);
}

// Unbiased integer in [0, n), by rejection sampling.
inline std::uint64_t uniform_index(std::mt19937& gen, std::uint64_t n) {
    std::uint64_t limit = UINT64_MAX - (UINT64_MAX % n);
    std::uint64_t r;
    do {
        r = (static_cast<std::uint64_t>(gen()) << 32) | gen();
    } while (r >= limit);

    return r % n;
}

// Fisher-Yates.
template <typename It>
void shuffle(It first, It last, std::mt19937& gen) {
    size_t n = static_cast<size_t>(last - first);
    for (size_t i = n; i > 1; i--) {
        size_t j = static_cast<size_t>(uniform_index(gen, i));
        std::swap(first[i - 1], first[j]);
    }
}

}
