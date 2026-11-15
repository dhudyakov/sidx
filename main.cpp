#include "key_index.hpp"
#include "key_index_map.hpp"

#include <iostream>
#include <cassert>
#include <array>
#include <string>
#include <vector>
#include <cstring>
#include <concepts>
#include <ranges>
#include <flat_set>
#include <meta>

template <typename T>
void scramble(T& val, uint8_t seed)
{
    if constexpr (!std::has_unique_object_representations_v<T>)
    {
        alignas(T) std::array<uint8_t, sizeof(T)> active_bits{};
        active_bits.fill(0xFF);
        __builtin_clear_padding(reinterpret_cast<T*>(active_bits.data()));

        auto bytes = std::bit_cast<std::array<uint8_t, sizeof(T)>>(val);
        for (std::size_t i = 0; i < sizeof(T); ++i)
        {
            uint8_t pad_mask = ~active_bits[i];
            bytes[i] ^= static_cast<uint8_t>((seed + i * 31) & pad_mask);
        }
        val = std::bit_cast<T>(bytes);
    }
}


// -------------------------------------------------------------------------
// Test Configuration 1: Pure Fixed-Size Static Layout Coverage
// -------------------------------------------------------------------------
void test_pure_static_map()
{
    std::cout << "[Test 1] Verifying Pure Fixed-Size Static Map...\n";

    auto static_map = KeyIndexMap(10, 'A', 20, 'B', 30, 'C');

    // Verify lookup functionality
    assert(static_map[10] == 'A');
    assert(static_map[30] == 'C');
    assert(static_map[99] == '\0'); // Default fallback slot hit

    constexpr bool can_emplace = []<typename M>(M &map)
    {
        if constexpr (requires { map.emplace(40, 'D'); })
        {
            return true;
        }
        else
        {
            return false;
        }
    }(static_map);

    constexpr bool can_assign = []<typename M>(M &map)
    {
        if constexpr (requires { map.insert_or_assign(40, 'D'); })
        {
            return true;
        }
        else
        {
            return false;
        }
    }(static_map);

    static_assert(!can_emplace, "Fixed static map should block emplace!");
    static_assert(!can_assign, "Fixed static map should block insert_or_assign!");

    std::cout << "-> Pure Fixed-Size Static Map constraints verified cleanly.\n\n";
}

// -------------------------------------------------------------------------
// Test Configuration 2: Hybrid Bounded Static Layout Coverage
// -------------------------------------------------------------------------
void test_hybrid_bounded_map()
{
    std::cout << "[Test 2] Verifying Hybrid Bounded Static Map...\n";

    KeyIndexMap<int, std::string, 2, 4> bounded_map(10, "Alpha", 20, "Beta");

    assert(bounded_map.size() == 2);
    assert(bounded_map.capacity() == 4);

    assert(bounded_map[10] == "Alpha");
    assert(bounded_map[20] == "Beta");

    bounded_map.emplace(30, "Gamma");
    assert(bounded_map.size() == 3);
    assert(bounded_map[30] == "Gamma");

    bounded_map.insert_or_assign(30, "GammaUpdated");
    assert(bounded_map[30] == "GammaUpdated");

    bounded_map.emplace(40, "Delta");
    assert(bounded_map.size() == 4);
    assert(bounded_map[40] == "Delta");

    bool caught_exception = [&]
    {
        try
        {
            bounded_map.emplace(50, "Omega");
        }
        catch (const std::bad_alloc &)
        {
            return true;
        }
        return false;
    }();
    assert(caught_exception && "Should have blocked insertion past Capacity=4 limit!");

    std::cout << "-> Hybrid Bounded Static Map metrics verified successfully.\n\n";
}

// -------------------------------------------------------------------------
// Test Configuration 3: Dynamic Layout Coverage
// -------------------------------------------------------------------------
void test_dynamic_map()
{
    std::cout << "[Test 3] Verifying Dynamic Map Layout...\n";

    KeyIndexMap<int, double, 0> dynamic_map(9.99);
    assert(dynamic_map[99] == 9.99);

    dynamic_map.emplace(5, 12.34);
    assert(dynamic_map[5] == 12.34);
    assert(dynamic_map[99] == 9.99);

    dynamic_map.emplace(5, 99.99);
    assert(dynamic_map[5] == 12.34);

    dynamic_map.insert_or_assign(5, 55.55);
    assert(dynamic_map[5] == 55.55);

    dynamic_map.insert_or_assign(42, 88.88);
    assert(dynamic_map[42] == 88.88);
    assert(dynamic_map[99] == 9.99);

    std::cout << "-> Dynamic Map Layout verified successfully.\n\n";
}

// -------------------------------------------------------------------------
// Test Configuration 4: SIMD Matrix and Padding Validation
// -------------------------------------------------------------------------
template <std::size_t MinVal, std::size_t MaxVal, std::size_t Lanes>
consteval auto make_boundary_sizes()
{
    std::flat_set<std::size_t> s;
    for (std::size_t x : {MinVal, 1uz, 2uz, 3uz, Lanes > 1 ? Lanes - 1 : 1uz, Lanes, Lanes + 1, 2 * Lanes - 1, 2 * Lanes, MaxVal})
        s.insert(std::clamp(x, MinVal, MaxVal));

    return std::define_static_array(s);
}

template <typename T>
void sweep_all_storage_modes(auto make_elem)
{
    constexpr std::size_t MaxCount = 128 / sizeof(T);
    constexpr std::size_t Lanes = 64 / sizeof(T);

    auto make_index = [&]<std::size_t N, std::size_t Cap = N>()
    {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>)
        {
            return KeyIndex<T, N, Cap>(make_elem(Is)...);
        }(std::make_index_sequence<N>{});
    };

    auto verify_contents = [&](const auto& index, std::size_t count)
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            T query = make_elem(i);
            if constexpr (!std::has_unique_object_representations_v<T>)
                scramble(query, 0xA5 ^ i);

            assert(index.find(query) == i);
        }
        assert(index.find(make_elem(count + 1)) == count);
    };

    // 1. Pure Fixed-Size Static Storage: KeyStorage<T, N, N> -> [1, MaxCount]
    template for (constexpr auto N : make_boundary_sizes<1, MaxCount, Lanes>())
    {
        auto static_idx = make_index.template operator()<N>();
        verify_contents(static_idx, N);
    }

    // 2. Hybrid Bounded Static Storage: KeyStorage<T, InitSize, MaxCount> -> [0, MaxCount - 1]
    template for (constexpr auto N : make_boundary_sizes<0, MaxCount-1, Lanes>())
    {
        auto bounded_idx = make_index.template operator()<N, MaxCount>();
        verify_contents(bounded_idx, N);

        for (std::size_t n = N; n < MaxCount; ++n)
        {
            bounded_idx.emplace(make_elem(n));
            verify_contents(bounded_idx, n + 1);
        }
    }

    // 3. Dynamic Heap Storage: KeyStorage<T, 0, 0>
    {
        KeyIndex<T, 0, 0> dynamic_idx;
        verify_contents(dynamic_idx, 0);

        for (std::size_t n = 0; n < MaxCount; ++n)
        {
            dynamic_idx.emplace(make_elem(n));
            verify_contents(dynamic_idx, n + 1);
        }
    }
}

void test_index_combinations()
{
    std::cout << "[Test 4] Sweeping All KeyStorage Modes & Types up to 128 bytes...\n";

    template for (constexpr auto R : std::array{^^uint8_t, ^^uint16_t, ^^uint32_t, ^^uint64_t})
    {
        using T = typename [:R:];
        sweep_all_storage_modes<T>([](std::size_t i) -> T
        {
            return i * 7 + 1;
        });
    }
    
    struct Padded2
    {
        char a;
        bool b;
        Padded2() = default;
        Padded2(std::size_t i) : a(i + 1), b((i & 1) == 0) {}
    };

    struct Padded4
    {
        char a;
        short b;
        Padded4() = default;
        Padded4(std::size_t i) : a(i + 1), b(i & 0x7F) {}
    };

    struct Padded8
    {
        char a;
        int b;
        Padded8() = default;
        Padded8(std::size_t i) : a(i + 1), b(i & 0x777F) {}
    };

    struct Bitfield2
    {
        uint8_t a;
        uint8_t b : 1;
        uint8_t   : 1; 
        uint8_t c : 1;
        uint8_t   : 1; 
        uint8_t d : 1;
        Bitfield2() = default;
        Bitfield2(std::size_t i) : a(i + 1), b(i & 1), c((i >> 1) & 1), d((i >> 2) & 1) {}
    };

    struct Bitfield4
    {
        char a;
        uint16_t b : 10;
        Bitfield4() = default;
        Bitfield4(std::size_t i) : a(i + 1), b(i % 1024) {}
    };

    struct Bitfield8
    {
        uint8_t a : 4;
        uint32_t b : 20;
        Bitfield8() = default;
        Bitfield8(std::size_t i) : a(i + 1), b(i * 111) {}
    };

    template for (constexpr auto R : std::array{^^Padded2, ^^Padded4, ^^Padded8, ^^Bitfield2, ^^Bitfield4, ^^Bitfield8})
    {
        using T = typename [:R:];
        sweep_all_storage_modes<T>([]<typename U = T>(std::size_t i) -> U
        {
            U v(i);
            scramble(v, 0x5A ^ i);
            return v;
        });
    }
    std::cout << "-> All KeyStorage specializations and SIMD widths verified cleanly.\n\n";
}

int main()
{
    try
    {
        test_pure_static_map();
        test_hybrid_bounded_map();
        test_dynamic_map();
        test_index_combinations();
        std::cout << "=== ALL TEST CRITERIA CONCLUDED SUCCESSFULLY ===\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Test failed with unexpected exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
