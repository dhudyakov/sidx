#include "key_index.hpp"
#include "key_index_map.hpp"

#include <iostream>
#include <cassert>
#include <array>
#include <string>
#include <vector>
#include <cstring>
#include <concepts>

template <typename T>
constexpr T make_scrambled_copy(T original, char scramble_pattern)
{
    constexpr char KnownPattern = static_cast<char>(0xFF);
    alignas(T) char mask_bytes[sizeof(T)];
    for (std::size_t i = 0; i < sizeof(T); ++i)
        mask_bytes[i] = KnownPattern;

    __builtin_clear_padding(reinterpret_cast<T *>(mask_bytes));

    for (std::size_t i = 0; i < sizeof(T); ++i)
        mask_bytes[i] ^= KnownPattern;

    T copy = original;
    char *ptr = reinterpret_cast<char *>(&copy);
    for (std::size_t i = 0; i < sizeof(T); ++i)
    {
        if (mask_bytes[i] == KnownPattern)
            ptr[i] ^= scramble_pattern;
    }
    return copy;
}

struct Padded8
{
    char a;
    int b;
    bool operator==(const Padded8 &) const = default;
};

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
void test_index_combinations()
{
    std::cout << "[Test 4] Sweeping FlatKeyIndex SIMD Combinations...\n";

    auto index_u8 = []<std::size_t... Is>(std::index_sequence<Is...>) 
    {
        return KeyIndex<uint8_t, 16>(static_cast<uint8_t>(Is + 5)...);
    }(std::make_index_sequence<16>{});

    assert(index_u8.find(5) == 0);
    assert(index_u8.find(20) == 15);

    Padded8 p1{'A', 100};
    Padded8 p2 = make_scrambled_copy(p1, 0x5A);

    KeyIndex index_padded(p1, Padded8{'B', 200}, Padded8{'C', 300});
    assert(index_padded.find(p2) == 0);

    std::cout << "-> SIMD Matrix and Padding Clearance swept successfully.\n\n";
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
