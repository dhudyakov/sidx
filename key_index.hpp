#pragma once
#include "key_storage.hpp"
#include <cstddef>
#include <tuple>
#include <bit>
#include <cstdint>
#include <type_traits>
#include <concepts>
#include <utility>
#include <immintrin.h>

template <typename T, size_t Size, size_t Capacity = Size>
requires(std::is_trivially_copyable_v<T> && std::has_single_bit(sizeof(T)) && sizeof(T) <= 8)
class KeyIndex 
{
public:
    static constexpr bool IsFixed = Capacity>0 && Size==Capacity;
 
    constexpr KeyIndex(std::convertible_to<T> auto &&...args) 
    requires(sizeof...(args) == Size || (Capacity == 0 && sizeof...(args) > 0)) : 
        storage(std::forward<decltype(args)>(args)...)
    {
        if constexpr (HasPadding)
        {
            for (size_t i = 0; i < storage.size(); ++i)
                __builtin_clear_padding(&storage[i]);
        }
    }

    constexpr size_t capacity() const { return storage.capacity(); }
    constexpr size_t size() const { return storage.size(); }

    size_t emplace(T t)
    requires(!IsFixed)
    {
        if (size_t index = find(t); index < storage.size())
        {
            return index;
        }
        else
        {
            auto &value = storage.emplace_back(t);
            if constexpr (HasPadding)
                __builtin_clear_padding(&value);
            return index;
        }
    }

private:
    KeyStorage<T, Size, Capacity> storage;

    using IntType = std::tuple_element_t<std::countr_zero(sizeof(T)), std::tuple<int8_t, int16_t, int32_t, int64_t>>;
    using MaskType = std::tuple_element_t<std::countr_zero(sizeof(T)), std::tuple<__mmask64, __mmask32, __mmask16, __mmask8>>;

    static constexpr bool HasPadding = !std::has_unique_object_representations_v<T> && !std::is_floating_point_v<T>;

    static __m512i broadcast(int8_t v) { return _mm512_set1_epi8(v); }
    static __m512i broadcast(int16_t v) { return _mm512_set1_epi16(v); }
    static __m512i broadcast(int32_t v) { return _mm512_set1_epi32(v); }
    static __m512i broadcast(int64_t v) { return _mm512_set1_epi64(v); }

    static MaskType compare(__m512i a, __m512i b)
    {
        if constexpr (sizeof(T) == 1)
            return _mm512_cmpeq_epi8_mask(a, b);
        else if constexpr (sizeof(T) == 2)
            return _mm512_cmpeq_epi16_mask(a, b);
        else if constexpr (sizeof(T) == 4)
            return _mm512_cmpeq_epi32_mask(a, b);
        else if constexpr (sizeof(T) == 8)
            return _mm512_cmpeq_epi64_mask(a, b);
    }

    static MaskType compare(__m512i a, __m512i b, MaskType mask)
    {
        if constexpr (sizeof(T) == 1)
            return _mm512_mask_cmpeq_epi8_mask(mask, a, b);
        else if constexpr (sizeof(T) == 2)
            return _mm512_mask_cmpeq_epi16_mask(mask, a, b);
        else if constexpr (sizeof(T) == 4)
            return _mm512_mask_cmpeq_epi32_mask(mask, a, b);
        else if constexpr (sizeof(T) == 8)
            return _mm512_mask_cmpeq_epi64_mask(mask, a, b);
    }

public:
    size_t find(T t) const
    {
        constexpr size_t Lanes = 64 / sizeof(T);
        if constexpr (HasPadding)
            __builtin_clear_padding(&t);

        __m512i match_v = broadcast(std::bit_cast<IntType>(t));
        size_t i = 0;

        for (; i + Lanes <= storage.size(); i += Lanes)
        {
            if (MaskType mask = compare(_mm512_loadu_si512(reinterpret_cast<const __m512i *>(&storage[i])), match_v); mask != 0)
                return i + std::countr_zero(mask);
        }

        auto process_tail = [&](size_t index, MaskType tail_mask)
        {
            if (MaskType mask = compare(_mm512_loadu_si512(reinterpret_cast<const __m512i *>(&storage[index])), match_v, tail_mask); mask != 0)
                return index + std::countr_zero(mask);
            return storage.size();
        };

        if constexpr (IsFixed)
        {
            if constexpr (Size % Lanes != 0)
            {
                constexpr MaskType mask = (static_cast<MaskType>(1) << (Size % Lanes)) - static_cast<MaskType>(1);
                return process_tail(i, mask);
            }
        }
        else
        {
            if (i < storage.size())
            {
                MaskType mask=[](unsigned shift) -> MaskType
                {
                    if constexpr (sizeof(T) == 1)
                    {
                        return _bextr_u64(0xFFFFFFFFFFFFFFFFULL, 64 - shift, 64);
                    }
                    else
                    {
                        return _bextr_u32(0xFFFFFFFF, 32 - shift, 32);
                    }
                }(storage.size()-i);
                return process_tail(i, mask);
            }
        }
        return storage.size();
    }
};

template <typename... Args>
requires(sizeof...(Args) > 0)
KeyIndex(Args &&...) -> KeyIndex<std::common_type_t<Args...>, sizeof...(Args)>;
