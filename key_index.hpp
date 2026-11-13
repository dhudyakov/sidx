#pragma once

#include <cstddef>
#include <tuple>
#include <bit>
#include <cstdint>
#include <type_traits>
#include <concepts>
#include <utility>
#include <immintrin.h>
#include <simd>
#include "key_storage.hpp"

template <typename T, size_t Size, size_t Capacity = Size>
requires(std::is_trivially_copyable_v<T> && std::has_single_bit(sizeof(T)) && sizeof(T) <= 8)
class KeyIndex 
{
public:
    static constexpr bool IsFixed = Capacity > 0 && Size == Capacity;
 
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
    using MaskType = std::simd::mask<IntType>;

    static constexpr bool HasPadding = !std::has_unique_object_representations_v<T> && !std::is_floating_point_v<T>;

public:
    size_t find(T t) const
    {
        using V = std::simd::vec<IntType>;
        constexpr size_t Lanes = V::size();

        if constexpr (HasPadding)
            __builtin_clear_padding(&t);

        V v{std::bit_cast<IntType>(t)};
        size_t i = 0;

        for (; i + Lanes <= storage.size(); i += Lanes)
        {
            V chunk = std::simd::unchecked_load<V>(std::span{reinterpret_cast<const IntType *>(&storage[i]), Lanes});
            if (MaskType mask = (chunk == v); std::simd::any_of(mask))
                return i + std::simd::reduce_min_index(mask);
        }

        auto process_tail = [&](size_t index,uint64_t tail_mask)
        {
            V chunk = std::simd::unchecked_load<V>(std::span{reinterpret_cast<const IntType *>(&storage[index]), Lanes});
            if (MaskType mask = (chunk == v) && MaskType{tail_mask}; std::simd::any_of(mask))
                return index + std::simd::reduce_min_index(mask);
            return storage.size();
        };
        if constexpr (IsFixed)
        {
            if constexpr (Size % Lanes != 0)
            {
                constexpr uint64_t bits = (1ULL << (Size % Lanes)) - 1ULL;
                return process_tail(i, bits);
            }
        }
        else
        {
            if (i < storage.size())
            {
                uint64_t bits = [](unsigned shift)
                {
                    if constexpr (sizeof(T) == 1)
                    {
                        return _bextr_u64(0xFFFFFFFFFFFFFFFFULL, 64 - shift, 64);
                    }
                    else
                    {
                        return _bextr_u32(0xFFFFFFFF, 32 - shift, 32);
                    }
                }(storage.size() - i);

                return process_tail(i, bits);
            }
        }
        return storage.size();
    }
};

template <typename... Args>
requires(sizeof...(Args) > 0)
KeyIndex(Args &&...) -> KeyIndex<std::common_type_t<Args...>, sizeof...(Args)>;
