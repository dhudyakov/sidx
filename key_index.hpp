#pragma once

#include <cstddef>
#include <tuple>
#include <bit>
#include <cstdint>
#include <type_traits>
#include <concepts>
#include <utility>
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

    static constexpr bool HasPadding = !std::has_unique_object_representations_v<T> && !std::is_floating_point_v<T>;
    

public:
    size_t find(T t) const
    {
        using IntType = std::tuple_element_t<std::countr_zero(sizeof(T)), std::tuple<int8_t, int16_t, int32_t, int64_t>>;
        using VecType = std::simd::vec<IntType>;
        using MaskType = std::tuple_element_t<std::countr_zero(std::max<size_t>(VecType::size(), 8)) - 3, std::tuple<uint8_t, uint16_t, uint32_t, uint64_t>>;
         
        if constexpr (HasPadding)
            __builtin_clear_padding(&t);

        VecType v{std::bit_cast<IntType>(t)};
    
        auto compare = [&](size_t index) 
        {
            VecType chunk = std::simd::unchecked_load<VecType>(std::span{reinterpret_cast<const IntType *>(&storage[index]), VecType::size()});
            return (chunk == v).to_ullong();
        };
        
        constexpr size_t Lanes = VecType::size();
        size_t i = 0;

        for (; i + Lanes <= storage.size(); i += Lanes)
        {
            if (MaskType mask = compare(i); mask)
                return i + std::countr_zero(mask);  
        }

        if constexpr (IsFixed)
        {
            if constexpr (Size % Lanes != 0)
            {
                if (MaskType mask = compare(i) & (~0ULL >> (64 - (Size % Lanes))); mask)
                    return i + std::countr_zero(mask);  
            }
        }
        else
        {
            if (i < storage.size())
            {
                if (MaskType mask = compare(i) & (~0ULL >> (64 - (storage.size() - i))); mask)
                    return i + std::countr_zero(mask);  
            }
        }
        return storage.size();
    }
};

template <typename... Args>
requires(sizeof...(Args) > 0)
KeyIndex(Args &&...) -> KeyIndex<std::common_type_t<Args...>, sizeof...(Args)>;
