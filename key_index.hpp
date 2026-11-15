#pragma once

#include <cstddef>
#include <bit>
#include <cstdint>
#include <type_traits>
#include <concepts>
#include <utility>
#include <simd>
#include "key_storage.hpp"
#include "util.hpp"

template <typename T>
concept IndexableKey = std::is_trivially_copyable_v<T> &&
                       std::has_single_bit(sizeof(T)) &&
                       sizeof(T) <= 8;

template <IndexableKey T, size_t Size, size_t Capacity = Size>
class KeyIndex 
{
public:
    static constexpr bool IsFixed = Capacity > 0 && Size == Capacity;
 
    constexpr KeyIndex(std::convertible_to<T> auto &&...args) 
    requires(sizeof...(args) == Size || (Capacity == 0 && sizeof...(args) > 0)) : 
        storage(std::forward<decltype(args)>(args)...)
    {
    }

    constexpr size_t capacity() const { return storage.capacity(); }
    constexpr size_t size() const { return storage.size(); }

    size_t find(T t) const
    {
        using IntType = std::tuple_element_t<std::countr_zero(sizeof(T)), std::tuple<int8_t, int16_t, int32_t, int64_t>>;
        using VecType = std::simd::vec<IntType>;
        using MaskType = std::tuple_element_t<std::countr_zero(std::max<size_t>(VecType::size(), 8)) - 3, std::tuple<uint8_t, uint16_t, uint32_t, uint64_t>>;

        VecType v{std::bit_cast<IntType>(t)};

        auto compare = [&] [[gnu::always_inline]] (size_t index) -> MaskType
        {
            auto v0 = std::simd::unchecked_load<VecType>(std::span{reinterpret_cast<const IntType *>(&storage[index]), VecType::size()});

            if constexpr (!std::has_unique_object_representations_v<T> && !std::is_floating_point_v<T>)
            {
                union U 
                {
                    IntType mask;
                    T val;
                } u;

                u.mask = -1;
                __builtin_clear_padding(&u.val);

                auto diff = (v ^ v0) & VecType{u.mask};
                return (diff == VecType{}).to_ullong();
            }
            else
            {
                return (v==v0).to_ullong();
            }
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

    size_t emplace(T t)
    requires(!IsFixed)
    {
        if (size_t index = find(t); index < storage.size())
        {
            return index;
        }
        else
        {
            storage.emplace_back(t);
            return index;
        }
    }

private:
    KeyStorage<T, Size, Capacity> storage;
};

template <typename... Args>
requires(sizeof...(Args) > 0)
KeyIndex(Args &&...) -> KeyIndex<std::common_type_t<Args...>, sizeof...(Args)>;
