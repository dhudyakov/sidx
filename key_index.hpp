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


/*  
#include <meta>
#include <type_traits>

template <IndexableKey T>
consteval auto make_value_byte_mask()
{
    std::array<uint8_t, sizeof(T)> mask{};

    if constexpr (std::is_bounded_array_v<T>)
    {
        using Elem = std::remove_extent_t<T>;
        constexpr auto elem_mask = make_value_byte_mask<Elem>();
        for (std::size_t i = 0; i < std::extent_v<T>; ++i)
            for (std::size_t j = 0; j < sizeof(Elem); ++j)
                mask[i * sizeof(Elem) + j] = elem_mask[j];
    }
    else if constexpr (std::meta::is_class_type(^^T))
    {
        constexpr auto ctx = std::meta::access_context::current();

        // 1. Cover non-empty base subobjects
        template for (constexpr auto base : std::define_static_array(
            std::meta::bases_of(^^T, ctx)))
        {
            constexpr auto base_type = std::meta::type_of(base);
            if constexpr (!std::meta::is_empty(base_type))
            {
                using BaseT = typename [:base_type:];
                constexpr auto off = std::meta::offset_of(base);
                constexpr auto sub_mask = make_value_byte_mask<BaseT>();
                for (std::size_t i = 0; i < sizeof(BaseT); ++i)
                    mask[off.bytes + i] |= sub_mask[i];
            }
        }

        // 2. Cover non-static data members (including bit-fields and [[no_unique_address]])
        template for (constexpr auto mem : std::define_static_array(
            std::meta::nonstatic_data_members_of(^^T, ctx)))
        {
            constexpr auto mem_type = std::meta::type_of(mem);
            constexpr auto off = std::meta::offset_of(mem);

            if constexpr (std::meta::is_bit_field(mem))
            {
                constexpr std::size_t start_bit = off.bytes * 8 + off.bits;
                constexpr std::size_t bit_width = std::meta::bit_size_of(mem);
                for (std::size_t b = 0; b < bit_width; ++b)
                {
                    std::size_t abs_bit = start_bit + b;
                    mask[abs_bit / 8] |= static_cast<uint8_t>(1u << (abs_bit % 8));
                }
            }
            else if constexpr (!std::meta::is_empty(mem_type))
            {
                using MemberT = typename [:mem_type:];
                constexpr auto sub_mask = make_value_byte_mask<MemberT>();
                for (std::size_t i = 0; i < sizeof(MemberT); ++i)
                    mask[off.bytes + i] |= sub_mask[i];
            }
        }
    }
    else
    {
        mask.fill(0xFF);
    }

    return mask;
}
*/

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
        using ByteVec = std::simd::vec<int8_t>;

        VecType v{std::bit_cast<IntType>(t)};

        auto compare = [&] [[gnu::always_inline]] (size_t index) -> MaskType
        {
            if constexpr (!std::has_unique_object_representations_v<T> && !std::is_floating_point_v<T>)
            {
                union U 
                {
                    uint8_t bytes[sizeof(T)];
                    T val;
                } u;

                __builtin_memset(u.bytes, 0xFF, sizeof(T));
                __builtin_clear_padding(&u.val);

                auto elem_mask = [&u]->std::optional<uint64_t>
                {
                    uint64_t mask = 0;
                    for (size_t b = 0; b < sizeof(T); ++b)
                    {
                        if(u.bytes[b] == 0xFF)
                            mask |= uint64_t{1} << b;
                        else if(u.bytes[b] != 0x00)
                            return std::nullopt;
                    }
                    return mask;
                }();    
                        
                if(elem_mask)
                {
                    constexpr uint64_t Repeat = ~0ULL / ((1ULL << sizeof(T)) - 1);
                    auto full_mask = *elem_mask * Repeat; 
    
                    auto v0 = std::simd::unchecked_load<ByteVec>(std::span{reinterpret_cast<const int8_t *>(&storage[index]), ByteVec::size()});
                    return (std::bit_cast<VecType>(std::simd::select(std::simd::mask<int8_t>{full_mask}, std::bit_cast<ByteVec>(v) - v0, ByteVec{})) == VecType{}).to_ullong() ;
                }
                else
                {
                    auto v0 = std::simd::unchecked_load<VecType>(std::span{reinterpret_cast<const IntType *>(&storage[index]), VecType::size()});
                    auto diff = (v ^ v0) & VecType(std::bit_cast<IntType>(u.val));
                    return (diff == VecType{}).to_ullong();
                }
            }
            else
            {
                auto v0 = std::simd::unchecked_load<VecType>(std::span{reinterpret_cast<const IntType *>(&storage[index]), VecType::size()});
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
