#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <concepts>
#include <utility>

#include "key_index.hpp"
#include "value_storage.hpp"
#include "util.hpp"

template <size_t Stride, size_t Offset, typename... Args>
using stride_common_type = stride_apply_t<std::common_type_t, Stride, Offset, Args...>;


template <typename K, typename V, size_t Size, size_t Capacity = Size>
class KeyIndexMap
{
public:
    static constexpr bool IsFixed = KeyIndex<K, Size, Capacity>::IsFixed;

    constexpr KeyIndexMap(auto &&...args)
    requires(sizeof...(args) == Size * 2 || (Capacity == 0 && sizeof...(args) > 0 && sizeof...(args) % 2 == 0)) : 
        KeyIndexMap(std::make_index_sequence<Size>{}, std::forward_as_tuple(std::forward<decltype(args)>(args)...), V{})
    {
    }

    constexpr KeyIndexMap(V fallback, auto &&...args)
    requires(sizeof...(args) == Size * 2 || (Capacity == 0 && sizeof...(args) > 0 && sizeof...(args) % 2 == 0)) : 
        KeyIndexMap(std::make_index_sequence<Size>{}, std::forward_as_tuple(std::forward<decltype(args)>(args)...), std::move(fallback))
    {
    }

    constexpr decltype(auto) operator[](this auto &&self, K key)
    {
        size_t idx = std::forward<decltype(self)>(self).keys.find(key);
        return std::forward<decltype(self)>(self).values[idx];
    }

    constexpr size_t capacity() const { return keys.capacity(); }
    constexpr size_t size() const { return keys.size(); }

    V& emplace(K key, auto &&...args)
    requires(std::constructible_from<V, decltype(args)...> && !IsFixed)
    {
        size_t size = keys.size();
        if (size_t idx = keys.emplace(key); idx < size)
            return values[idx];
        else
            return values.insert(idx, std::forward<decltype(args)>(args)...);
    }

    V& insert_or_assign(K key, auto &&...args)
    requires(std::constructible_from<V, decltype(args)...> && !IsFixed)
    {
        size_t size = keys.size();
        if (size_t idx = keys.emplace(key); idx < size)
            return  values[idx] = V(std::forward<decltype(args)>(args)...);
        else
            return values.insert(idx, std::forward<decltype(args)>(args)...);
    }

private:
    KeyIndex<K, Size, Capacity> keys;
    ValueStorage<V, Capacity> values;


    template <std::size_t... Is,typename... Args>
    constexpr KeyIndexMap(std::index_sequence<Is...>, std::tuple<Args...>&& args, V &&fallback): 
        keys(std::get<Is * 2>(std::forward<decltype(args)>(args))...),
        values(std::get<Is * 2 + 1>(std::forward<decltype(args)>(args))..., std::forward<V>(fallback)) {}
};

template <typename... Args>
requires(sizeof...(Args) >= 2)
KeyIndexMap(Args &&...) -> KeyIndexMap<
    stride_common_type<2, sizeof...(Args) % 2, Args...>,
    stride_common_type<2, sizeof...(Args) % 2 + 1, Args...>,
    sizeof...(Args) / 2>;
