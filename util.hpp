#pragma once
#include <cstddef>
#include <tuple>
#include <utility>
#include <type_traits> 

template <template <typename...> typename Target,size_t Stride, size_t Offset, typename... Args>
using stride_apply_t = decltype([]<size_t... Is>(std::index_sequence<Is...>) 
{
    return std::type_identity<Target<std::tuple_element_t<Offset + Is * Stride, std::tuple<Args...>>...>>{};
}(std::make_index_sequence<(sizeof...(Args) + Stride - 1 - Offset) / Stride>()))::type;
