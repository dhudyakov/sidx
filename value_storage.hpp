#pragma once
#include <cstddef>
#include <utility>
#include <new>
#include <vector>
#include <concepts>
#include <cstdint>
#include <algorithm>

template <typename T, size_t Capacity>
struct ValueStorage
{
    constexpr ValueStorage(std::convertible_to<T> auto &&...args)
    requires(sizeof...(args) > 0): 
        data{std::forward<decltype(args)>(args)...}
    {
        constexpr size_t PackSize = sizeof...(args);
        if constexpr (PackSize < Capacity + 1)
        {
            std::fill(data + PackSize, data + Capacity + 1, data[PackSize - 1]);
        }
    }

    T& insert(size_t index, auto &&...args)
    requires(std::constructible_from<T, decltype(args)...>)
    {
        return data[index] = T(std::forward<decltype(args)>(args)...);
    }

    constexpr decltype(auto) operator[](this auto &&self, size_t index) { return std::forward<decltype(self)>(self).data[index]; }
    constexpr size_t size() const noexcept { return Capacity + 1; }

    T data[Capacity + 1];
};

template <typename T>
struct ValueStorage<T, 0>
{
    ValueStorage() noexcept = default;

    explicit ValueStorage(std::convertible_to<T> auto &&...args)
    requires(sizeof...(args) > 0): 
        data{std::forward<decltype(args)>(args)...} {}

    T& insert(size_t index, auto &&...args)
    requires(std::constructible_from<T, decltype(args)...>)
    {
        return *data.emplace(data.begin() + index, std::forward<decltype(args)>(args)...);
    }

    constexpr decltype(auto) operator[](this auto &&self, size_t index) { return std::forward<decltype(self)>(self).data[index]; }
    constexpr size_t size() const noexcept { return data.size(); }

    std::vector<T> data;
};
