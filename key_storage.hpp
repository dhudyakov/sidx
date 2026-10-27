#pragma once
#include <cstddef>
#include <utility>
#include <new>
#include <vector>
#include <concepts>
#include <cstdint>
#include <algorithm>

template <typename T, size_t Size, size_t Capacity = Size>
requires(Size <= Capacity)
struct KeyStorage
{
    constexpr KeyStorage(std::convertible_to<T> auto &&...args) 
    requires(sizeof...(args) == Size): 
        data{std::forward<decltype(args)>(args)...}, 
        count{sizeof...(args)} {}

    T &emplace_back(auto &&...args)
    requires(std::constructible_from<T, decltype(args)...>)
    {
        if (count >= Capacity)
            throw std::bad_alloc();

        return data[count++] = T(std::forward<decltype(args)>(args)...);
    }

    constexpr decltype(auto) operator[](this auto &&self, size_t index) { return std::forward<decltype(self)>(self).data[index]; }
    constexpr size_t size() const noexcept { return count; }
    constexpr size_t capacity() const noexcept { return Capacity; }

    alignas(64) T data[Capacity];
    [[no_unique_address]] uint16_t count;

};

template <typename T, size_t Size>
struct KeyStorage<T, Size, Size>
{
    constexpr KeyStorage(std::convertible_to<T> auto &&...args)
    requires(sizeof...(args) == Size): 
        data{std::forward<decltype(args)>(args)...} {}

    constexpr decltype(auto) operator[](this auto &&self, size_t index) { return std::forward<decltype(self)>(self).data[index]; }
    constexpr size_t size() const noexcept { return Size; }
    constexpr size_t capacity() const noexcept { return Size; }

    alignas(64) T data[Size];
};


template <typename T>
struct KeyStorage<T, 0, 0>
{
    KeyStorage() noexcept = default;

    explicit KeyStorage(std::convertible_to<T> auto &&...args)
    requires(sizeof...(args) >0): 
        data{std::forward<decltype(args)>(args)...} {}

    T &emplace_back(auto &&...args)
    requires(std::constructible_from<T, decltype(args)...>)
    {
        return data.emplace_back(std::forward<decltype(args)>(args)...);
    }

    constexpr decltype(auto) operator[](this auto &&self, size_t index) { return std::forward<decltype(self)>(self).data[index]; }
    constexpr size_t size() const noexcept { return data.size(); }
    constexpr size_t capacity() const noexcept { return data.max_size(); }
    
    struct AlignedAllocator 
    {
        using value_type = T;

        template <typename O> struct rebind { using other = AlignedAllocator; };

        T* allocate(std::size_t n) 
        {
            return n ? static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{64})) : nullptr;
        }

        void deallocate(T* p, [[maybe_unused]] std::size_t n) noexcept 
        { 
            ::operator delete(p, std::align_val_t{64}); 
        }

        bool operator==(const AlignedAllocator&) const noexcept { return true; }
    };

    std::vector<T, AlignedAllocator> data;
};
