// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <bit>
#include <concepts>
#include <ranges>

#include <algorithm>
#include <iostream>
#include <span>
#include <vector>

namespace glomap {

    // Concept to ensure we only work with trivially copyable types
    template <typename T>
    concept TriviallyCopyable = std::is_trivially_copyable_v<T>;

    // Reverse the order of bytes
    template <TriviallyCopyable T>
    [[nodiscard]] constexpr T ReverseBytes(const T& data) noexcept {
        auto bytes = std::bit_cast<std::array<std::byte, sizeof(T)>>(data);
        std::ranges::reverse(bytes);
        return std::bit_cast<T>(bytes);
    }

    inline constexpr bool is_little_endian = std::endian::native == std::endian::little;
    inline constexpr bool is_big_endian = std::endian::native == std::endian::big;

    [[nodiscard]] consteval bool IsLittleEndian() noexcept {
        return is_little_endian;
    }

    [[nodiscard]] consteval bool IsBigEndian() noexcept {
        return is_big_endian;
    }

    // Endianness conversion functions
    template <TriviallyCopyable T>
    [[nodiscard]] constexpr T LittleEndianToNative(T x) noexcept {
        if constexpr (is_little_endian)
        {
            return x;
        } else
        {
            return ReverseBytes(x);
        }
    }

    template <TriviallyCopyable T>
    [[nodiscard]] constexpr T BigEndianToNative(T x) noexcept {
        if constexpr (is_big_endian)
        {
            return x;
        } else
        {
            return ReverseBytes(x);
        }
    }

    template <TriviallyCopyable T>
    [[nodiscard]] constexpr T NativeToLittleEndian(T x) noexcept {
        if constexpr (is_little_endian)
        {
            return x;
        } else
        {
            return ReverseBytes(x);
        }
    }

    template <TriviallyCopyable T>
    [[nodiscard]] constexpr T NativeToBigEndian(T x) noexcept {
        if constexpr (is_big_endian)
        {
            return x;
        } else
        {
            return ReverseBytes(x);
        }
    }

    // Rest of the implementation remains the same...
    // Binary I/O functions
    template <TriviallyCopyable T>
    [[nodiscard]] T ReadBinaryLittleEndian(std::istream* stream) {
        T data;
        if (!stream->read(reinterpret_cast<char*>(&data), sizeof(T)))
        {
            throw std::runtime_error("Failed to read binary data");
        }
        return LittleEndianToNative(data);
    }

    template <TriviallyCopyable T>
    void ReadBinaryLittleEndian(std::istream* stream, std::vector<T>* data) {
        for (auto& element : *data)
        {
            element = ReadBinaryLittleEndian<T>(stream);
        }
    }

    template <TriviallyCopyable T>
    void ReadBinaryLittleEndian(std::istream* stream, std::vector<T>& data) {
        ReadBinaryLittleEndian(stream, &data);
    }

    template <TriviallyCopyable T>
    void WriteBinaryLittleEndian(std::ostream* stream, const T& data) {
        const T converted = NativeToLittleEndian(data);
        if (!stream->write(reinterpret_cast<const char*>(&converted), sizeof(T)))
        {
            throw std::runtime_error("Failed to write binary data");
        }
    }

    template <TriviallyCopyable T>
    void WriteBinaryLittleEndian(std::ostream* stream, const std::vector<T>* data) {
        for (const auto& element : *data)
        {
            WriteBinaryLittleEndian(stream, element);
        }
    }

    template <TriviallyCopyable T>
    void WriteBinaryLittleEndian(std::ostream* stream, const std::vector<T>& data) {
        WriteBinaryLittleEndian(stream, &data);
    }

    template <TriviallyCopyable T>
    void WriteBinaryLittleEndian(std::ostream* stream, std::span<const T> data) {
        for (const auto& element : data)
        {
            WriteBinaryLittleEndian(stream, element);
        }
    }

} // namespace glomap