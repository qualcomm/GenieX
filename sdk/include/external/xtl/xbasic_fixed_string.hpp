/***************************************************************************
* Copyright (c) Sylvain Corlay and Johan Mabille and Wolf Vollprecht       *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTL_BASIC_FIXED_STRING_HPP
#define XTL_BASIC_FIXED_STRING_HPP

#include <cstddef>
#include <exception>
#include <functional>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <cassert>
#include <algorithm>
#include <type_traits>

#ifdef __CLING__
#include <nlohmann/json.hpp>
#endif

#include "xhash.hpp"
#include "xtl_config.hpp"

namespace xtl
{

    namespace string_policy
    {
        template <std::size_t>
        struct silent_error;

        template <std::size_t>
        struct throwing_error;
    }

    /***********************
     * xbasic_fixed_string *
     ***********************/

    enum storage_options
    {
        buffer = 1 << 0,
        pointer = 1 << 1,
        store_size = 1 << 2,
        is_const = 1 << 3
    };

    template <class CT, std::size_t N = 55, int ST = buffer | store_size, template <std::size_t> class EP = string_policy::silent_error, class TR = std::char_traits<CT>>
    class xbasic_fixed_string;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    std::basic_ostream<CT, TR>& operator<<(std::basic_ostream<CT, TR>& os,
                                           const xbasic_fixed_string<CT, N, ST, EP, TR>& str);

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    std::basic_istream<CT, TR>& operator>>(std::basic_istream<CT, TR>& is,
                                           xbasic_fixed_string<CT, N, ST, EP, TR>& str);

    template <class CT>
    using xbasic_string_view = xbasic_fixed_string<const CT, 0, pointer | store_size | is_const>;

    namespace detail
    {
        template <int selector>
        struct select_storage;

        template <typename T>
        struct fixed_small_string_storage_impl;

        template <class T, std::size_t N>
        struct fixed_small_string_storage_impl<T[N]>
        {
            static_assert(N <= (1u << (8 * sizeof(T))), "small string");

            fixed_small_string_storage_impl()
            {
                set_size(0);
            }

            fixed_small_string_storage_impl(T ptr[N], std::size_t size)
                : m_buffer(ptr)
            {
                m_buffer[N - 1] = N - size;
            }

            T* buffer()
            {
                return m_buffer;
            }

            const T* buffer() const
            {
                return m_buffer;
            }

            std::size_t size() const
            {
                // Don't use std::make_unsinged_t here, this should remain C++11 compatible
                using unsigned_type = typename std::make_unsigned<T>::type;
                return N - reinterpret_cast<unsigned_type const*>(m_buffer)[N - 1];
            }

            void set_size(std::size_t sz)
            {
                assert(sz < N && "setting a small size");
                // Don't use std::make_unsinged_t here, this should remain C++11 compatible
                using unsigned_type = typename std::make_unsigned<T>::type;
                reinterpret_cast<unsigned_type*>(m_buffer)[N - 1] = static_cast<unsigned_type>(N - sz);
                m_buffer[sz] = '\0';
            }

            void adjust_size(std::ptrdiff_t val)
            {
                assert(size() + val >= 0 && "adjusting to positive size");
                set_size(static_cast<std::size_t>(static_cast<std::ptrdiff_t>(size()) + val));
            }

            T m_buffer[N];
        };

        template <class T>
        struct fixed_string_storage_impl
        {
            fixed_string_storage_impl() = default;

            fixed_string_storage_impl(T ptr, std::size_t size)
                : m_buffer(ptr), m_size(size)
            {
            }

            T& buffer()
            {
                return m_buffer;
            }

            const T& buffer() const
            {
                return m_buffer;
            }

            std::size_t size() const
            {
                return m_size;
            }

            void set_size(std::size_t sz)
            {
                m_size = sz;
                m_buffer[sz] = '\0';
            }

            void adjust_size(std::ptrdiff_t val)
            {
                m_size += std::size_t(val);
                m_buffer[m_size] = '\0';
            }

            T m_buffer;
            std::size_t m_size;
        };

        template <class T>
        struct fixed_string_external_storage_impl
        {
            fixed_string_external_storage_impl() = default;

            fixed_string_external_storage_impl(T ptr, std::ptrdiff_t/*size*/)
            {
                m_buffer = ptr;
            }

            T& buffer()
            {
                return m_buffer;
            }

            const T& buffer() const
            {
                return m_buffer;
            }

            void set_size(std::size_t sz)
            {
                m_buffer[sz] = '\0';
            }

            void adjust_size(std::ptrdiff_t val)
            {
                m_buffer[size() + val] = '\0';
            }

            std::size_t size() const
            {
                return std::strlen(m_buffer);
            }

            T m_buffer;
        };

        template <class T, bool Small>
        struct select_fixed_storage {
          using type = fixed_string_storage_impl<T>;
        };

        template <class T>
        struct select_fixed_storage<T, true> {
          using type = fixed_small_string_storage_impl<T>;
        };

        template <>
        struct select_storage<buffer | store_size>
        {
            template <class T, std::size_t N>
            using type = typename select_fixed_storage<T[N + 1], N < (1u << (8 * sizeof(T)))>::type;
        };

        template <>
        struct select_storage<buffer>
        {
            template <class T, std::size_t N>
            using type = fixed_string_external_storage_impl<T[N + 1]>;
        };
    }

    template <class CT,
              std::size_t N,
              int ST,
              template <std::size_t> class EP,
              class TR>
    class xbasic_fixed_string
    {
    public:

        using traits_type = TR;
        using value_type = CT;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        using storage_type = typename detail::select_storage<ST>::template type<CT, N>;

        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using iterator = pointer;
        using const_iterator = const_pointer;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;


        static const size_type npos;

        using self_type = xbasic_fixed_string;
        using initializer_type = std::initializer_list<value_type>;
        using string_type = std::basic_string<value_type, traits_type>;

        using error_policy = EP<N>;

        xbasic_fixed_string();

        explicit xbasic_fixed_string(size_type count, value_type ch);
        explicit xbasic_fixed_string(const self_type& other,
                            size_type pos,
                            size_type count = npos);
        explicit xbasic_fixed_string(const string_type& other);
        explicit xbasic_fixed_string(const string_type& other,
                                     size_type pos,
                                     size_type count = npos);
        xbasic_fixed_string(const_pointer s, size_type count);
        xbasic_fixed_string(const_pointer s);
        xbasic_fixed_string(initializer_type ilist);

        template <class InputIt>
        xbasic_fixed_string(InputIt first, InputIt last);

        operator string_type() const;

        ~xbasic_fixed_string() = default;

        xbasic_fixed_string(const self_type&) = default;
        xbasic_fixed_string(self_type&&) = default;

        self_type& operator=(const self_type&) = default;
        self_type& operator=(self_type&&) = default;
        self_type& operator=(const_pointer s);
        self_type& operator=(value_type ch);
        self_type& operator=(initializer_type ilist);
        self_type& operator=(const string_type& str);

        self_type& assign(size_type count, value_type ch);
        self_type& assign(const self_type& other,
                          size_type pos,
                          size_type count = npos);
        self_type& assign(const_pointer s, size_type count);
        self_type& assign(const_pointer s);
        self_type& assign(initializer_type ilist);
        template <class InputIt>
        self_type& assign(InputIt first, InputIt last);
        self_type& assign(const self_type& rhs);
        self_type& assign(self_type&& rhs);
        self_type& assign(const string_type& str);
        self_type& assign(const string_type& other,
                          size_type pos,
                          size_type count = npos);

        reference at(size_type pos);
        const_reference at(size_type pos) const;

        reference operator[](size_type pos);
        const_reference operator[](size_type pos) const;

        reference front();
        const_reference front() const;

        reference back();
        const_reference back() const;

        pointer data() noexcept;
        const_pointer data() const noexcept;

        const_pointer c_str() const noexcept;

        iterator begin() noexcept;
        iterator end() noexcept;
        const_iterator begin() const noexcept;
        const_iterator end() const noexcept;
        const_iterator cbegin() const noexcept;
        const_iterator cend() const noexcept;

        reverse_iterator rbegin() noexcept;
        reverse_iterator rend() noexcept;
        const_reverse_iterator rbegin() const noexcept;
        const_reverse_iterator rend() const noexcept;
        const_reverse_iterator crbegin() const noexcept;
        const_reverse_iterator crend() const noexcept;

        bool empty() const noexcept;
        size_type size() const noexcept;
        size_type length() const noexcept;
        size_type max_size() const noexcept;

        void clear() noexcept;
        void push_back(value_type ch);
        void pop_back();
        self_type substr(size_type pos = 0, size_type count = npos) const;
        size_type copy(pointer dest, size_type count, size_type pos = 0) const;
        void resize(size_type count);
        void resize(size_type count, value_type ch);
        void swap(self_type& rhs) noexcept;

        self_type& insert(size_type index, size_type count, value_type ch);
        self_type& insert(size_type index, const_pointer s);
        self_type& insert(size_type index, const_pointer s, size_type count);
        self_type& insert(size_type index, const self_type& str);
        self_type& insert(size_type index, const self_type& str,
                          size_type index_str, size_type count = npos);
        self_type& insert(size_type index, const string_type& str);
        self_type& insert(size_type index, const string_type& str,
                          size_type index_str, size_type count = npos);
        iterator insert(const_iterator pos, value_type ch);
        iterator insert(const_iterator pos, size_type count, value_type ch);
        iterator insert(const_iterator pos, initializer_type ilist);
        template <class InputIt>
        iterator insert(const_iterator pos, InputIt first, InputIt last);

        self_type& erase(size_type index = 0, size_type count = npos);
        iterator erase(const_iterator position);
        iterator erase(const_iterator first, const_iterator last);

        self_type& append(size_type count, value_type ch);
        self_type& append(const self_type& str);
        self_type& append(const self_type& str,
                          size_type pos, size_type count = npos);
        self_type& append(const string_type& str);
        self_type& append(const string_type& str,
                          size_type pos, size_type count = npos);
        self_type& append(const_pointer s, size_type count);
        self_type& append(const_pointer s);
        self_type& append(initializer_type ilist);
        template <class InputIt>
        self_type& append(InputIt first, InputIt last);

        self_type& operator+=(const self_type& str);
        self_type& operator+=(const string_type& str);
        self_type& operator+=(value_type ch);
        self_type& operator+=(const_pointer s);
        self_type& operator+=(initializer_type ilist);

        int compare(const self_type& str) const noexcept;
        int compare(size_type pos1, size_type count1, const self_type& str) const;
        int compare(size_type pos1, size_type count1, const self_type& str,
                    size_type pos2, size_type count2 = npos) const;
        int compare(const string_type& str) const noexcept;
        int compare(size_type pos1, size_type count1, const string_type& str) const;
        int compare(size_type pos1, size_type count1, const string_type& str,
                    size_type pos2, size_type count2 = npos) const;
        int compare(const_pointer s) const noexcept;
        int compare(size_type pos1, size_type count1, const_pointer s) const;
        int compare(size_type pos1, size_type count1, const_pointer s, size_type count2) const;

        self_type& replace(size_type pos, size_type count, const self_type& str);
        self_type& replace(const_iterator first, const_iterator last, const self_type& str);
        self_type& replace(size_type pos1, size_type count1, const self_type& str,
                           size_type pos2, size_type count2 = npos);
        self_type& replace(size_type pos, size_type count, const string_type& str);
        self_type& replace(const_iterator first, const_iterator last, const string_type& str);
        self_type& replace(size_type pos1, size_type count1, const string_type& str,
                           size_type pos2, size_type count2 = npos);
        self_type& replace(size_type pos, size_type count, const_pointer cstr, size_type count2);
        self_type& replace(const_iterator first, const_iterator last, const_pointer cstr, size_type count2);
        self_type& replace(size_type pos, size_type count, const_pointer cstr);
        self_type& replace(const_iterator first, const_iterator last, const_pointer cstr);
        self_type& replace(size_type pos, size_type count, size_type count2, value_type ch);
        self_type& replace(const_iterator first, const_iterator last, size_type count2, value_type ch);
        self_type& replace(const_iterator first, const_iterator last, initializer_type ilist);
        template <class InputIt>
        self_type& replace(const_iterator first, const_iterator last, InputIt first2, InputIt last2);

        size_type find(const self_type& str, size_type pos = 0) const noexcept;
        size_type find(const string_type& str, size_type pos = 0) const noexcept;
        size_type find(const_pointer s, size_type pos, size_type count) const;
        size_type find(const_pointer s, size_type pos = 0) const;
        size_type find(value_type ch, size_type pos = 0) const;

        size_type rfind(const self_type& str, size_type pos = npos) const noexcept;
        size_type rfind(const string_type& str, size_type pos = npos) const noexcept;
        size_type rfind(const_pointer s, size_type pos, size_type count) const;
        size_type rfind(const_pointer s, size_type pos = npos) const;
        size_type rfind(value_type ch, size_type pos = npos) const;

        size_type find_first_of(const self_type& str, size_type pos = 0) const noexcept;
        size_type find_first_of(const string_type& str, size_type pos = 0) const noexcept;
        size_type find_first_of(const_pointer s, size_type pos, size_type count) const;
        size_type find_first_of(const_pointer s, size_type pos = 0) const;
        size_type find_first_of(value_type ch, size_type pos = 0) const;

        size_type find_first_not_of(const self_type& str, size_type pos = 0) const noexcept;
        size_type find_first_not_of(const string_type& str, size_type pos = 0) const noexcept;
        size_type find_first_not_of(const_pointer s, size_type pos, size_type count) const;
        size_type find_first_not_of(const_pointer s, size_type pos = 0) const;
        size_type find_first_not_of(value_type ch, size_type pos = 0) const;

        size_type find_last_of(const self_type& str, size_type pos = 0) const noexcept;
        size_type find_last_of(const string_type& str, size_type pos = 0) const noexcept;
        size_type find_last_of(const_pointer s, size_type pos, size_type count) const;
        size_type find_last_of(const_pointer s, size_type pos = 0) const;
        size_type find_last_of(value_type ch, size_type pos = 0) const;

        size_type find_last_not_of(const self_type& str, size_type pos = npos) const noexcept;
        size_type find_last_not_of(const string_type& str, size_type pos = npos) const noexcept;
        size_type find_last_not_of(const_pointer s, size_type pos, size_type count) const;
        size_type find_last_not_of(const_pointer s, size_type pos = npos) const;
        size_type find_last_not_of(value_type ch, size_type pos = npos) const;

    private:

        int compare_impl(const_pointer s1, size_type count1, const_pointer s2, size_type count2) const noexcept;
        void update_null_termination() noexcept;
        void check_index(size_type pos, size_type size, const char* what) const;
        void check_index_strict(size_type pos, size_type size, const char* what) const;

        storage_type m_storage;
    };

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    const typename xbasic_fixed_string<CT, N, ST, EP, TR>::size_type xbasic_fixed_string<CT, N, ST, EP, TR>::npos
        = std::basic_string<value_type, traits_type>::npos;

    template <std::size_t N>
    using xfixed_string = xbasic_fixed_string<char, N>;

    template <std::size_t N>
    using xwfixed_string = xbasic_fixed_string<wchar_t, N>;

    template <std::size_t N>
    using xu16fixed_string = xbasic_fixed_string<char16_t, N>;

    template <std::size_t N>
    using xu32fixed_string = xbasic_fixed_string<char32_t, N>;

    /**************************
     * Concatenation operator *
     **************************/

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    xbasic_fixed_string<CT, N, ST, EP, TR>
    operator+(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
              const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs);

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    xbasic_fixed_string<CT, N, ST, EP, TR>
    operator+(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
              const CT* rhs);

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    xbasic_fixed_string<CT, N, ST, EP, TR>
    operator+(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
              CT rhs);

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    xbasic_fixed_string<CT, N, ST, EP, TR>
    operator+(const CT* lhs,
              const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs);

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    xbasic_fixed_string<CT, N, ST, EP, TR>
    operator+(CT lhs,
              const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs);

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    xbasic_fixed_string<CT, N, ST, EP, TR>
    operator+(const xbasic_fixed_string<CT, N, ST, EP, TR>&& lhs,
              const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs);

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    xbasic_fixed_string<CT, N, ST, EP, TR>
    operator+(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
              const xbasic_fixed_string<CT, N, ST, EP, TR>&& rhs);

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    xbasic_fixed_string<CT, N, ST, EP, TR>
    operator+(const xbasic_fixed_string<CT, N, ST, EP, TR>&& lhs,
              const xbasic_fixed_string<CT, N, ST, EP, TR>&& rhs);

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    xbasic_fixed_string<CT, N, ST, EP, TR>
    operator+(const xbasic_fixed_string<CT, N, ST, EP, TR>&& lhs,
              const CT* rhs);

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    xbasic_fixed_string<CT, N, ST, EP, TR>
    operator+(const xbasic_fixed_string<CT, N, ST, EP, TR>&& lhs,
              CT rhs);

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    xbasic_fixed_string<CT, N, ST, EP, TR>
    operator+(const CT* lhs,
              const xbasic_fixed_string<CT, N, ST, EP, TR>&& rhs);

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    xbasic_fixed_string<CT, N, ST, EP, TR>
    operator+(CT lhs,
              const xbasic_fixed_string<CT, N, ST, EP, TR>&& rhs);

    /************************
     * Comparison operators *
     ************************/

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator==(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                    const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator==(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                    const CT* rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator==(const CT* lhs,
                    const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator==(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                    const std::basic_string<CT, TR>& rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator==(const std::basic_string<CT, TR>& lhs,
                    const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator!=(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                    const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator!=(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                    const CT* rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator!=(const CT* lhs,
                    const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator!=(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                    const std::basic_string<CT, TR>& rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator!=(const std::basic_string<CT, TR>& lhs,
                    const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator<(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                   const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator<(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                   const CT* rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator<(const CT* lhs,
                   const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator<(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                   const std::basic_string<CT, TR>& rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator<(const std::basic_string<CT, TR>& lhs,
                   const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator<=(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                    const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator<=(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                    const CT* rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator<=(const CT* lhs,
                    const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator<=(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                    const std::basic_string<CT, TR>& rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator<=(const std::basic_string<CT, TR>& lhs,
                    const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator>(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                   const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator>(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                   const CT* rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator>(const CT* lhs,
                   const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator>(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                   const std::basic_string<CT, TR>& rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator>(const std::basic_string<CT, TR>& lhs,
                   const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator>=(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                    const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator>=(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                    const CT* rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator>=(const CT* lhs,
                    const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator>=(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                    const std::basic_string<CT, TR>& rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    bool operator>=(const std::basic_string<CT, TR>& lhs,
                    const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept;

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    void swap(xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
              xbasic_fixed_string<CT, N, ST, EP, TR>& rhs);

    /******************************
     * Input / output declaration *
     ******************************/

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    std::basic_istream<CT, TR>& getline(std::basic_istream<CT, TR>& input,
                                        xbasic_fixed_string<CT, N, ST, EP, TR>& str,
                                        CT delim);

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    std::basic_istream<CT, TR>& getline(std::basic_istream<CT, TR>&& input,
                                        xbasic_fixed_string<CT, N, ST, EP, TR>& str,
                                        CT delim);

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    std::basic_istream<CT, TR>& getline(std::basic_istream<CT, TR>& input,
                                        xbasic_fixed_string<CT, N, ST, EP, TR>& str);

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    std::basic_istream<CT, TR>& getline(std::basic_istream<CT, TR>&& input,
                                        xbasic_fixed_string<CT, N, ST, EP, TR>& str);

}  // namespace xtl

namespace std
{
    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    struct hash<::xtl::xbasic_fixed_string<CT, N, ST, EP, TR>>
    {
        using argument_type = ::xtl::xbasic_fixed_string<CT, N, ST, EP, TR>;
        using result_type = std::size_t;
        inline result_type operator()(const argument_type& arg) const
        {
            return ::xtl::hash_bytes(arg.data(), arg.size(), static_cast<std::size_t>(0xc70f6907UL));
        }
    };
}  // namespace std

namespace xtl
{
    /********************************
     * xbasic_fixed_string policies *
     ********************************/

    namespace string_policy
    {
        template <std::size_t N>
        struct silent_error
        {
            inline static std::size_t check_size(std::size_t size)
            {
                return size;
            }
            inline static std::size_t check_add(std::size_t size1, std::size_t size2)
            {
                return size1 + size2;
            }
        };

        template <std::size_t N>
        struct throwing_error
        {
            inline static std::size_t check_size(std::size_t size)
            {
                if (size > N)
                {
                    std::ostringstream oss;
                    oss << "Invalid size (" << size << ") for xbasic_fixed_string - maximal size: " << N;
#if defined(XTL_NO_EXCEPTIONS)
                    std::fprintf(stderr, "%s\n", oss.str().c_str());
                    std::terminate();
#else
                    throw std::length_error(oss.str());
#endif
                }
                return size;
            }

            inline static std::size_t check_add(std::size_t size1, std::size_t size2)
            {
                return check_size(size1 + size2);
            }
        };
    }  // string_policy

    /**************************************
     * xbasic_fixed_string implementation *
     **************************************/

    /****************
     * Constructors *
     ****************/

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline xbasic_fixed_string<CT, N, ST, EP, TR>::xbasic_fixed_string()
        : m_storage()
    {
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline xbasic_fixed_string<CT, N, ST, EP, TR>::xbasic_fixed_string(size_type count, value_type ch)
        : m_storage()
    {
        assign(count, ch);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline xbasic_fixed_string<CT, N, ST, EP, TR>::xbasic_fixed_string(const self_type& other,
                                                                   size_type pos,
                                                                   size_type count)
        : m_storage()
    {
        assign(other, pos, count);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline xbasic_fixed_string<CT, N, ST, EP, TR>::xbasic_fixed_string(const string_type& other)
        : m_storage()
    {
        assign(other);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline xbasic_fixed_string<CT, N, ST, EP, TR>::xbasic_fixed_string(const string_type& other,
                                                                   size_type pos,
                                                                   size_type count)
        : m_storage()
    {
        assign(other, pos, count);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline xbasic_fixed_string<CT, N, ST, EP, TR>::xbasic_fixed_string(const_pointer s, size_type count)
        : m_storage()
    {
        assign(s, count);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline xbasic_fixed_string<CT, N, ST, EP, TR>::xbasic_fixed_string(const_pointer s)
        : m_storage()
    {
        assign(s);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline xbasic_fixed_string<CT, N, ST, EP, TR>::xbasic_fixed_string(initializer_type ilist)
        : m_storage()
    {
        assign(ilist);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    template <class InputIt>
    inline xbasic_fixed_string<CT, N, ST, EP, TR>::xbasic_fixed_string(InputIt first, InputIt last)
        : m_storage()
    {
        assign(first, last);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline xbasic_fixed_string<CT, N, ST, EP, TR>::operator string_type() const
    {
        return string_type(data());
    }

    /**************
     * Assignment *
     **************/

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::operator=(const_pointer s) -> self_type&
    {
        return assign(s);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::operator=(value_type ch) -> self_type&
    {
        return assign(size_type(1), ch);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::operator=(initializer_type ilist) -> self_type&
    {
        return assign(ilist);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::operator=(const string_type& str) -> self_type&
    {
        return assign(str);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::assign(size_type count, value_type ch) -> self_type&
    {
        m_storage.set_size(error_policy::check_size(count));
        traits_type::assign(data(), count, ch);
        return *this;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::assign(const self_type& other,
                                                           size_type pos,
                                                           size_type count) -> self_type&
    {
        check_index_strict(pos, other.size(), "xbasic_fixed_string::assign");
        size_type copy_count = std::min(other.size() - pos, count);
        m_storage.set_size(error_policy::check_size(copy_count));
        traits_type::copy(data(), other.data() + pos, copy_count);
        return *this;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::assign(const_pointer s, size_type count) -> self_type&
    {
        m_storage.set_size(error_policy::check_size(count));
        traits_type::copy(data(), s, count);
        return *this;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::assign(const_pointer s) -> self_type&
    {
        std::size_t ssize = traits_type::length(s);
        return assign(s, ssize);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::assign(initializer_type ilist) -> self_type&
    {
        return assign(ilist.begin(), ilist.end());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    template <class InputIt>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::assign(InputIt first, InputIt last) -> self_type&
    {
        m_storage.set_size(error_policy::check_size(static_cast<size_type>(std::distance(first, last))));
        std::copy(first, last, data());
        return *this;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::assign(const self_type& rhs) -> self_type&
    {
        if (this != &rhs)
        {
            m_storage.set_size(rhs.size());
            traits_type::copy(data(), rhs.data(), rhs.size());
        }
        return *this;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::assign(self_type&& rhs) -> self_type&
    {
        if (this != &rhs)
        {
            m_storage.set_size(rhs.size());
            traits_type::copy(data(), rhs.data(), rhs.size());
        }
        return *this;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::assign(const string_type& other) -> self_type&
    {
        return assign(other.c_str());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::assign(const string_type& other,
                                                           size_type pos,
                                                           size_type count) -> self_type&
    {
        return assign(other.c_str() + pos, std::min(count, other.size() - pos));
    }

    /******************
     * Element access *
     ******************/

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::at(size_type pos) -> reference
    {
        check_index(pos, size(), "basic_fixed_string::at");
        return this->operator[](pos);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::at(size_type pos) const -> const_reference
    {
        check_index(pos, size(), "basic_fixed_string::at");
        return this->operator[](pos);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::operator[](size_type pos) -> reference
    {
        return data()[pos];
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::operator[](size_type pos) const -> const_reference
    {
        return data()[pos];
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::front() -> reference
    {
        return this->operator[](0);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::front() const -> const_reference
    {
        return this->operator[](0);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::back() -> reference
    {
        return this->operator[](size() - 1);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::back() const -> const_reference
    {
        return this->operator[](size() - 1);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::data() noexcept -> pointer
    {
        return m_storage.buffer();
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::data() const noexcept -> const_pointer
    {
        return m_storage.buffer();
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::c_str() const noexcept -> const_pointer
    {
        return m_storage.buffer();
    }

    /*************
     * Iterators *
     *************/

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::begin() noexcept -> iterator
    {
        return data();
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::end() noexcept -> iterator
    {
        return data() + size();
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::begin() const noexcept -> const_iterator
    {
        return cbegin();
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::end() const noexcept -> const_iterator
    {
        return cend();
    }
    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::cbegin() const noexcept -> const_iterator
    {
        return data();
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::cend() const noexcept -> const_iterator
    {
        return data() + size();
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::rbegin() noexcept -> reverse_iterator
    {
        return reverse_iterator(end());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::rend() noexcept -> reverse_iterator
    {
        return reverse_iterator(begin());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::rbegin() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator(end());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::rend() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator(begin());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::crbegin() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator(end());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::crend() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator(begin());
    }

    /************
     * Capacity *
     ************/

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool xbasic_fixed_string<CT, N, ST, EP, TR>::empty() const noexcept
    {
        return size() == 0;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::size() const noexcept -> size_type
    {
        return m_storage.size();
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::length() const noexcept -> size_type
    {
        return m_storage.size();
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::max_size() const noexcept -> size_type
    {
        return N;
    }

    /**************
     * Operations *
     **************/

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline void xbasic_fixed_string<CT, N, ST, EP, TR>::clear() noexcept
    {
        m_storage.set_size(0);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline void xbasic_fixed_string<CT, N, ST, EP, TR>::push_back(value_type ch)
    {
        error_policy::check_add(size(), size_type(1));
        data()[size()] = ch;
        m_storage.adjust_size(+1);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline void xbasic_fixed_string<CT, N, ST, EP, TR>::pop_back()
    {
        m_storage.adjust_size(-1);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::substr(size_type pos, size_type count) const -> self_type
    {
        return self_type(*this, pos, count);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::copy(pointer dest, size_type count, size_type pos) const -> size_type
    {
        check_index_strict(pos, size(), "xbasic_fixed_string::copy");
        size_type nb_copied = std::min(count, size() - pos);
        traits_type::copy(dest, data() + pos, nb_copied);
        return nb_copied;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline void xbasic_fixed_string<CT, N, ST, EP, TR>::resize(size_type count)
    {
        resize(count, value_type(' '));  // need to initialize with some value != \0
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline void xbasic_fixed_string<CT, N, ST, EP, TR>::resize(size_type count, value_type ch)
    {
        size_type old_size = size();
        m_storage.set_size(error_policy::check_size(count));
        if (old_size < size())
        {
            traits_type::assign(data() + old_size, size() - old_size, ch);
        }
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline void xbasic_fixed_string<CT, N, ST, EP, TR>::swap(self_type& rhs) noexcept
    {
        self_type tmp(std::move(rhs));
        rhs = std::move(*this);
        *this = std::move(tmp);
    }

    /**********
     * insert *
     **********/

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    auto xbasic_fixed_string<CT, N, ST, EP, TR>::insert(size_type index, size_type count, value_type ch) -> self_type&
    {
        check_index_strict(index, size(), "xbasic_fixed_string::insert");
        size_type old_size = size();
        m_storage.set_size(error_policy::check_add(size(), count));
        std::copy_backward(data() + index, data() + old_size, end());
        traits_type::assign(data() + index, count, ch);
        return *this;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::insert(size_type index, const_pointer s) -> self_type&
    {
        return insert(index, s, traits_type::length(s));
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    auto xbasic_fixed_string<CT, N, ST, EP, TR>::insert(size_type index, const_pointer s, size_type count) -> self_type&
    {
        check_index_strict(index, size(), "xbasic_fixed_string::insert");
        size_type old_size = size();
        m_storage.set_size(error_policy::check_add(size(), count));
        std::copy_backward(data() + index, data() + old_size, end());
        traits_type::copy(data() + index, s, count);
        return *this;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::insert(size_type index, const self_type& str) -> self_type&
    {
        return insert(index, str.data(), str.size());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::insert(size_type index, const self_type& str,
                                                           size_type index_str, size_type count) -> self_type&
    {
        check_index_strict(index_str, str.size(), "xbasic_fixed_string::insert");
        return insert(index, str.data() + index_str, std::min(count, str.size() - index_str));
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::insert(size_type index, const string_type& str) -> self_type&
    {
        return insert(index, str.c_str(), str.size());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::insert(size_type index, const string_type& str,
                                                           size_type index_str, size_type count) -> self_type&
    {
        check_index_strict(index_str, str.size(), "xbasic_fixed_string::insert");
        return insert(index, str.c_str() + index_str, std::min(count, str.size() - index_str));
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::insert(const_iterator pos, value_type ch) -> iterator
    {
        return insert(pos, size_type(1), ch);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::insert(const_iterator pos, size_type count, value_type ch) -> iterator
    {
        if (cbegin() <= pos && pos < cend())
        {
            size_type index = static_cast<size_type>(pos - cbegin());
            insert(index, count, ch);
            return const_cast<iterator>(pos);
        }
        return end();
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::insert(const_iterator pos, initializer_type ilist) -> iterator
    {
        return insert(pos, ilist.begin(), ilist.end());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    template <class InputIt>
    auto xbasic_fixed_string<CT, N, ST, EP, TR>::insert(const_iterator pos, InputIt first, InputIt last) -> iterator
    {
        if (cbegin() <= pos && pos < cend())
        {
            size_type index = static_cast<size_type>(pos - cbegin());
            size_type count = static_cast<size_type>(std::distance(first, last));
            size_type old_size = size();
            m_storage.set_size(error_policy::check_add(size(), count));
            std::copy_backward(data() + index, data() + old_size, end());
            std::copy(first, last, data() + index);
            return begin() + index;
        }
        return end();
    }

    /*********
     * erase *
     *********/

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    auto xbasic_fixed_string<CT, N, ST, EP, TR>::erase(size_type index, size_type count) -> self_type&
    {
        check_index_strict(index, size(), "xbasic_fixed_string::erase");
        size_type erase_count = std::min(count, size() - index);
        // cannot use traits_type::copy because of overlapping
        std::copy(data() + index + erase_count, data() + size(), data() + index);
        m_storage.adjust_size(-static_cast<std::ptrdiff_t>(erase_count));
        return *this;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::erase(const_iterator position) -> iterator
    {
        return erase(position, position + 1);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    auto xbasic_fixed_string<CT, N, ST, EP, TR>::erase(const_iterator first, const_iterator last) -> iterator
    {
        if (cbegin() <= first && first < cend())
        {
            const_iterator adapted_last = std::min(last, cend());
            size_type erase_count = static_cast<size_type>(adapted_last - first);
            // cannot use traits_type::copy because of overlapping
            std::copy(adapted_last, cend(), iterator(first));
            m_storage.adjust_size(-static_cast<std::ptrdiff_t>(erase_count));
            return const_cast<iterator>(first);
        }
        return end();
    }

    /**********
     * append *
     **********/

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::append(size_type count, value_type ch) -> self_type&
    {
        size_type old_size = m_storage.size();
        m_storage.set_size(error_policy::check_add(size(), count));
        traits_type::assign(data() + old_size, count, ch);
        return *this;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::append(const self_type& str) -> self_type&
    {
        return append(str.data(), str.size());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::append(const self_type& str,
                                                           size_type pos, size_type count) -> self_type&
    {
        check_index_strict(pos, str.size(), "xbasic_fixed_string::append");
        return append(str.data() + pos, std::min(count, str.size() - pos));
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::append(const string_type& str) -> self_type&
    {
        return append(str.c_str(), str.size());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::append(const string_type& str,
                                                           size_type pos, size_type count) -> self_type&
    {
        check_index_strict(pos, str.size(), "xbasic_fixed_string::append");
        return append(str.c_str() + pos, std::min(count, str.size() - pos));
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::append(const_pointer s, size_type count) -> self_type&
    {
        size_type old_size = m_storage.size();
        m_storage.set_size(error_policy::check_add(size(), count));
        traits_type::copy(data() + old_size, s, count);
        return *this;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::append(const_pointer s) -> self_type&
    {
        return append(s, traits_type::length(s));
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::append(initializer_type ilist) -> self_type&
    {
        return append(ilist.begin(), ilist.end());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    template <class InputIt>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::append(InputIt first, InputIt last) -> self_type&
    {
        size_type count = static_cast<size_type>(std::distance(first, last));
        size_type old_size = m_storage.size();
        m_storage.set_size(error_policy::check_add(size(), count));
        std::copy(first, last, data() + old_size);
        return *this;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::operator+=(const self_type& str) -> self_type&
    {
        return append(str);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::operator+=(const string_type& str) -> self_type&
    {
        return append(str);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::operator+=(value_type ch) -> self_type&
    {
        return append(size_type(1), ch);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::operator+=(const_pointer s) -> self_type&
    {
        return append(s);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::operator+=(initializer_type ilist) -> self_type&
    {
        return append(ilist);
    }

    /***********
     * compare *
     ***********/

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline int xbasic_fixed_string<CT, N, ST, EP, TR>::compare(const self_type& str) const noexcept
    {
        return compare_impl(data(), size(), str.data(), str.size());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline int xbasic_fixed_string<CT, N, ST, EP, TR>::compare(size_type pos1, size_type count1, const self_type& str) const
    {
        check_index_strict(pos1, size(), "xbasic_fixed_string::compare");
        return compare_impl(data() + pos1, std::min(count1, size() - pos1), str.data(), str.size());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline int xbasic_fixed_string<CT, N, ST, EP, TR>::compare(size_type pos1, size_type count1, const self_type& str,
                                                           size_type pos2, size_type count2) const
    {
        check_index_strict(pos1, size(), "xbasic_fixed_string::compare");
        check_index_strict(pos2, str.size(), "xbasic_fixed_string::compare");
        return compare_impl(data() + pos1, std::min(count1, size() - pos1),
                            str.data() + pos2, std::min(count2, str.size() - pos2));
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline int xbasic_fixed_string<CT, N, ST, EP, TR>::compare(const string_type& str) const noexcept
    {
        return compare_impl(data(), size(), str.data(), str.size());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline int xbasic_fixed_string<CT, N, ST, EP, TR>::compare(size_type pos1, size_type count1, const string_type& str) const
    {
        check_index_strict(pos1, size(), "xbasic_fixed_string::compare");
        return compare_impl(data() + pos1, std::min(count1, size() - pos1), str.data(), str.size());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline int xbasic_fixed_string<CT, N, ST, EP, TR>::compare(size_type pos1, size_type count1, const string_type& str,
                                                           size_type pos2, size_type count2) const
    {
        check_index_strict(pos1, size(), "xbasic_fixed_string::compare");
        check_index_strict(pos2, str.size(), "xbasic_fixed_string::compare");
        return compare_impl(data() + pos1, std::min(count1, size() - pos1),
                            str.data() + pos2, std::min(count2, str.size() - pos2));
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline int xbasic_fixed_string<CT, N, ST, EP, TR>::compare(const_pointer s) const noexcept
    {
        return compare_impl(data(), size(), s, traits_type::length(s));
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    int xbasic_fixed_string<CT, N, ST, EP, TR>::compare(size_type pos1, size_type count1, const_pointer s) const
    {
        return compare(pos1, count1, s, traits_type::length(s));
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    int xbasic_fixed_string<CT, N, ST, EP, TR>::compare(size_type pos1, size_type count1, const_pointer s, size_type count2) const
    {
        check_index_strict(pos1, size(), "xbasic_fixed_string::compare");
        return compare_impl(data() + pos1, std::min(count1, size() - pos1),
                            s, count2);
    }

    /***********
     * replace *
     ***********/

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::replace(size_type pos, size_type count, const self_type& str) -> self_type&
    {
        return replace(pos, count, str.data(), str.size());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::replace(const_iterator first, const_iterator last,
                                                            const self_type& str) -> self_type&
    {
        return replace(first, last, str.data(), str.size());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::replace(size_type pos1, size_type count1, const self_type& str,
                                                            size_type pos2, size_type count2) -> self_type&
    {
        check_index_strict(pos2, str.size(), "xbasic_fixed_string::replace");
        return replace(pos1, count1, str.data() + pos2, std::min(count2, str.size() - pos2));
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::replace(size_type pos, size_type count, const string_type& str) -> self_type&
    {
        return replace(pos, count, str.data(), str.size());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::replace(const_iterator first, const_iterator last,
                                                            const string_type& str) -> self_type&
    {
        return replace(first, last, str.data(), str.size());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::replace(size_type pos1, size_type count1, const string_type& str,
                                                            size_type pos2, size_type count2) -> self_type&
    {
        check_index_strict(pos2, str.size(), "xbasic_fixed_string::replace");
        return replace(pos1, count1, str.data() + pos2, std::min(count2, str.size() - pos2));
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    auto xbasic_fixed_string<CT, N, ST, EP, TR>::replace(size_type pos, size_type count,
                                                     const_pointer cstr, size_type count2) -> self_type&
    {
        check_index_strict(pos, size(), "xbasic_fixed_string::replace");
        size_type erase_count = std::min(count, size() - pos);
        size_type new_size = error_policy::check_add(size() - erase_count, count2);
        if (erase_count > count2)
        {
            traits_type::copy(data() + pos, cstr, count2);
            std::copy(cbegin() + pos + erase_count, cend(), data() + pos + count2);
            m_storage.set_size(new_size);
        }
        else if (erase_count < count2)
        {
            std::copy_backward(cbegin() + pos + erase_count, cend(), data() + new_size);
            traits_type::copy(data() + pos, cstr, count2);
            m_storage.set_size(new_size);
        }
        else
        {
            traits_type::copy(data() + pos, cstr, count2);
        }
        return *this;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::replace(const_iterator first, const_iterator last,
                                                            const_pointer cstr, size_type count2) -> self_type&
    {
        if (cbegin() <= first && first < last && last <= cend())
        {
            size_type pos = static_cast<size_type>(first - cbegin());
            size_type count = static_cast<size_type>(last - first);
            return replace(pos, count, cstr, count2);
        }
        return *this;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::replace(size_type pos, size_type count,
                                                            const_pointer cstr) -> self_type&
    {
        return replace(pos, count, cstr, traits_type::length(cstr));
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::replace(const_iterator first, const_iterator last,
                                                            const_pointer cstr) -> self_type&
    {
        return replace(first, last, cstr, traits_type::length(cstr));
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::replace(size_type pos, size_type count,
                                                            size_type count2, value_type ch) -> self_type&
    {
        check_index_strict(pos, size(), "xbasic_fixed_string::replace");
        size_type erase_count = std::min(count, size() - pos);
        size_type new_size = error_policy::check_add(size() - erase_count, count2);
        if (erase_count > count2)
        {
            traits_type::assign(data() + pos, count2, ch);
            std::copy(cbegin() + pos + erase_count, cend(), data() + pos + count2);
            m_storage.set_size(new_size);
        }
        else if (erase_count < count2)
        {
            std::copy_backward(cbegin() + pos + erase_count, cend(), data() + new_size);
            traits_type::assign(data() + pos, count2, ch);
            m_storage.set_size(new_size);
        }
        else
        {
            traits_type::assign(data() + pos, count2, ch);
        }
        return *this;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::replace(const_iterator first, const_iterator last,
                                                            size_type count2, value_type ch) -> self_type&
    {
        if (cbegin() <= first && first < last && last <= cend())
        {
            size_type pos = static_cast<size_type>(first - cbegin());
            size_type count = static_cast<size_type>(last - first);
            return replace(pos, count, count2, ch);
        }
        return *this;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::replace(const_iterator first, const_iterator last,
                                                            initializer_type ilist) -> self_type&
    {
        return replace(first, last, ilist.begin(), ilist.end());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    template <class InputIt>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::replace(const_iterator first, const_iterator last,
                                                            InputIt first2, InputIt last2) -> self_type&
    {
        if (cbegin() <= first && first < last && last <= cend())
        {
            size_type pos = static_cast<size_type>(first - cbegin());
            size_type erase_count = static_cast<size_type>(last - first);
            size_type count2 = static_cast<size_type>(std::distance(first2, last2));
            size_type new_size = error_policy::check_add(size() - erase_count, count2);
            if (erase_count > count2)
            {
                std::copy(first2, last2, data() + pos);
                std::copy(cbegin() + pos + erase_count, cend(), data() + pos + count2);
                m_storage.set_size(new_size);
            }
            else if (erase_count < count2)
            {
                std::copy_backward(cbegin() + pos + erase_count, cend(), data() + new_size);
                std::copy(first2, last2, data() + pos);
                m_storage.set_size(new_size);
            }
            else
            {
                std::copy(first2, last2, data() + pos);
            }
        }
        return *this;
    }

    /********
     * find *
     ********/

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::find(const self_type& str, size_type pos) const noexcept -> size_type
    {
        return find(str.data(), pos, str.size());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::find(const string_type& str, size_type pos) const noexcept -> size_type
    {
        return find(str.data(), pos, str.size());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    auto xbasic_fixed_string<CT, N, ST, EP, TR>::find(const_pointer s, size_type pos, size_type count) const -> size_type
    {
        if (count == size_type(0) && pos <= size())
        {
            return pos;
        }

        size_type nm;
        if (pos < size() && count <= (nm = size() - pos))
        {
            const_pointer uptr, vptr;
            for (nm -= count - 1, vptr = data() + pos;
                 (uptr = traits_type::find(vptr, nm, *s)) != 0;
                 nm -= size_type(uptr - vptr) + 1ul, vptr = uptr + 1ul)
            {
                if (traits_type::compare(uptr, s, count) == 0)
                {
                    return size_type(uptr - data());
                }
            }
        }
        return npos;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::find(const_pointer s, size_type pos) const -> size_type
    {
        return find(s, pos, traits_type::length(s));
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::find(value_type ch, size_type pos) const -> size_type
    {
        return find((const_pointer)(&ch), pos, size_type(1));
    }

    /*********
     * rfind *
     *********/

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::rfind(const self_type& str, size_type pos) const noexcept -> size_type
    {
        return rfind(str.data(), pos, str.size());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::rfind(const string_type& str, size_type pos) const noexcept -> size_type
    {
        return rfind(str.data(), pos, str.size());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    auto xbasic_fixed_string<CT, N, ST, EP, TR>::rfind(const_pointer s, size_type pos, size_type count) const -> size_type
    {
        if (count == 0)
        {
            return std::min(pos, size());
        }

        if (count <= size())
        {
            const_pointer uptr = data() + std::min(pos, size() - count);
            for (;; --uptr)
            {
                if (traits_type::eq(*uptr, *s) && traits_type::compare(uptr, s, count) == 0)
                {
                    return size_type(uptr - data());
                }
                else if (uptr == data())
                {
                    break;
                }
            }
        }
        return npos;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::rfind(const_pointer s, size_type pos) const -> size_type
    {
        return rfind(s, pos, traits_type::length(s));
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::rfind(value_type ch, size_type pos) const -> size_type
    {
        return rfind((const_pointer)(&ch), pos, size_type(1));
    }

    /*****************
     * find_first_of *
     *****************/

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::find_first_of(const self_type& str, size_type pos) const noexcept -> size_type
    {
        return find_first_of(str.data(), pos, str.size());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::find_first_of(const string_type& str, size_type pos) const noexcept -> size_type
    {
        return find_first_of(str.data(), pos, str.size());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    auto xbasic_fixed_string<CT, N, ST, EP, TR>::find_first_of(const_pointer s, size_type pos, size_type count) const -> size_type
    {
        if (size_type(0) < count && pos < size())
        {
            const_pointer vptr = data() + size();
            for (const_pointer uptr = data() + pos; uptr < vptr; ++uptr)
            {
                if (traits_type::find(s, count, *uptr) != 0)
                {
                    return size_type(uptr - data());
                }
            }
        }
        return npos;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::find_first_of(const_pointer s, size_type pos) const -> size_type
    {
        return find_first_of(s, pos, traits_type::length(s));
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::find_first_of(value_type ch, size_type pos) const -> size_type
    {
        return find_first_of((const_pointer)(&ch), pos, size_type(1));
    }

    /*********************
     * find_first_not_of *
     *********************/

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::find_first_not_of(const self_type& str, size_type pos) const noexcept -> size_type
    {
        return find_first_not_of(str.data(), pos, str.size());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::find_first_not_of(const string_type& str, size_type pos) const noexcept -> size_type
    {
        return find_first_not_of(str.data(), pos, str.size());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    auto xbasic_fixed_string<CT, N, ST, EP, TR>::find_first_not_of(const_pointer s, size_type pos, size_type count) const -> size_type
    {
        if (pos < size())
        {
            const_pointer vptr = data() + size();
            for (const_pointer uptr = data() + pos; uptr < vptr; ++uptr)
            {
                if (traits_type::find(s, count, *uptr) == 0)
                {
                    return size_type(uptr - data());
                }
            }
        }
        return npos;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::find_first_not_of(const_pointer s, size_type pos) const -> size_type
    {
        return find_first_not_of(s, pos, traits_type::length(s));
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::find_first_not_of(value_type ch, size_type pos) const -> size_type
    {
        return find_first_not_of((const_pointer)(&ch), pos, size_type(1));
    }

    /****************
     * find_last_of *
     ****************/

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::find_last_of(const self_type& str, size_type pos) const noexcept -> size_type
    {
        return find_last_of(str.data(), pos, str.size());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::find_last_of(const string_type& str, size_type pos) const noexcept -> size_type
    {
        return find_last_of(str.data(), pos, str.size());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    auto xbasic_fixed_string<CT, N, ST, EP, TR>::find_last_of(const_pointer s, size_type pos, size_type count) const -> size_type
    {
        if (size_type(0) < count && size_type(0) < size())
        {
            const_pointer uptr = data() + std::min(pos, size() - 1);
            for (;; --uptr)
            {
                if (traits_type::find(s, count, *uptr) != 0)
                {
                    return size_type(uptr - data());
                }
                else if (uptr == data())
                {
                    break;
                }
            }
        }
        return npos;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::find_last_of(const_pointer s, size_type pos) const -> size_type
    {
        return find_last_of(s, pos, traits_type::length(s));
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::find_last_of(value_type ch, size_type pos) const -> size_type
    {
        return find_last_of((const_pointer)(&ch), pos, size_type(1));
    }

    /********************
     * find_last_not_of *
     ********************/

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::find_last_not_of(const self_type& str, size_type pos) const noexcept -> size_type
    {
        return find_last_not_of(str.data(), pos, str.size());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::find_last_not_of(const string_type& str, size_type pos) const noexcept -> size_type
    {
        return find_last_not_of(str.data(), pos, str.size());
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    auto xbasic_fixed_string<CT, N, ST, EP, TR>::find_last_not_of(const_pointer s, size_type pos, size_type count) const -> size_type
    {
        if (size_type(0) < size())
        {
            const_pointer uptr = data() + std::min(pos, size() - 1);
            for (;; --uptr)
            {
                if (traits_type::find(s, count, *uptr) == 0)
                {
                    return size_type(uptr - data());
                }
                else if (uptr == data())
                {
                    break;
                }
            }
        }
        return npos;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::find_last_not_of(const_pointer s, size_type pos) const -> size_type
    {
        return find_last_not_of(s, pos, traits_type::length(s));
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline auto xbasic_fixed_string<CT, N, ST, EP, TR>::find_last_not_of(value_type ch, size_type pos) const -> size_type
    {
        return find_last_not_of((const_pointer)(&ch), pos, size_type(1));
    }

    /*******************
     * Private methods *
     *******************/

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    int xbasic_fixed_string<CT, N, ST, EP, TR>::compare_impl(const_pointer s1, size_type count1,
                                                         const_pointer s2, size_type count2) const noexcept
    {
        size_type rlen = std::min(count1, count2);
        int res = traits_type::compare(s1, s2, rlen);
        if (res == 0)
        {
            return count1 < count2 ? -1 : (count1 > count2 ? 1 : 0);
        }
        else
        {
            return res;
        }
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline void xbasic_fixed_string<CT, N, ST, EP, TR>::update_null_termination() noexcept
    {
        data()[size()] = '\0';
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    void xbasic_fixed_string<CT, N, ST, EP, TR>::check_index(size_type pos, size_type size, const char* what) const
    {
        if (pos >= size)
        {
#if defined(XTL_NO_EXCEPTIONS)
            std::fprintf(stderr, "%s\n", what);
            std::terminate();
#else
            throw std::out_of_range(what);
#endif
        }
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    void xbasic_fixed_string<CT, N, ST, EP, TR>::check_index_strict(size_type pos, size_type size, const char* what) const
    {
        check_index(pos, size + 1, what);
    }

    /**************************
     * Concatenation operator *
     **************************/

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline xbasic_fixed_string<CT, N, ST, EP, TR>
    operator+(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
              const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs)
    {
        xbasic_fixed_string<CT, N, ST, EP, TR> res(lhs);
        return res += rhs;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline xbasic_fixed_string<CT, N, ST, EP, TR>
    operator+(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
              const CT* rhs)
    {
        xbasic_fixed_string<CT, N, ST, EP, TR> res(lhs);
        return res += rhs;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline xbasic_fixed_string<CT, N, ST, EP, TR>
    operator+(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
              CT rhs)
    {
        xbasic_fixed_string<CT, N, ST, EP, TR> res(lhs);
        return res += rhs;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline xbasic_fixed_string<CT, N, ST, EP, TR>
    operator+(const CT* lhs,
              const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs)
    {
        xbasic_fixed_string<CT, N, ST, EP, TR> res(lhs);
        return res += rhs;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline xbasic_fixed_string<CT, N, ST, EP, TR>
    operator+(CT lhs,
              const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs)
    {
        using size_type = typename xbasic_fixed_string<CT, N, ST, EP, TR>::size_type;
        xbasic_fixed_string<CT, N, ST, EP, TR> res(size_type(1), lhs);
        return res += rhs;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline xbasic_fixed_string<CT, N, ST, EP, TR>
    operator+(const xbasic_fixed_string<CT, N, ST, EP, TR>&& lhs,
              const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs)
    {
        xbasic_fixed_string<CT, N, ST, EP, TR> res(std::move(lhs));
        return res += rhs;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline xbasic_fixed_string<CT, N, ST, EP, TR>
    operator+(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
              const xbasic_fixed_string<CT, N, ST, EP, TR>&& rhs)
    {
        xbasic_fixed_string<CT, N, ST, EP, TR> res(lhs);
        return res += std::move(rhs);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline xbasic_fixed_string<CT, N, ST, EP, TR>
    operator+(const xbasic_fixed_string<CT, N, ST, EP, TR>&& lhs,
              const xbasic_fixed_string<CT, N, ST, EP, TR>&& rhs)
    {
        xbasic_fixed_string<CT, N, ST, EP, TR> res(std::move(lhs));
        return res += std::move(rhs);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline xbasic_fixed_string<CT, N, ST, EP, TR>
    operator+(const xbasic_fixed_string<CT, N, ST, EP, TR>&& lhs,
              const CT* rhs)
    {
        xbasic_fixed_string<CT, N, ST, EP, TR> res(std::move(lhs));
        return res += rhs;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline xbasic_fixed_string<CT, N, ST, EP, TR>
    operator+(const xbasic_fixed_string<CT, N, ST, EP, TR>&& lhs,
              CT rhs)
    {
        xbasic_fixed_string<CT, N, ST, EP, TR> res(std::move(lhs));
        return res += rhs;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline xbasic_fixed_string<CT, N, ST, EP, TR>
    operator+(const CT* lhs,
              const xbasic_fixed_string<CT, N, ST, EP, TR>&& rhs)
    {
        xbasic_fixed_string<CT, N, ST, EP, TR> res(lhs);
        return res += std::move(rhs);
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline xbasic_fixed_string<CT, N, ST, EP, TR>
    operator+(CT lhs,
              const xbasic_fixed_string<CT, N, ST, EP, TR>&& rhs)
    {
        using size_type = typename xbasic_fixed_string<CT, N, ST, EP, TR>::size_type;
        xbasic_fixed_string<CT, N, ST, EP, TR> res(size_type(1), lhs);
        return res += std::move(rhs);
    }

    /************************
    * Comparison operators *
    ************************/

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator==(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                           const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept
    {
        return lhs.compare(rhs) == 0;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator==(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                           const CT* rhs) noexcept
    {
        return lhs.compare(rhs) == 0;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator==(const CT* lhs,
                           const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept
    {
        return rhs == lhs;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator==(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                           const std::basic_string<CT, TR>& rhs) noexcept
    {
        return lhs == rhs.c_str();
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator==(const std::basic_string<CT, TR>& lhs,
                           const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept
    {
        return lhs.c_str() == rhs;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator!=(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                           const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept
    {
        return lhs.compare(rhs) != 0;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator!=(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                           const CT* rhs) noexcept
    {
        return lhs.compare(rhs) != 0;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator!=(const CT* lhs,
                           const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept
    {
        return rhs != lhs;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator!=(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                           const std::basic_string<CT, TR>& rhs) noexcept
    {
        return lhs != rhs.c_str();
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator!=(const std::basic_string<CT, TR>& lhs,
                           const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept
    {
        return lhs.c_str() != rhs;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator<(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                          const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept
    {
        return lhs.compare(rhs) < 0;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator<(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                          const CT* rhs) noexcept
    {
        return lhs.compare(rhs) < 0;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator<(const CT* lhs,
                          const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept
    {
        return rhs > lhs;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator<(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                          const std::basic_string<CT, TR>& rhs) noexcept
    {
        return lhs < rhs.c_str();
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator<(const std::basic_string<CT, TR>& lhs,
                          const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept
    {
        return lhs.c_str() < rhs;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator<=(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                           const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept
    {
        return lhs.compare(rhs) <= 0;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator<=(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                           const CT* rhs) noexcept
    {
        return lhs.compare(rhs) <= 0;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator<=(const CT* lhs,
                           const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept
    {
        return rhs >= lhs;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator<=(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                           const std::basic_string<CT, TR>& rhs) noexcept
    {
        return lhs <= rhs.c_str();
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator<=(const std::basic_string<CT, TR>& lhs,
                           const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept
    {
        return lhs.c_str() <= rhs;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator>(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                          const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept
    {
        return lhs.compare(rhs) > 0;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator>(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                          const CT* rhs) noexcept
    {
        return lhs.compare(rhs) > 0;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator>(const CT* lhs,
                          const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept
    {
        return rhs < lhs;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator>(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                          const std::basic_string<CT, TR>& rhs) noexcept
    {
        return lhs > rhs.c_str();
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator>(const std::basic_string<CT, TR>& lhs,
                          const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept
    {
        return lhs.c_str() > rhs;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator>=(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                           const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept
    {
        return lhs.compare(rhs) >= 0;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator>=(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                           const CT* rhs) noexcept
    {
        return lhs.compare(rhs) >= 0;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator>=(const CT* lhs,
                           const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept
    {
        return rhs <= lhs;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator>=(const xbasic_fixed_string<CT, N, ST, EP, TR>& lhs,
                           const std::basic_string<CT, TR>& rhs) noexcept
    {
        return lhs >= rhs.c_str();
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline bool operator>=(const std::basic_string<CT, TR>& lhs,
                           const xbasic_fixed_string<CT, N, ST, EP, TR>& rhs) noexcept
    {
        return lhs.c_str() >= rhs;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline void swap(xbasic_fixed_string<CT, N, ST, EP, TR>& lhs, xbasic_fixed_string<CT, N, ST, EP, TR>& rhs)
    {
        lhs.swap(rhs);
    }

    /******************
     * Input / output *
     ******************/

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline std::basic_ostream<CT, TR>& operator<<(std::basic_ostream<CT, TR>& os,
                                                  const xbasic_fixed_string<CT, N, ST, EP, TR>& str)
    {
        os << str.c_str();
        return os;
    }

#ifdef __CLING__
    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    nlohmann::json mime_bundle_repr(const xbasic_fixed_string<CT, N, ST, EP, TR>& str)
    {
        auto bundle = nlohmann::json::object();
        bundle["text/plain"] = str.c_str();
        return bundle;
    }
#endif

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline std::basic_istream<CT, TR>& operator>>(std::basic_istream<CT, TR>& is,
                                                  xbasic_fixed_string<CT, N, ST, EP, TR>& str)
    {
        // Not optimal
        std::string tmp;
        is >> tmp;
        str = tmp.c_str();
        return is;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline std::basic_istream<CT, TR>& getline(std::basic_istream<CT, TR>& input,
                                               xbasic_fixed_string<CT, N, ST, EP, TR>& str,
                                               CT delim)
    {
        std::string tmp;
        auto& ret = std::getline(input, tmp, delim);
        str = tmp;
        return ret;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline std::basic_istream<CT, TR>& getline(std::basic_istream<CT, TR>&& input,
                                               xbasic_fixed_string<CT, N, ST, EP, TR>& str,
                                               CT delim)
    {
        std::string tmp;
        auto& ret = std::getline(std::move(input), tmp, delim);
        str = tmp;
        return ret;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline std::basic_istream<CT, TR>& getline(std::basic_istream<CT, TR>& input,
                                               xbasic_fixed_string<CT, N, ST, EP, TR>& str)
    {
        std::string tmp;
        auto& ret = std::getline(input, tmp);
        str = tmp;
        return ret;
    }

    template <class CT, std::size_t N, int ST, template <std::size_t> class EP, class TR>
    inline std::basic_istream<CT, TR>& getline(std::basic_istream<CT, TR>&& input,
                                               xbasic_fixed_string<CT, N, ST, EP, TR>& str)
    {
        std::string tmp;
        auto& ret = std::getline(std::move(input), tmp);
        str = tmp;
        return ret;
    }
}

#endif  // xtl
