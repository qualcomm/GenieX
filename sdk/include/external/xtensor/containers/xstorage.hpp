/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_STORAGE_HPP
#define XTENSOR_STORAGE_HPP

#include <algorithm>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <type_traits>

#include "../core/xtensor_config.hpp"
#include "../utils/xexception.hpp"
#include "../utils/xtensor_simd.hpp"
#include "../utils/xutils.hpp"

namespace xt
{
    template <class C>
    struct is_contiguous_container : std::true_type
    {
    };

    template <class T, class A = std::allocator<T>>
    class uvector
    {
    public:

        using allocator_type = A;

        using value_type = typename std::allocator_traits<A>::value_type;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = typename std::allocator_traits<A>::pointer;
        using const_pointer = typename std::allocator_traits<A>::const_pointer;

        using size_type = typename std::allocator_traits<A>::size_type;
        using difference_type = typename std::allocator_traits<A>::difference_type;

        using iterator = pointer;
        using const_iterator = const_pointer;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        uvector() noexcept;
        explicit uvector(const allocator_type& alloc) noexcept;
        explicit uvector(size_type count, const allocator_type& alloc = allocator_type());
        uvector(size_type count, const_reference value, const allocator_type& alloc = allocator_type());

        template <std::input_iterator InputIt>
        uvector(InputIt first, InputIt last, const allocator_type& alloc = allocator_type());

        uvector(std::initializer_list<T> init, const allocator_type& alloc = allocator_type());

        ~uvector();

        uvector(const uvector& rhs);
        uvector(const uvector& rhs, const allocator_type& alloc);
        uvector& operator=(const uvector&);

        uvector(uvector&& rhs) noexcept;
        uvector(uvector&& rhs, const allocator_type& alloc) noexcept;
        uvector& operator=(uvector&& rhs) noexcept;

        allocator_type get_allocator() const noexcept;

        bool empty() const noexcept;
        size_type size() const noexcept;
        void resize(size_type size);
        size_type max_size() const noexcept;
        void reserve(size_type new_cap);
        size_type capacity() const noexcept;
        void shrink_to_fit();
        void clear();

        reference operator[](size_type i);
        const_reference operator[](size_type i) const;

        reference at(size_type i);
        const_reference at(size_type i) const;

        reference front();
        const_reference front() const;

        reference back();
        const_reference back() const;

        pointer data() noexcept;
        const_pointer data() const noexcept;

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

        void swap(uvector& rhs) noexcept;

    private:

        template <class I>
        void init_data(I first, I last);

        void resize_impl(size_type new_size);

        allocator_type m_allocator;

        // Storing a pair of pointers is more efficient for iterating than
        // storing a pointer to the beginning and the size of the container
        pointer p_begin;
        pointer p_end;
    };

    template <class T, class A>
    bool operator==(const uvector<T, A>& lhs, const uvector<T, A>& rhs);

    template <class T, class A>
    bool operator!=(const uvector<T, A>& lhs, const uvector<T, A>& rhs);

    template <class T, class A>
    bool operator<(const uvector<T, A>& lhs, const uvector<T, A>& rhs);

    template <class T, class A>
    bool operator<=(const uvector<T, A>& lhs, const uvector<T, A>& rhs);

    template <class T, class A>
    bool operator>(const uvector<T, A>& lhs, const uvector<T, A>& rhs);

    template <class T, class A>
    bool operator>=(const uvector<T, A>& lhs, const uvector<T, A>& rhs);

    template <class T, class A>
    void swap(uvector<T, A>& lhs, uvector<T, A>& rhs) noexcept;

    /**************************
     * uvector implementation *
     **************************/

    namespace detail
    {
        template <class A>
        inline typename std::allocator_traits<A>::pointer
        safe_init_allocate(A& alloc, typename std::allocator_traits<A>::size_type size)
        {
            using traits = std::allocator_traits<A>;
            using pointer = typename traits::pointer;
            using value_type = typename traits::value_type;
            pointer res = alloc.allocate(size);
            if (!xtrivially_default_constructible<value_type>::value)
            {
                for (pointer p = res; p != res + size; ++p)
                {
                    traits::construct(alloc, p, value_type());
                }
            }
            return res;
        }

        template <class A>
        inline void safe_destroy_deallocate(
            A& alloc,
            typename std::allocator_traits<A>::pointer ptr,
            typename std::allocator_traits<A>::size_type size
        )
        {
            using traits = std::allocator_traits<A>;
            using pointer = typename traits::pointer;
            using value_type = typename traits::value_type;
            if (ptr != nullptr)
            {
                if (!xtrivially_default_constructible<value_type>::value)
                {
                    for (pointer p = ptr; p != ptr + size; ++p)
                    {
                        traits::destroy(alloc, p);
                    }
                }
                traits::deallocate(alloc, ptr, size);
            }
        }
    }

    template <class T, class A>
    template <class I>
    inline void uvector<T, A>::init_data(I first, I last)
    {
        size_type size = static_cast<size_type>(std::distance(first, last));
        if (size != size_type(0))
        {
            p_begin = m_allocator.allocate(size);
            std::uninitialized_copy(first, last, p_begin);
            p_end = p_begin + size;
        }
    }

    template <class T, class A>
    inline void uvector<T, A>::resize_impl(size_type new_size)
    {
        size_type old_size = size();
        pointer old_begin = p_begin;
        if (new_size != old_size)
        {
            p_begin = detail::safe_init_allocate(m_allocator, new_size);
            p_end = p_begin + new_size;
            detail::safe_destroy_deallocate(m_allocator, old_begin, old_size);
        }
    }

    template <class T, class A>
    inline uvector<T, A>::uvector() noexcept
        : uvector(allocator_type())
    {
    }

    template <class T, class A>
    inline uvector<T, A>::uvector(const allocator_type& alloc) noexcept
        : m_allocator(alloc)
        , p_begin(nullptr)
        , p_end(nullptr)
    {
    }

    template <class T, class A>
    inline uvector<T, A>::uvector(size_type count, const allocator_type& alloc)
        : m_allocator(alloc)
        , p_begin(nullptr)
        , p_end(nullptr)
    {
        if (count != 0)
        {
            p_begin = detail::safe_init_allocate(m_allocator, count);
            p_end = p_begin + count;
        }
    }

    template <class T, class A>
    inline uvector<T, A>::uvector(size_type count, const_reference value, const allocator_type& alloc)
        : m_allocator(alloc)
        , p_begin(nullptr)
        , p_end(nullptr)
    {
        if (count != 0)
        {
            p_begin = m_allocator.allocate(count);
            p_end = p_begin + count;
            std::uninitialized_fill(p_begin, p_end, value);
        }
    }

    template <class T, class A>
    template <std::input_iterator InputIt>
    inline uvector<T, A>::uvector(InputIt first, InputIt last, const allocator_type& alloc)
        : m_allocator(alloc)
        , p_begin(nullptr)
        , p_end(nullptr)
    {
        init_data(first, last);
    }

    template <class T, class A>
    inline uvector<T, A>::uvector(std::initializer_list<T> init, const allocator_type& alloc)
        : m_allocator(alloc)
        , p_begin(nullptr)
        , p_end(nullptr)
    {
        init_data(init.begin(), init.end());
    }

    template <class T, class A>
    inline uvector<T, A>::~uvector()
    {
        detail::safe_destroy_deallocate(m_allocator, p_begin, size());
        p_begin = nullptr;
        p_end = nullptr;
    }

    template <class T, class A>
    inline uvector<T, A>::uvector(const uvector& rhs)
        : m_allocator(
            std::allocator_traits<allocator_type>::select_on_container_copy_construction(rhs.get_allocator())
        )
        , p_begin(nullptr)
        , p_end(nullptr)
    {
        init_data(rhs.p_begin, rhs.p_end);
    }

    template <class T, class A>
    inline uvector<T, A>::uvector(const uvector& rhs, const allocator_type& alloc)
        : m_allocator(alloc)
        , p_begin(nullptr)
        , p_end(nullptr)
    {
        init_data(rhs.p_begin, rhs.p_end);
    }

    template <class T, class A>
    inline uvector<T, A>& uvector<T, A>::operator=(const uvector& rhs)
    {
        // No copy and swap idiom here due to performance issues
        if (this != &rhs)
        {
            m_allocator = std::allocator_traits<allocator_type>::select_on_container_copy_construction(
                rhs.get_allocator()
            );
            resize_impl(rhs.size());
            if (xtrivially_default_constructible<value_type>::value)
            {
                std::uninitialized_copy(rhs.p_begin, rhs.p_end, p_begin);
            }
            else
            {
                std::copy(rhs.p_begin, rhs.p_end, p_begin);
            }
        }
        return *this;
    }

    template <class T, class A>
    inline uvector<T, A>::uvector(uvector&& rhs) noexcept
        : m_allocator(std::move(rhs.m_allocator))
        , p_begin(rhs.p_begin)
        , p_end(rhs.p_end)
    {
        rhs.p_begin = nullptr;
        rhs.p_end = nullptr;
    }

    template <class T, class A>
    inline uvector<T, A>::uvector(uvector&& rhs, const allocator_type& alloc) noexcept
        : m_allocator(alloc)
        , p_begin(rhs.p_begin)
        , p_end(rhs.p_end)
    {
        rhs.p_begin = nullptr;
        rhs.p_end = nullptr;
    }

    template <class T, class A>
    inline uvector<T, A>& uvector<T, A>::operator=(uvector&& rhs) noexcept
    {
        using std::swap;
        uvector tmp(std::move(rhs));
        swap(p_begin, tmp.p_begin);
        swap(p_end, tmp.p_end);
        return *this;
    }

    template <class T, class A>
    inline auto uvector<T, A>::get_allocator() const noexcept -> allocator_type
    {
        return allocator_type(m_allocator);
    }

    template <class T, class A>
    inline bool uvector<T, A>::empty() const noexcept
    {
        return size() == size_type(0);
    }

    template <class T, class A>
    inline auto uvector<T, A>::size() const noexcept -> size_type
    {
        return static_cast<size_type>(p_end - p_begin);
    }

    template <class T, class A>
    inline void uvector<T, A>::resize(size_type size)
    {
        resize_impl(size);
    }

    template <class T, class A>
    inline auto uvector<T, A>::max_size() const noexcept -> size_type
    {
        return m_allocator.max_size();
    }

    template <class T, class A>
    inline void uvector<T, A>::reserve(size_type /*new_cap*/)
    {
    }

    template <class T, class A>
    inline auto uvector<T, A>::capacity() const noexcept -> size_type
    {
        return size();
    }

    template <class T, class A>
    inline void uvector<T, A>::shrink_to_fit()
    {
    }

    template <class T, class A>
    inline void uvector<T, A>::clear()
    {
        resize(size_type(0));
    }

    template <class T, class A>
    inline auto uvector<T, A>::operator[](size_type i) -> reference
    {
        return p_begin[i];
    }

    template <class T, class A>
    inline auto uvector<T, A>::operator[](size_type i) const -> const_reference
    {
        return p_begin[i];
    }

    template <class T, class A>
    inline auto uvector<T, A>::at(size_type i) -> reference
    {
        if (i >= size())
        {
            XTENSOR_THROW(std::out_of_range, "Out of range in uvector access");
        }
        return this->operator[](i);
    }

    template <class T, class A>
    inline auto uvector<T, A>::at(size_type i) const -> const_reference
    {
        if (i >= size())
        {
            XTENSOR_THROW(std::out_of_range, "Out of range in uvector access");
        }
        return this->operator[](i);
    }

    template <class T, class A>
    inline auto uvector<T, A>::front() -> reference
    {
        return p_begin[0];
    }

    template <class T, class A>
    inline auto uvector<T, A>::front() const -> const_reference
    {
        return p_begin[0];
    }

    template <class T, class A>
    inline auto uvector<T, A>::back() -> reference
    {
        return *(p_end - 1);
    }

    template <class T, class A>
    inline auto uvector<T, A>::back() const -> const_reference
    {
        return *(p_end - 1);
    }

    template <class T, class A>
    inline auto uvector<T, A>::data() noexcept -> pointer
    {
        return p_begin;
    }

    template <class T, class A>
    inline auto uvector<T, A>::data() const noexcept -> const_pointer
    {
        return p_begin;
    }

    template <class T, class A>
    inline auto uvector<T, A>::begin() noexcept -> iterator
    {
        return p_begin;
    }

    template <class T, class A>
    inline auto uvector<T, A>::end() noexcept -> iterator
    {
        return p_end;
    }

    template <class T, class A>
    inline auto uvector<T, A>::begin() const noexcept -> const_iterator
    {
        return p_begin;
    }

    template <class T, class A>
    inline auto uvector<T, A>::end() const noexcept -> const_iterator
    {
        return p_end;
    }

    template <class T, class A>
    inline auto uvector<T, A>::cbegin() const noexcept -> const_iterator
    {
        return begin();
    }

    template <class T, class A>
    inline auto uvector<T, A>::cend() const noexcept -> const_iterator
    {
        return end();
    }

    template <class T, class A>
    inline auto uvector<T, A>::rbegin() noexcept -> reverse_iterator
    {
        return reverse_iterator(end());
    }

    template <class T, class A>
    inline auto uvector<T, A>::rend() noexcept -> reverse_iterator
    {
        return reverse_iterator(begin());
    }

    template <class T, class A>
    inline auto uvector<T, A>::rbegin() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator(end());
    }

    template <class T, class A>
    inline auto uvector<T, A>::rend() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator(begin());
    }

    template <class T, class A>
    inline auto uvector<T, A>::crbegin() const noexcept -> const_reverse_iterator
    {
        return rbegin();
    }

    template <class T, class A>
    inline auto uvector<T, A>::crend() const noexcept -> const_reverse_iterator
    {
        return rend();
    }

    template <class T, class A>
    inline void uvector<T, A>::swap(uvector<T, A>& rhs) noexcept
    {
        using std::swap;
        swap(m_allocator, rhs.m_allocator);
        swap(p_begin, rhs.p_begin);
        swap(p_end, rhs.p_end);
    }

    template <class T, class A>
    inline bool operator==(const uvector<T, A>& lhs, const uvector<T, A>& rhs)
    {
        return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    template <class T, class A>
    inline bool operator!=(const uvector<T, A>& lhs, const uvector<T, A>& rhs)
    {
        return !(lhs == rhs);
    }

    template <class T, class A>
    inline bool operator<(const uvector<T, A>& lhs, const uvector<T, A>& rhs)
    {
        return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
    }

    template <class T, class A>
    inline bool operator<=(const uvector<T, A>& lhs, const uvector<T, A>& rhs)
    {
        return !(lhs > rhs);
    }

    template <class T, class A>
    inline bool operator>(const uvector<T, A>& lhs, const uvector<T, A>& rhs)
    {
        return rhs < lhs;
    }

    template <class T, class A>
    inline bool operator>=(const uvector<T, A>& lhs, const uvector<T, A>& rhs)
    {
        return !(lhs < rhs);
    }

    template <class T, class A>
    inline void swap(uvector<T, A>& lhs, uvector<T, A>& rhs) noexcept
    {
        lhs.swap(rhs);
    }

    /**************************
     * svector implementation *
     **************************/

    namespace detail
    {
        template <class T>
        struct allocator_alignment
        {
            static constexpr std::size_t value = 0;
        };

        template <class T, std::size_t A>
        struct allocator_alignment<xt_simd::aligned_allocator<T, A>>
        {
            static constexpr std::size_t value = A;
        };
    }

    template <class T, std::size_t N = 4, class A = std::allocator<T>, bool Init = true>
    class svector
    {
    public:

        using self_type = svector<T, N, A, Init>;
        using allocator_type = A;
        using size_type = typename std::allocator_traits<A>::size_type;
        using value_type = typename std::allocator_traits<A>::value_type;
        using pointer = typename std::allocator_traits<A>::pointer;
        using const_pointer = typename std::allocator_traits<A>::const_pointer;
        using reference = value_type&;
        using const_reference = const value_type&;
        using difference_type = typename std::allocator_traits<A>::difference_type;

        using iterator = pointer;
        using const_iterator = const_pointer;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

#if defined(_MSC_VER) && _MSC_VER < 1910
        static constexpr std::size_t alignment = detail::allocator_alignment<A>::value;
#else
        static constexpr std::size_t alignment = detail::allocator_alignment<A>::value != 0
                                                     ? detail::allocator_alignment<A>::value
                                                     : alignof(T);
#endif

        svector() noexcept;
        ~svector();

        explicit svector(const allocator_type& alloc) noexcept;
        explicit svector(size_type n, const allocator_type& alloc = allocator_type());
        svector(size_type n, const value_type& v, const allocator_type& alloc = allocator_type());
        svector(std::initializer_list<T> il, const allocator_type& alloc = allocator_type());

        svector(const std::vector<T>& vec);

        template <std::input_iterator IT>
        svector(IT begin, IT end, const allocator_type& alloc = allocator_type());

        template <std::size_t N2, bool I2>
        explicit svector(const svector<T, N2, A, I2>& rhs)
            requires(N != N2);

        svector& operator=(const svector& rhs);
        svector& operator=(svector&& rhs) noexcept(std::is_nothrow_move_assignable<value_type>::value);
        svector& operator=(const std::vector<T>& rhs);
        svector& operator=(std::initializer_list<T> il);

        template <std::size_t N2, bool I2>
        svector& operator=(const svector<T, N2, A, I2>& rhs)
            requires(N != N2);

        svector(const svector& other);
        svector(svector&& other) noexcept(std::is_nothrow_move_constructible<value_type>::value);

        void assign(size_type n, const value_type& v);

        template <class V>
        void assign(std::initializer_list<V> il);

        template <class IT>
        void assign(IT other_begin, IT other_end);

        reference operator[](size_type idx);
        const_reference operator[](size_type idx) const;

        reference at(size_type idx);
        const_reference at(size_type idx) const;

        pointer data();
        const_pointer data() const;

        void push_back(const T& elt);
        void push_back(T&& elt);
        void pop_back();

        iterator begin();
        const_iterator begin() const;
        const_iterator cbegin() const;
        iterator end();
        const_iterator end() const;
        const_iterator cend() const;

        reverse_iterator rbegin();
        const_reverse_iterator rbegin() const;
        const_reverse_iterator crbegin() const;
        reverse_iterator rend();
        const_reverse_iterator rend() const;
        const_reverse_iterator crend() const;

        bool empty() const;
        size_type size() const;
        void resize(size_type n);
        size_type max_size() const noexcept;
        size_type capacity() const;
        void reserve(size_type n);
        void shrink_to_fit();
        void clear();

        reference front();
        const_reference front() const;
        reference back();
        const_reference back() const;

        bool on_stack();

        iterator erase(const_iterator cit);
        iterator erase(const_iterator cfirst, const_iterator clast);

        iterator insert(const_iterator it, const T& elt);

        template <class It>
        iterator insert(const_iterator pos, It first, It last);

        iterator insert(const_iterator pos, std::initializer_list<T> l);

        template <std::size_t ON, class OA, bool InitA>
        void swap(svector<T, ON, OA, InitA>& rhs);

        allocator_type get_allocator() const noexcept;

    private:

        A m_allocator;

        T* m_begin = std::begin(m_data);
        T* m_end = std::begin(m_data);
        T* m_capacity = std::end(m_data);

        // stack allocated memory
        alignas(alignment) T m_data[N > 0 ? N : 1];

        void grow(size_type min_capacity = 0);
        void destroy_range(T* begin, T* end);
    };

    template <class T, std::size_t N, class A, bool Init>
    inline svector<T, N, A, Init>::~svector()
    {
        if (!on_stack())
        {
            detail::safe_destroy_deallocate(m_allocator, m_begin, static_cast<std::size_t>(m_capacity - m_begin));
        }
    }

    template <class T, std::size_t N, class A, bool Init>
    inline svector<T, N, A, Init>::svector() noexcept
        : svector(allocator_type())
    {
    }

    template <class T, std::size_t N, class A, bool Init>
    inline svector<T, N, A, Init>::svector(const allocator_type& alloc) noexcept
        : m_allocator(alloc)
    {
    }

    template <class T, std::size_t N, class A, bool Init>
    inline svector<T, N, A, Init>::svector(size_type n, const allocator_type& alloc)
        : m_allocator(alloc)
    {
        if (Init)
        {
            assign(n, T(0));
        }
        else
        {
            resize(n);
        }
    }

    template <class T, std::size_t N, class A, bool Init>
    template <std::input_iterator IT>
    inline svector<T, N, A, Init>::svector(IT begin, IT end, const allocator_type& alloc)
        : m_allocator(alloc)
    {
        assign(begin, end);
    }

    template <class T, std::size_t N, class A, bool Init>
    template <std::size_t N2, bool I2>
    inline svector<T, N, A, Init>::svector(const svector<T, N2, A, I2>& rhs)
        requires(N != N2)
        : m_allocator(rhs.get_allocator())
    {
        assign(rhs.begin(), rhs.end());
    }

    template <class T, std::size_t N, class A, bool Init>
    inline svector<T, N, A, Init>::svector(const std::vector<T>& vec)
    {
        assign(vec.begin(), vec.end());
    }

    template <class T, std::size_t N, class A, bool Init>
    inline svector<T, N, A, Init>::svector(size_type n, const value_type& v, const allocator_type& alloc)
        : m_allocator(alloc)
    {
        assign(n, v);
    }

    template <class T, std::size_t N, class A, bool Init>
    inline svector<T, N, A, Init>::svector(std::initializer_list<T> il, const allocator_type& alloc)
        : m_allocator(alloc)
    {
        assign(il.begin(), il.end());
    }

    template <class T, std::size_t N, class A, bool Init>
    inline svector<T, N, A, Init>& svector<T, N, A, Init>::operator=(const svector& rhs)
    {
        assign(rhs.begin(), rhs.end());
        return *this;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline svector<T, N, A, Init>& svector<T, N, A, Init>::operator=(svector&& rhs
    ) noexcept(std::is_nothrow_move_assignable<value_type>::value)
    {
        assign(rhs.begin(), rhs.end());
        return *this;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline svector<T, N, A, Init>& svector<T, N, A, Init>::operator=(const std::vector<T>& rhs)
    {
        m_allocator = std::allocator_traits<allocator_type>::select_on_container_copy_construction(
            rhs.get_allocator()
        );
        assign(rhs.begin(), rhs.end());
        return *this;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline svector<T, N, A, Init>& svector<T, N, A, Init>::operator=(std::initializer_list<T> il)
    {
        return operator=(self_type(il));
    }

    template <class T, std::size_t N, class A, bool Init>
    template <std::size_t N2, bool I2>
    inline svector<T, N, A, Init>& svector<T, N, A, Init>::operator=(const svector<T, N2, A, I2>& rhs)
        requires(N != N2)
    {
        m_allocator = std::allocator_traits<allocator_type>::select_on_container_copy_construction(
            rhs.get_allocator()
        );
        assign(rhs.begin(), rhs.end());
        return *this;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline svector<T, N, A, Init>::svector(const svector& rhs)
        : m_allocator(
            std::allocator_traits<allocator_type>::select_on_container_copy_construction(rhs.get_allocator())
        )
    {
        assign(rhs.begin(), rhs.end());
    }

    template <class T, std::size_t N, class A, bool Init>
    inline svector<T, N, A, Init>::svector(svector&& rhs
    ) noexcept(std::is_nothrow_move_constructible<value_type>::value)
    {
        this->swap(rhs);
    }

    template <class T, std::size_t N, class A, bool Init>
    inline void svector<T, N, A, Init>::assign(size_type n, const value_type& v)
    {
        if (n > N && n > capacity())
        {
            grow(n);
        }
        m_end = m_begin + n;
        std::fill(begin(), end(), v);
    }

    template <class T, std::size_t N, class A, bool Init>
    template <class V>
    inline void svector<T, N, A, Init>::assign(std::initializer_list<V> il)
    {
        assign(il.begin(), il.end());
    }

    template <class T, std::size_t N, class A, bool Init>
    template <class IT>
    inline void svector<T, N, A, Init>::assign(IT other_begin, IT other_end)
    {
        std::size_t size = static_cast<std::size_t>(other_end - other_begin);
        if (size > N && size > capacity())
        {
            grow(size);
        }
        std::uninitialized_copy(other_begin, other_end, m_begin);
        m_end = m_begin + size;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::operator[](size_type idx) -> reference
    {
        return m_begin[idx];
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::operator[](size_type idx) const -> const_reference
    {
        return m_begin[idx];
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::at(size_type idx) -> reference
    {
        if (idx >= size())
        {
            XTENSOR_THROW(std::out_of_range, "Out of range in svector access");
        }
        return this->operator[](idx);
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::at(size_type idx) const -> const_reference
    {
        if (idx >= size())
        {
            XTENSOR_THROW(std::out_of_range, "Out of range in svector access");
        }
        return this->operator[](idx);
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::data() -> pointer
    {
        return m_begin;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::data() const -> const_pointer
    {
        return m_begin;
    }

    template <class T, std::size_t N, class A, bool Init>
    void svector<T, N, A, Init>::resize(size_type n)
    {
        if (n > N && n > capacity())
        {
            grow(n);
        }
        size_type old_size = size();
        m_end = m_begin + n;
        if (Init && old_size < size())
        {
            std::fill(begin() + old_size, end(), T());
        }
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::max_size() const noexcept -> size_type
    {
        return m_allocator.max_size();
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::capacity() const -> size_type
    {
        return static_cast<std::size_t>(m_capacity - m_begin);
    }

    template <class T, std::size_t N, class A, bool Init>
    inline void svector<T, N, A, Init>::reserve(size_type n)
    {
        if (n > N && n > capacity())
        {
            grow(n);
        }
    }

    template <class T, std::size_t N, class A, bool Init>
    inline void svector<T, N, A, Init>::shrink_to_fit()
    {
        // No op for now
    }

    template <class T, std::size_t N, class A, bool Init>
    inline void svector<T, N, A, Init>::clear()
    {
        resize(size_type(0));
    }

    template <class T, std::size_t N, class A, bool Init>
    void svector<T, N, A, Init>::push_back(const T& elt)
    {
        if (m_end >= m_capacity)
        {
            grow();
        }
        *(m_end++) = elt;
    }

    template <class T, std::size_t N, class A, bool Init>
    void svector<T, N, A, Init>::push_back(T&& elt)
    {
        if (m_end >= m_capacity)
        {
            grow();
        }
        *(m_end++) = std::move(elt);
    }

    template <class T, std::size_t N, class A, bool Init>
    void svector<T, N, A, Init>::pop_back()
    {
        --m_end;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::begin() -> iterator
    {
        return m_begin;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::begin() const -> const_iterator
    {
        return m_begin;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::cbegin() const -> const_iterator
    {
        return m_begin;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::end() -> iterator
    {
        return m_end;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::end() const -> const_iterator
    {
        return m_end;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::cend() const -> const_iterator
    {
        return m_end;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::rbegin() -> reverse_iterator
    {
        return reverse_iterator(m_end);
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::rbegin() const -> const_reverse_iterator
    {
        return const_reverse_iterator(m_end);
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::crbegin() const -> const_reverse_iterator
    {
        return const_reverse_iterator(m_end);
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::rend() -> reverse_iterator
    {
        return reverse_iterator(m_begin);
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::rend() const -> const_reverse_iterator
    {
        return const_reverse_iterator(m_begin);
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::crend() const -> const_reverse_iterator
    {
        return const_reverse_iterator(m_begin);
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::size() const -> size_type
    {
        return static_cast<size_type>(m_end - m_begin);
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::empty() const -> bool
    {
        return m_begin == m_end;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::front() -> reference
    {
        XTENSOR_ASSERT(!empty());
        return m_begin[0];
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::front() const -> const_reference
    {
        XTENSOR_ASSERT(!empty());
        return m_begin[0];
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::back() -> reference
    {
        XTENSOR_ASSERT(!empty());
        return m_end[-1];
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::back() const -> const_reference
    {
        XTENSOR_ASSERT(!empty());
        return m_end[-1];
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::on_stack() -> bool
    {
        return m_begin == &m_data[0];
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::get_allocator() const noexcept -> allocator_type
    {
        return m_allocator;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::erase(const_iterator cit) -> iterator
    {
        auto it = const_cast<pointer>(cit);
        iterator ret_val = it;
        std::move(it + 1, m_end, it);
        --m_end;
        return ret_val;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::erase(const_iterator cfirst, const_iterator clast) -> iterator
    {
        auto first = const_cast<pointer>(cfirst);
        auto last = const_cast<pointer>(clast);
        if (last == m_end)
        {
            m_end = first;
            return first;
        }

        iterator new_end = std::move(last, m_end, first);
        m_end = new_end;
        return first;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::insert(const_iterator cit, const T& elt) -> iterator
    {
        auto it = const_cast<pointer>(cit);
        if (it == m_end)
        {
            push_back(elt);
            return m_end - 1;
        }

        if (m_end >= m_capacity)
        {
            std::ptrdiff_t elt_no = it - m_begin;
            grow();
            it = m_begin + elt_no;
        }

        (*m_end) = back();
        std::move_backward(it, m_end - 1, m_end);
        ++m_end;

        // Update ref if element moved
        const T* elt_ptr = &elt;
        bool cond = it <= elt_ptr && elt_ptr < m_end;
        // More complicated than incrementing elt_ptr, but this avoids
        // false positive array-bounds warning on GCC 10
        const T* src_ptr = cond ? it + (elt_ptr - it) + std::ptrdiff_t(1) : elt_ptr;
        *it = *src_ptr;
        return it;
    }

    template <class T, std::size_t N, class A, bool Init>
    template <class It>
    inline auto svector<T, N, A, Init>::insert(const_iterator pos, It first, It last) -> iterator
    {
        auto it = const_cast<pointer>(pos);
        difference_type n = std::distance(first, last);
        if (n > 0)
        {
            if (n > m_capacity - m_end)
            {
                std::ptrdiff_t elt_no = it - m_begin;
                grow(static_cast<size_t>((m_capacity - m_begin) + n));
                it = m_begin + elt_no;
            }

            std::move_backward(it, m_end, m_end + n);
            m_end += n;
            std::copy(first, last, it);
        }
        return it;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline auto svector<T, N, A, Init>::insert(const_iterator pos, std::initializer_list<T> l) -> iterator
    {
        return insert(pos, l.begin(), l.end());
    }

    template <class T, std::size_t N, class A, bool Init>
    inline void svector<T, N, A, Init>::destroy_range(T* begin, T* end)
    {
        if (!xtrivially_default_constructible<T>::value)
        {
            while (begin != end)
            {
                --end;
                end->~T();
            }
        }
    }

    template <class T, std::size_t N, class A, bool Init>
    template <std::size_t ON, class OA, bool InitA>
    inline void svector<T, N, A, Init>::swap(svector<T, ON, OA, InitA>& rhs)
    {
        using std::swap;
        if (this == &rhs)
        {
            return;
        }

        // We can only avoid copying elements if neither vector is small.
        if (!this->on_stack() && !rhs.on_stack())
        {
            swap(this->m_begin, rhs.m_begin);
            swap(this->m_end, rhs.m_end);
            swap(this->m_capacity, rhs.m_capacity);
            return;
        }

        size_type rhs_old_size = rhs.size();
        size_type old_size = this->size();

        if (rhs_old_size > old_size)
        {
            this->resize(rhs_old_size);
        }
        else if (old_size > rhs_old_size)
        {
            rhs.resize(old_size);
        }

        // Swap the shared elements.
        size_type min_size = (std::min)(old_size, rhs_old_size);
        for (size_type i = 0; i < min_size; ++i)
        {
            swap((*this)[i], rhs[i]);
        }

        // Copy over the extra elts.
        if (old_size > rhs_old_size)
        {
            std::copy(this->begin() + min_size, this->end(), rhs.begin() + min_size);
            this->destroy_range(this->begin() + min_size, this->end());
            this->m_end = this->begin() + min_size;
        }
        else if (rhs_old_size > old_size)
        {
            std::copy(rhs.begin() + min_size, rhs.end(), this->begin() + min_size);
            this->destroy_range(rhs.begin() + min_size, rhs.end());
            rhs.m_end = rhs.begin() + min_size;
        }
    }

    template <class T, std::size_t N, class A, bool Init>
    inline void svector<T, N, A, Init>::grow(size_type min_capacity)
    {
        size_type current_size = size();
        size_type new_capacity = 2 * current_size + 1;  // Always grow.
        if (new_capacity < min_capacity)
        {
            new_capacity = min_capacity;
        }

        T* new_alloc;
        // is data stack allocated?
        if (m_begin == &m_data[0])
        {
            new_alloc = m_allocator.allocate(new_capacity);
            std::uninitialized_copy(m_begin, m_end, new_alloc);
        }
        else
        {
            // If this wasn't grown from the inline copy, grow the allocated space.
            new_alloc = m_allocator.allocate(new_capacity);
            std::uninitialized_copy(m_begin, m_end, new_alloc);
            m_allocator.deallocate(m_begin, std::size_t(m_capacity - m_begin));
        }
        XTENSOR_ASSERT(new_alloc);

        m_end = new_alloc + current_size;
        m_begin = new_alloc;
        m_capacity = new_alloc + new_capacity;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline bool operator==(const std::vector<T>& lhs, const svector<T, N, A, Init>& rhs)
    {
        return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    template <class T, std::size_t N, class A, bool Init>
    inline bool operator==(const svector<T, N, A, Init>& lhs, const std::vector<T>& rhs)
    {
        return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    template <class T, std::size_t N, class A, bool Init>
    inline bool operator==(const svector<T, N, A, Init>& lhs, const svector<T, N, A, Init>& rhs)
    {
        return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    template <class T, std::size_t N, class A, bool Init>
    inline bool operator!=(const svector<T, N, A, Init>& lhs, const svector<T, N, A, Init>& rhs)
    {
        return !(lhs == rhs);
    }

    template <class T, std::size_t N, class A, bool Init>
    inline bool operator<(const svector<T, N, A, Init>& lhs, const svector<T, N, A, Init>& rhs)
    {
        return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
    }

    template <class T, std::size_t N, class A, bool Init>
    inline bool operator<=(const svector<T, N, A, Init>& lhs, const svector<T, N, A, Init>& rhs)
    {
        return !(lhs > rhs);
    }

    template <class T, std::size_t N, class A, bool Init>
    inline bool operator>(const svector<T, N, A, Init>& lhs, const svector<T, N, A, Init>& rhs)
    {
        return rhs < lhs;
    }

    template <class T, std::size_t N, class A, bool Init>
    inline bool operator>=(const svector<T, N, A, Init>& lhs, const svector<T, N, A, Init>& rhs)
    {
        return !(lhs < rhs);
    }

    template <class T, std::size_t N, class A, bool Init>
    inline void swap(svector<T, N, A, Init>& lhs, svector<T, N, A, Init>& rhs) noexcept
    {
        lhs.swap(rhs);
    }

    template <class X, class T, std::size_t N, class A, bool B>
    struct rebind_container<X, svector<T, N, A, B>>
    {
        using traits = std::allocator_traits<A>;
        using allocator = typename traits::template rebind_alloc<X>;
        using type = svector<X, N, allocator, B>;
    };

    /**
     * This array class is modeled after ``std::array`` but adds optional alignment through a template
     * parameter.
     *
     * To be moved to xtl, along with the rest of xstorage.hpp
     */
    template <class T, std::size_t N, std::size_t Align = XTENSOR_SELECT_ALIGN(T)>
    class alignas(Align) aligned_array : public std::array<T, N>
    {
    public:

        // Note: this is for alignment detection. The allocator serves no other purpose than
        //       that of a trait here.
        using allocator_type = std::conditional_t<Align != 0, xt_simd::aligned_allocator<T, Align>, std::allocator<T>>;
    };

#if defined(_MSC_VER)
#define XTENSOR_CONST
#else
#define XTENSOR_CONST const
#endif

#if defined(__GNUC__) && __GNUC__ < 5 && !defined(__clang__)
#define GCC4_FALLBACK

    namespace const_array_detail
    {
        template <class T, std::size_t N>
        struct array_traits
        {
            using storage_type = T[N];

            static constexpr T& ref(const storage_type& t, std::size_t n) noexcept
            {
                return const_cast<T&>(t[n]);
            }

            static constexpr T* ptr(const storage_type& t) noexcept
            {
                return const_cast<T*>(t);
            }
        };

        template <class T>
        struct array_traits<T, 0>
        {
            struct empty
            {
            };

            using storage_type = empty;

            static constexpr T& ref(const storage_type& /*t*/, std::size_t /*n*/) noexcept
            {
                return *static_cast<T*>(nullptr);
            }

            static constexpr T* ptr(const storage_type& /*t*/) noexcept
            {
                return nullptr;
            }
        };
    }
#endif

    /**
     * A std::array like class with all member function (except reverse iterators)
     * as constexpr. The data is immutable once set.
     */
    template <class T, std::size_t N>
    struct const_array
    {
        using size_type = std::size_t;
        using value_type = T;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using reference = value_type&;
        using const_reference = const value_type&;
        using difference_type = std::ptrdiff_t;
        using iterator = pointer;
        using const_iterator = const_pointer;

        using reverse_iterator = std::reverse_iterator<const_iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        constexpr const_reference operator[](std::size_t idx) const
        {
#ifdef GCC4_FALLBACK
            return const_array_detail::array_traits<T, N>::ref(m_data, idx);
#else
            return m_data[idx];
#endif
        }

        constexpr const_iterator begin() const noexcept
        {
            return cbegin();
        }

        constexpr const_iterator end() const noexcept
        {
            return cend();
        }

        constexpr const_iterator cbegin() const noexcept
        {
            return data();
        }

        constexpr const_iterator cend() const noexcept
        {
            return data() + N;
        }

        // TODO make constexpr once C++17 arrives
        reverse_iterator rbegin() const noexcept
        {
            return crbegin();
        }

        reverse_iterator rend() const noexcept
        {
            return crend();
        }

        const_reverse_iterator crbegin() const noexcept
        {
            return const_reverse_iterator(end());
        }

        const_reverse_iterator crend() const noexcept
        {
            return const_reverse_iterator(begin());
        }

        constexpr const_pointer data() const noexcept
        {
#ifdef GCC4_FALLBACK
            return const_array_detail::array_traits<T, N>::ptr(m_data);
#else
            return m_data;
#endif
        }

        constexpr const_reference front() const noexcept
        {
#ifdef GCC4_FALLBACK
            return const_array_detail::array_traits<T, N>::ref(m_data, 0);
#else
            return m_data[0];
#endif
        }

        constexpr const_reference back() const noexcept
        {
#ifdef GCC4_FALLBACK
            return N ? const_array_detail::array_traits<T, N>::ref(m_data, N - 1)
                     : const_array_detail::array_traits<T, N>::ref(m_data, 0);
#else
            return m_data[size() - 1];
#endif
        }

        constexpr bool empty() const noexcept
        {
            return size() == size_type(0);
        }

        constexpr size_type size() const noexcept
        {
            return N;
        }

#ifdef GCC4_FALLBACK
        XTENSOR_CONST typename const_array_detail::array_traits<T, N>::storage_type m_data;
#else
        XTENSOR_CONST T m_data[N > 0 ? N : 1];
#endif
    };

#undef GCC4_FALLBACK

    template <class T, std::size_t N>
    inline bool operator==(const const_array<T, N>& lhs, const const_array<T, N>& rhs)
    {
        return std::equal(lhs.cbegin(), lhs.cend(), rhs.cbegin());
    }

    template <class T, std::size_t N>
    inline bool operator!=(const const_array<T, N>& lhs, const const_array<T, N>& rhs)
    {
        return !(lhs == rhs);
    }

    template <class T, std::size_t N>
    inline bool operator<(const const_array<T, N>& lhs, const const_array<T, N>& rhs)
    {
        return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
    }

    template <class T, std::size_t N>
    inline bool operator<=(const const_array<T, N>& lhs, const const_array<T, N>& rhs)
    {
        return !(lhs > rhs);
    }

    template <class T, std::size_t N>
    inline bool operator>(const const_array<T, N>& lhs, const const_array<T, N>& rhs)
    {
        return rhs < lhs;
    }

    template <class T, std::size_t N>
    inline bool operator>=(const const_array<T, N>& lhs, const const_array<T, N>& rhs)
    {
        return !(lhs < rhs);
    }

// Workaround for rebind_container problems when C++17 feature is enabled
#ifdef __cpp_template_template_args
    template <class X, class T, std::size_t N>
    struct rebind_container<X, aligned_array<T, N>>
    {
        using type = aligned_array<X, N>;
    };

    template <class X, class T, std::size_t N>
    struct rebind_container<X, const_array<T, N>>
    {
        using type = const_array<X, N>;
    };
#endif

    /**
     * @class fixed_shape
     * Fixed shape implementation for compile time defined arrays.
     * @sa xshape
     */
    template <std::size_t... X>
    class fixed_shape
    {
    public:

#if defined(_MSC_VER)
        using cast_type = std::array<std::size_t, sizeof...(X)>;
#define XTENSOR_FIXED_SHAPE_CONSTEXPR inline
#else
        using cast_type = const_array<std::size_t, sizeof...(X)>;
#define XTENSOR_FIXED_SHAPE_CONSTEXPR constexpr
#endif
        using value_type = std::size_t;
        using size_type = std::size_t;
        using const_iterator = typename cast_type::const_iterator;

        static constexpr std::size_t size()
        {
            return sizeof...(X);
        }

        template <std::size_t idx>
        static constexpr auto get()
        {
            using tmp_cast_type = std::array<std::size_t, sizeof...(X)>;
            return std::get<idx>(tmp_cast_type{X...});
        }

        XTENSOR_FIXED_SHAPE_CONSTEXPR operator cast_type() const
        {
            return cast_type({X...});
        }

        XTENSOR_FIXED_SHAPE_CONSTEXPR auto begin() const
        {
            return m_array.begin();
        }

        XTENSOR_FIXED_SHAPE_CONSTEXPR auto end() const
        {
            return m_array.end();
        }

        auto rbegin() const
        {
            return m_array.rbegin();
        }

        auto rend() const
        {
            return m_array.rend();
        }

        XTENSOR_FIXED_SHAPE_CONSTEXPR auto cbegin() const
        {
            return m_array.cbegin();
        }

        XTENSOR_FIXED_SHAPE_CONSTEXPR auto cend() const
        {
            return m_array.cend();
        }

        XTENSOR_FIXED_SHAPE_CONSTEXPR std::size_t operator[](std::size_t idx) const
        {
            return m_array[idx];
        }

        XTENSOR_FIXED_SHAPE_CONSTEXPR bool empty() const
        {
            return sizeof...(X) == 0;
        }

    private:

        XTENSOR_CONSTEXPR_ENHANCED_STATIC cast_type m_array = cast_type({X...});
    };

#ifdef XTENSOR_HAS_CONSTEXPR_ENHANCED
    template <std::size_t... X>
    constexpr typename fixed_shape<X...>::cast_type fixed_shape<X...>::m_array;
#endif

#undef XTENSOR_FIXED_SHAPE_CONSTEXPR

    template <class E, std::ptrdiff_t Start, std::ptrdiff_t End = -1>
    class sequence_view
    {
    public:

        using value_type = typename E::value_type;
        using reference = typename E::reference;
        using const_reference = typename E::const_reference;
        using pointer = typename E::pointer;
        using const_pointer = typename E::const_pointer;

        using size_type = typename E::size_type;
        using difference_type = typename E::difference_type;

        using iterator = typename E::iterator;
        using const_iterator = typename E::const_iterator;
        using reverse_iterator = typename E::reverse_iterator;
        using const_reverse_iterator = typename E::const_reverse_iterator;

        explicit sequence_view(const E& container);

        template <std::ptrdiff_t OS, std::ptrdiff_t OE>
        explicit sequence_view(const sequence_view<E, OS, OE>& other);

        template <class T, class R = decltype(std::declval<T>().begin())>
        operator T() const;

        bool empty() const;
        size_type size() const;
        const_reference operator[](std::size_t idx) const;

        const_iterator end() const;
        const_iterator begin() const;
        const_iterator cend() const;
        const_iterator cbegin() const;

        const_reverse_iterator rend() const;
        const_reverse_iterator rbegin() const;
        const_reverse_iterator crend() const;
        const_reverse_iterator crbegin() const;

        const_reference front() const;
        const_reference back() const;

        const E& storage() const;

    private:

        const E& m_sequence;
    };

    template <class E, std::ptrdiff_t Start, std::ptrdiff_t End>
    sequence_view<E, Start, End>::sequence_view(const E& container)
        : m_sequence(container)
    {
    }

    template <class E, std::ptrdiff_t Start, std::ptrdiff_t End>
    template <std::ptrdiff_t OS, std::ptrdiff_t OE>
    sequence_view<E, Start, End>::sequence_view(const sequence_view<E, OS, OE>& other)
        : m_sequence(other.storage())
    {
    }

    template <class E, std::ptrdiff_t Start, std::ptrdiff_t End>
    template <class T, class R>
    sequence_view<E, Start, End>::operator T() const
    {
        T ret = xtl::make_sequence<T>(this->size());
        std::copy(this->cbegin(), this->cend(), ret.begin());
        return ret;
    }

    template <class E, std::ptrdiff_t Start, std::ptrdiff_t End>
    bool sequence_view<E, Start, End>::empty() const
    {
        return size() == size_type(0);
    }

    template <class E, std::ptrdiff_t Start, std::ptrdiff_t End>
    auto sequence_view<E, Start, End>::size() const -> size_type
    {
        if (End == -1)
        {
            return m_sequence.size() - static_cast<size_type>(Start);
        }
        else
        {
            return static_cast<size_type>(End - Start);
        }
    }

    template <class E, std::ptrdiff_t Start, std::ptrdiff_t End>
    auto sequence_view<E, Start, End>::operator[](std::size_t idx) const -> const_reference
    {
        return m_sequence[idx + static_cast<std::size_t>(Start)];
    }

    template <class E, std::ptrdiff_t Start, std::ptrdiff_t End>
    auto sequence_view<E, Start, End>::end() const -> const_iterator
    {
        if (End != -1)
        {
            return m_sequence.begin() + End;
        }
        else
        {
            return m_sequence.end();
        }
    }

    template <class E, std::ptrdiff_t Start, std::ptrdiff_t End>
    auto sequence_view<E, Start, End>::begin() const -> const_iterator
    {
        return m_sequence.begin() + Start;
    }

    template <class E, std::ptrdiff_t Start, std::ptrdiff_t End>
    auto sequence_view<E, Start, End>::cend() const -> const_iterator
    {
        return end();
    }

    template <class E, std::ptrdiff_t Start, std::ptrdiff_t End>
    auto sequence_view<E, Start, End>::cbegin() const -> const_iterator
    {
        return begin();
    }

    template <class E, std::ptrdiff_t Start, std::ptrdiff_t End>
    auto sequence_view<E, Start, End>::rend() const -> const_reverse_iterator
    {
        return const_reverse_iterator(begin());
    }

    template <class E, std::ptrdiff_t Start, std::ptrdiff_t End>
    auto sequence_view<E, Start, End>::rbegin() const -> const_reverse_iterator
    {
        return const_reverse_iterator(end());
    }

    template <class E, std::ptrdiff_t Start, std::ptrdiff_t End>
    auto sequence_view<E, Start, End>::crend() const -> const_reverse_iterator
    {
        return rend();
    }

    template <class E, std::ptrdiff_t Start, std::ptrdiff_t End>
    auto sequence_view<E, Start, End>::crbegin() const -> const_reverse_iterator
    {
        return rbegin();
    }

    template <class E, std::ptrdiff_t Start, std::ptrdiff_t End>
    auto sequence_view<E, Start, End>::front() const -> const_reference
    {
        return *(m_sequence.begin() + Start);
    }

    template <class E, std::ptrdiff_t Start, std::ptrdiff_t End>
    auto sequence_view<E, Start, End>::back() const -> const_reference
    {
        if (End == -1)
        {
            return m_sequence.back();
        }
        else
        {
            return m_sequence[static_cast<std::size_t>(End - 1)];
        }
    }

    template <class E, std::ptrdiff_t Start, std::ptrdiff_t End>
    const E& sequence_view<E, Start, End>::storage() const
    {
        return m_sequence;
    }

    template <class T, std::ptrdiff_t TB, std::ptrdiff_t TE>
    inline bool operator==(const sequence_view<T, TB, TE>& lhs, const sequence_view<T, TB, TE>& rhs)
    {
        return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    template <class T, std::ptrdiff_t TB, std::ptrdiff_t TE>
    inline bool operator!=(const sequence_view<T, TB, TE>& lhs, const sequence_view<T, TB, TE>& rhs)
    {
        return !(lhs == rhs);
    }
}

/******************************
 * std::tuple_size extensions *
 ******************************/

// The C++ standard defines tuple_size as a class, however
// G++ 8 C++ library does define it as a struct hence we get
// clang warnings here

// Do not remove space between "#" and "pragma". This is required for CRAN checks.
// clang-format off
#if defined(__clang__)
 # pragma clang diagnostic push
 # pragma clang diagnostic ignored "-Wmismatched-tags"
#endif
// clang-format on

namespace std
{
    template <class T, std::size_t N>
    class tuple_size<xt::const_array<T, N>> : public integral_constant<std::size_t, N>
    {
    };

    template <std::size_t... N>
    class tuple_size<xt::fixed_shape<N...>> : public integral_constant<std::size_t, sizeof...(N)>
    {
    };

    template <class T, std::ptrdiff_t Start, std::ptrdiff_t End>
    class tuple_size<xt::sequence_view<T, Start, End>>
        : public integral_constant<std::size_t, std::size_t(End - Start)>
    {
    };

    // Undefine tuple size for not-known sequence view size
    template <class T, std::ptrdiff_t Start>
    class tuple_size<xt::sequence_view<T, Start, -1>>;
}

// Do not remove space between "#" and "pragma". This is required for CRAN checks.
// clang-format off
#if defined(__clang__)
 # pragma clang diagnostic pop
#endif
// clang-format on

#undef XTENSOR_CONST

#endif
