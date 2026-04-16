/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTL_XCOMPLEX_SEQUENCE_HPP
#define XTL_XCOMPLEX_SEQUENCE_HPP

#include <array>
#include <vector>
#include <algorithm>

#include "xclosure.hpp"
#include "xcomplex.hpp"
#include "xiterator_base.hpp"
#include "xsequence.hpp"

namespace xtl
{
    /************************************
     * Optimized 1-D xcomplex container *
     ************************************/

    template <class IT, bool ieee_compliant>
    class xcomplex_iterator;

    template <class C, bool ieee_compliant>
    class xcomplex_sequence
    {
    public:

        using container_type = C;
        using cvt = typename C::value_type;

        using value_type = xcomplex<cvt, cvt, ieee_compliant>;
        using reference = xcomplex<cvt&, cvt&, ieee_compliant>;
        using const_reference = xcomplex<const cvt&, const cvt&, ieee_compliant>;
        using pointer = xclosure_pointer<reference>;
        using const_pointer = xclosure_pointer<const_reference>;
        using size_type = typename container_type::size_type;
        using difference_type = typename container_type::difference_type;

        using iterator = xcomplex_iterator<typename C::iterator, ieee_compliant>;
        using const_iterator = xcomplex_iterator<typename C::const_iterator, ieee_compliant>;
        using reverse_iterator = xcomplex_iterator<typename C::reverse_iterator, ieee_compliant>;
        using const_reverse_iterator = xcomplex_iterator<typename C::const_reverse_iterator, ieee_compliant>;

        bool empty() const noexcept;
        size_type size() const noexcept;
        size_type max_size() const noexcept;

        reference at(size_type i);
        const_reference at(size_type i) const;

        reference operator[](size_type i);
        const_reference operator[](size_type i) const;

        reference front();
        const_reference front() const;

        reference back();
        const_reference back() const;

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

        container_type real() && noexcept;
        container_type& real() & noexcept;
        const container_type& real() const & noexcept;

        container_type imag() && noexcept;
        container_type& imag() & noexcept;
        const container_type& imag() const & noexcept;

    protected:

        xcomplex_sequence() = default;
        xcomplex_sequence(size_type s);
        xcomplex_sequence(size_type s, const value_type& v);
        template <class TR, class TC, bool B>
        xcomplex_sequence(size_type s, const xcomplex<TR, TC, B>& v);
        xcomplex_sequence(std::initializer_list<value_type> init);

        ~xcomplex_sequence() = default;

        xcomplex_sequence(const xcomplex_sequence&) = default;
        xcomplex_sequence& operator=(const xcomplex_sequence&) = default;

        xcomplex_sequence(xcomplex_sequence&&) = default;
        xcomplex_sequence& operator=(xcomplex_sequence&&) = default;

        container_type m_real;
        container_type m_imag;
    };

    template <class C, bool B>
    bool operator==(const xcomplex_sequence<C, B>& lhs, const xcomplex_sequence<C, B>& rhs);

    template <class C, bool B>
    bool operator!=(const xcomplex_sequence<C, B>& lhs, const xcomplex_sequence<C, B>& rhs);

    /******************
     * xcomplex_array *
     ******************/

    template <class T, std::size_t N, bool ieee_compliant = false>
    class xcomplex_array : public xcomplex_sequence<std::array<T, N>, ieee_compliant>
    {
    public:

        using base_type = xcomplex_sequence<std::array<T, N>, ieee_compliant>;
        using value_type = typename base_type::value_type;
        using size_type = typename base_type::size_type;

        xcomplex_array() = default;
        xcomplex_array(size_type s);
        xcomplex_array(size_type s, const value_type& v);

        template <class TR, class TI, bool B>
        xcomplex_array(size_type s, const xcomplex<TR, TI, B>& v);
    };

    /*******************
     * xcomplex_vector *
     *******************/

    template <class T, bool ieee_compliant = false, class A = std::allocator<T>>
    class xcomplex_vector : public xcomplex_sequence<std::vector<T, A>, ieee_compliant>
    {
    public:

        using base_type = xcomplex_sequence<std::vector<T, A>, ieee_compliant>;
        using value_type = typename base_type::value_type;
        using size_type = typename base_type::size_type;

        xcomplex_vector() = default;
        xcomplex_vector(size_type s);
        xcomplex_vector(size_type s, const value_type& v);
        xcomplex_vector(std::initializer_list<value_type> init);

        template <class TR, class TI, bool B>
        xcomplex_vector(size_type s, const xcomplex<TR, TI, B>& v);

        void resize(size_type);
        void resize(size_type, const value_type&);
        template <class TR, class TI, bool B>
        void resize(size_type s, const xcomplex<TR, TI, B>& v);
    };

    /*********************
     * xcomplex_iterator *
     *********************/

    template <class IT, bool ieee_compliant>
    struct xcomplex_iterator_traits
    {
        using iterator_type = xcomplex_iterator<IT, ieee_compliant>;
        using value_type = xcomplex<typename IT::value_type, typename IT::value_type, ieee_compliant>;
        using reference = xcomplex<typename IT::reference, typename IT::reference, ieee_compliant>;
        using pointer = xclosure_pointer<reference>;
        using difference_type = typename IT::difference_type;
    };

    template <class IT, bool B>
    class xcomplex_iterator : public xrandom_access_iterator_base2<xcomplex_iterator_traits<IT, B>>
    {
    public:

        using self_type = xcomplex_iterator<IT, B>;
        using base_type = xrandom_access_iterator_base2<xcomplex_iterator_traits<IT, B>>;

        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using pointer = typename base_type::pointer;
        using difference_type = typename base_type::difference_type;

        xcomplex_iterator() = default;
        xcomplex_iterator(IT it_real, IT it_imag);

        self_type& operator++();
        self_type& operator--();

        self_type& operator+=(difference_type n);
        self_type& operator-=(difference_type n);

        difference_type operator-(const self_type& rhs) const;

        reference operator*() const;
        pointer operator->() const;

        bool operator==(const self_type& rhs) const;

    private:

        IT m_it_real;
        IT m_it_imag;
    };

    /************************************
     * xcomplex_sequence implementation *
     ************************************/

    template <class C, bool B>
    inline xcomplex_sequence<C, B>::xcomplex_sequence(size_type s)
        : m_real(make_sequence<container_type>(s)),
          m_imag(make_sequence<container_type>(s))
    {
    }

    template <class C, bool B>
    inline xcomplex_sequence<C, B>::xcomplex_sequence(size_type s, const value_type& v)
        : m_real(make_sequence<container_type>(s, v.real())),
          m_imag(make_sequence<container_type>(s, v.imag()))
    {
    }

    template <class C, bool B>
    template <class TR, class TC, bool B2>
    inline xcomplex_sequence<C, B>::xcomplex_sequence(size_type s, const xcomplex<TR, TC, B2>& v)
        : m_real(make_sequence<container_type>(s, v.real())),
          m_imag(make_sequence<container_type>(s, v.imag()))
    {
    }

    template <class C, bool B>
    inline xcomplex_sequence<C, B>::xcomplex_sequence(std::initializer_list<value_type> init)
        : m_real(make_sequence<container_type>(init.size())),
          m_imag(make_sequence<container_type>(init.size()))
    {
        std::transform(init.begin(), init.end(), m_real.begin(), [](const auto& v) { return v.real(); });
        std::transform(init.begin(), init.end(), m_imag.begin(), [](const auto& v) { return v.imag(); });
    }

    template <class C, bool B>
    inline bool xcomplex_sequence<C, B>::empty() const noexcept
    {
        return m_real.empty();
    }

    template <class C, bool B>
    inline auto xcomplex_sequence<C, B>::size() const noexcept -> size_type
    {
        return m_real.size();
    }

    template <class C, bool B>
    inline auto xcomplex_sequence<C, B>::max_size() const noexcept -> size_type
    {
        return m_real.max_size();
    }

    template <class C, bool B>
    inline auto xcomplex_sequence<C, B>::at(size_type i) -> reference
    {
        return reference(m_real.at(i), m_imag.at(i));
    }

    template <class C, bool B>
    inline auto xcomplex_sequence<C, B>::at(size_type i) const -> const_reference
    {
        return const_reference(m_real.at(i), m_imag.at(i));
    }

    template <class C, bool B>
    inline auto xcomplex_sequence<C, B>::operator[](size_type i) -> reference
    {
        return reference(m_real[i], m_imag[i]);
    }

    template <class C, bool B>
    inline auto xcomplex_sequence<C, B>::operator[](size_type i) const -> const_reference
    {
        return const_reference(m_real[i], m_imag[i]);
    }

    template <class C, bool B>
    inline auto xcomplex_sequence<C, B>::front() -> reference
    {
        return reference(m_real.front(), m_imag.front());
    }

    template <class C, bool B>
    inline auto xcomplex_sequence<C, B>::front() const -> const_reference
    {
        return const_reference(m_real.front(), m_imag.front());
    }

    template <class C, bool B>
    inline auto xcomplex_sequence<C, B>::back() -> reference
    {
        return reference(m_real.back(), m_imag.back());
    }

    template <class C, bool B>
    inline auto xcomplex_sequence<C, B>::back() const -> const_reference
    {
        return const_reference(m_real.back(), m_imag.back());
    }

    template <class C, bool B>
    inline auto xcomplex_sequence<C, B>::begin() noexcept -> iterator
    {
        return iterator(m_real.begin(), m_imag.begin());
    }

    template <class C, bool B>
    inline auto xcomplex_sequence<C, B>::end() noexcept -> iterator
    {
        return iterator(m_real.end(), m_imag.end());
    }

    template <class C, bool B>
    inline auto xcomplex_sequence<C, B>::begin() const noexcept -> const_iterator
    {
        return cbegin();
    }

    template <class C, bool B>
    inline auto xcomplex_sequence<C, B>::end() const noexcept -> const_iterator
    {
        return cend();
    }
    
    template <class C, bool B>
    inline auto xcomplex_sequence<C, B>::cbegin() const noexcept -> const_iterator
    {
        return const_iterator(m_real.cbegin(), m_imag.cbegin());
    }
    
    template <class C, bool B>
    inline auto xcomplex_sequence<C, B>::cend() const noexcept -> const_iterator
    {
        return const_iterator(m_real.cend(), m_imag.cend());
    }

    template <class C, bool B>
    inline auto xcomplex_sequence<C, B>::rbegin() noexcept -> reverse_iterator
    {
        return reverse_iterator(m_real.rbegin(), m_imag.rbegin());
    }
    
    template <class C, bool B>
    inline auto xcomplex_sequence<C, B>::rend() noexcept -> reverse_iterator
    {
        return reverse_iterator(m_real.rend(), m_imag.rend());
    }

    template <class C, bool B>
    inline auto xcomplex_sequence<C, B>::rbegin() const noexcept -> const_reverse_iterator
    {
        return crbegin();
    }

    template <class C, bool B>
    inline auto xcomplex_sequence<C, B>::rend() const noexcept -> const_reverse_iterator
    {
        return crend();
    }

    template <class C, bool B>
    inline auto xcomplex_sequence<C, B>::crbegin() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator(m_real.crbegin(), m_imag.crbegin());
    }

    template <class C, bool B>
    inline auto xcomplex_sequence<C, B>::crend() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator(m_real.crend(), m_imag.crend());
    }

    template <class C, bool B>
    inline auto xcomplex_sequence<C, B>::real() && noexcept -> container_type
    {
        return m_real;
    }

    template <class C, bool B>
    inline auto xcomplex_sequence<C, B>::real() & noexcept -> container_type&
    {
        return m_real;
    }

    template <class C, bool B>
    inline auto xcomplex_sequence<C, B>::real() const & noexcept -> const container_type&
    {
        return m_real;
    }

    template <class C, bool B>
    inline auto xcomplex_sequence<C, B>::imag() && noexcept -> container_type
    {
        return m_imag;
    }

    template <class C, bool B>
    inline auto xcomplex_sequence<C, B>::imag() & noexcept -> container_type&
    {
        return m_imag;
    }

    template <class C, bool B>
    inline auto xcomplex_sequence<C, B>::imag() const & noexcept -> const container_type&
    {
        return m_imag;
    }

    template <class C, bool B>
    inline bool operator==(const xcomplex_sequence<C, B>& lhs, const xcomplex_sequence<C, B>& rhs)
    {
        return lhs.real() == rhs.real() && lhs.imag() == rhs.imag();
    }

    template <class C, bool B>
    inline bool operator!=(const xcomplex_sequence<C, B>& lhs, const xcomplex_sequence<C, B>& rhs)
    {
        return !(lhs == rhs);
    }

    /*********************************
     * xcomplex_array implementation *
     *********************************/

    template <class T, std::size_t N, bool B>
    inline xcomplex_array<T, N, B>::xcomplex_array(size_type s)
        : base_type(s)
    {
    }
    
    template <class T, std::size_t N, bool B>
    inline xcomplex_array<T, N, B>::xcomplex_array(size_type s, const value_type& v)
        : base_type(s, v)
    {
    }

    template <class T, std::size_t N, bool B>
    template <class TR, class TI, bool B2>
    inline xcomplex_array<T, N, B>::xcomplex_array(size_type s, const xcomplex<TR, TI, B2>& v)
        : base_type(s, v)
    {
    }

    /**********************************
     * xcomplex_vector implementation *
     **********************************/

    template <class T, bool B, class A>
    inline xcomplex_vector<T, B, A>::xcomplex_vector(size_type s)
        : base_type(s)
    {
    }

    template <class T, bool B, class A>
    inline xcomplex_vector<T, B, A>::xcomplex_vector(size_type s, const value_type& v)
        : base_type(s, v)
    {
    }

    template <class T, bool B, class A>
    template <class TR, class TI, bool B2>
    inline xcomplex_vector<T, B, A>::xcomplex_vector(size_type s, const xcomplex<TR, TI, B2>& v)
        : base_type(s, v)
    {
    }

    template <class T, bool B, class A>
    inline xcomplex_vector<T, B, A>::xcomplex_vector(std::initializer_list<value_type> init)
        : base_type(init)
    {
    }

    template <class T, bool B, class A>
    void xcomplex_vector<T, B, A>::resize(size_type s)
    {
        this->m_real.resize(s);
        this->m_imag.resize(s);
    }

    template <class T, bool B, class A>
    void xcomplex_vector<T, B, A>::resize(size_type s, const value_type& v)
    {
        this->m_real.resize(s, v.real());
        this->m_imag.resize(s, v.imag());
    }

    template <class T, bool B, class A>
    template <class TR, class TI, bool B2>
    inline void xcomplex_vector<T, B, A>::resize(size_type s, const xcomplex<TR, TI, B2>& v)
    {
        this->m_real.resize(s, v.real());
        this->m_imag.resize(s, v.imag());
    }

    /************************************
     * xcomplex_iterator implementation *
     ************************************/

    template <class IT, bool B>
    inline xcomplex_iterator<IT, B>::xcomplex_iterator(IT it_real, IT it_imag)
        : m_it_real(it_real), m_it_imag(it_imag)
    {
    }

    template <class IT, bool B>
    inline auto xcomplex_iterator<IT, B>::operator++() -> self_type&
    {
        ++m_it_real;
        ++m_it_imag;
        return *this;
    }

    template <class IT, bool B>
    inline auto xcomplex_iterator<IT, B>::operator--() -> self_type&
    {
        --m_it_real;
        --m_it_imag;
        return *this;
    }

    template <class IT, bool B>
    inline auto xcomplex_iterator<IT, B>::operator+=(difference_type n) -> self_type&
    {
        m_it_real += n;
        m_it_imag += n;
        return *this;
    }

    template <class IT, bool B>
    inline auto xcomplex_iterator<IT, B>::operator-=(difference_type n) -> self_type&
    {
        m_it_real -= n;
        m_it_imag -= n;
        return *this;
    }
    
    template <class IT, bool B>
    inline auto xcomplex_iterator<IT, B>::operator-(const self_type& rhs) const -> difference_type
    {
        return m_it_real - rhs.m_it_real;
    }

    template <class IT, bool B>
    inline auto xcomplex_iterator<IT, B>::operator*() const -> reference
    {
        return reference(*m_it_real, *m_it_imag);
    }

    template <class IT, bool B>
    inline auto xcomplex_iterator<IT, B>::operator->() const -> pointer
    {
        return pointer(operator*());
    }

    template <class IT, bool B>
    inline bool xcomplex_iterator<IT, B>::operator==(const self_type& rhs) const
    {
        return m_it_real == rhs.m_it_real && m_it_imag == rhs.m_it_imag;
    }
}

#endif
