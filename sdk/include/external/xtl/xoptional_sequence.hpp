/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTL_OPTIONAL_SEQUENCE_HPP
#define XTL_OPTIONAL_SEQUENCE_HPP

#include <array>
#include <bitset>
#include <cstddef>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "xdynamic_bitset.hpp"
#include "xiterator_base.hpp"
#include "xoptional.hpp"
#include "xsequence.hpp"

namespace xtl
{
    /**************************************
     * Optimized 1-D xoptional containers *
     **************************************/

    template <class ITV, class ITB>
    class xoptional_iterator;

    template <class BC, class FC>
    class xoptional_sequence
    {
    public:

        // Internal typedefs

        using base_container_type = BC;
        using base_value_type = typename base_container_type::value_type;
        using base_reference = typename base_container_type::reference;
        using base_const_reference = typename base_container_type::const_reference;

        using flag_container_type = FC;
        using flag_type = typename flag_container_type::value_type;
        using flag_reference = typename flag_container_type::reference;
        using flag_const_reference = typename flag_container_type::const_reference;

        // Container typedefs
        using value_type = xoptional<base_value_type, flag_type>;
        using reference = xoptional<base_reference, flag_reference>;
        using const_reference = xoptional<base_const_reference, flag_const_reference>;
        using pointer = xclosure_pointer<reference>;
        using const_pointer = xclosure_pointer<const_reference>;

        // Other typedefs
        using size_type = typename base_container_type::size_type;
        using difference_type = typename base_container_type::difference_type;
        using iterator = xoptional_iterator<typename base_container_type::iterator,
                                            typename flag_container_type::iterator>;
        using const_iterator = xoptional_iterator<typename base_container_type::const_iterator,
                                                  typename flag_container_type::const_iterator>;

        using reverse_iterator = xoptional_iterator<typename base_container_type::reverse_iterator,
                                                    typename flag_container_type::reverse_iterator>;
        using const_reverse_iterator = xoptional_iterator<typename base_container_type::const_reverse_iterator,
                                                          typename flag_container_type::const_reverse_iterator>;

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

        base_container_type value() && noexcept;
        base_container_type& value() & noexcept;
        const base_container_type& value() const & noexcept;

        flag_container_type has_value() && noexcept;
        flag_container_type& has_value() & noexcept;
        const flag_container_type& has_value() const & noexcept;

    protected:

        xoptional_sequence() = default;
        xoptional_sequence(size_type s, const base_value_type& v);
        template <class CTO, class CBO>
        xoptional_sequence(size_type s, const xoptional<CTO, CBO>& v);

        ~xoptional_sequence() = default;

        xoptional_sequence(const xoptional_sequence&) = default;
        xoptional_sequence& operator=(const xoptional_sequence&) = default;

        xoptional_sequence(xoptional_sequence&&) = default;
        xoptional_sequence& operator=(xoptional_sequence&&) = default;

        base_container_type m_values;
        flag_container_type m_flags;
    };

    template <class BC, class FC>
    bool operator==(const xoptional_sequence<BC, FC>& lhs, const xoptional_sequence<BC, FC>& rhs);

    template <class BC, class FC>
    bool operator!=(const xoptional_sequence<BC, FC>& lhs, const xoptional_sequence<BC, FC>& rhs);

    template <class BC, class FC>
    bool operator<(const xoptional_sequence<BC, FC>& lhs, const xoptional_sequence<BC, FC>& rhs);

    template <class BC, class FC>
    bool operator<=(const xoptional_sequence<BC, FC>& lhs, const xoptional_sequence<BC, FC>& rhs);

    template <class BC, class FC>
    bool operator>(const xoptional_sequence<BC, FC>& lhs, const xoptional_sequence<BC, FC>& rhs);

    template <class BC, class FC>
    bool operator>=(const xoptional_sequence<BC, FC>& lhs, const xoptional_sequence<BC, FC>& rhs);

    /********************************
     * xoptional_array declarations *
     ********************************/

    // There is no value_type in std::bitset ...
    template <class T, std::size_t I, class BC = xdynamic_bitset<std::size_t>>
    class xoptional_array : public xoptional_sequence<std::array<T, I>, BC>
    {
    public:

        using self_type = xoptional_array;
        using base_container_type = std::array<T, I>;
        using flag_container_type = BC;
        using base_type = xoptional_sequence<base_container_type, flag_container_type>;
        using base_value_type = typename base_type::base_value_type;
        using size_type = typename base_type::size_type;

        xoptional_array() = default;
        xoptional_array(size_type s, const base_value_type& v);

        template <class CTO, class CBO>
        xoptional_array(size_type s, const xoptional<CTO, CBO>& v);
    };

    /********************
     * xoptional_vector *
     ********************/

    template <class T, class A = std::allocator<T>, class BC = xdynamic_bitset<std::size_t>>
    class xoptional_vector : public xoptional_sequence<std::vector<T, A>, BC>
    {
    public:

        using self_type = xoptional_vector;
        using base_container_type = std::vector<T, A>;
        using flag_container_type = BC;
        using base_type = xoptional_sequence<base_container_type, flag_container_type>;
        using base_value_type = typename base_type::base_value_type;
        using allocator_type = A;

        using value_type = typename base_type::value_type;
        using size_type = typename base_type::size_type;
        using difference_type = typename base_type::difference_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;

        using iterator = typename base_type::iterator;
        using const_iterator = typename base_type::const_iterator;
        using reverse_iterator = typename base_type::reverse_iterator;
        using const_reverse_iterator = typename base_type::const_reverse_iterator;

        xoptional_vector() = default;
        xoptional_vector(size_type, const base_value_type&);

        template <class CTO, class CBO>
        xoptional_vector(size_type, const xoptional<CTO, CBO>&);

        void resize(size_type);
        void resize(size_type, const base_value_type&);
        template <class CTO, class CBO>
        void resize(size_type, const xoptional<CTO, CBO>&);
    };

    /**********************************
     * xoptional_iterator declaration *
     **********************************/

    template <class ITV, class ITB>
    struct xoptional_iterator_traits
    {
        using iterator_type = xoptional_iterator<ITV, ITB>;
        using value_type = xoptional<typename ITV::value_type, typename ITB::value_type>;
        using reference = xoptional<typename ITV::reference, typename ITB::reference>;
        using pointer = xclosure_pointer<reference>;
        using difference_type = typename ITV::difference_type;
    };

    template <class ITV, class ITB>
    class xoptional_iterator : public xrandom_access_iterator_base2<xoptional_iterator_traits<ITV, ITB>>
    {
    public:

        using self_type = xoptional_iterator<ITV, ITB>;
        using base_type = xrandom_access_iterator_base2<xoptional_iterator_traits<ITV, ITB>>;

        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using pointer = typename base_type::pointer;
        using difference_type = typename base_type::difference_type;

        xoptional_iterator() = default;
        xoptional_iterator(ITV itv, ITB itb);

        self_type& operator++();
        self_type& operator--();

        self_type& operator+=(difference_type n);
        self_type& operator-=(difference_type n);

        difference_type operator-(const self_type& rhs) const;

        reference operator*() const;
        pointer operator->() const;

        bool operator==(const self_type& rhs) const;
        bool operator<(const self_type& rhs) const;

    private:

        ITV m_itv;
        ITB m_itb;
    };

    /*************************************
     * xoptional_sequence implementation *
     *************************************/

    template <class BC, class FC>
    inline xoptional_sequence<BC, FC>::xoptional_sequence(size_type s, const base_value_type& v)
        : m_values(make_sequence<base_container_type>(s, v)),
          m_flags(make_sequence<flag_container_type>(s, true))
    {
    }

    template <class BC, class FC>
    template <class CTO, class CBO>
    inline xoptional_sequence<BC, FC>::xoptional_sequence(size_type s, const xoptional<CTO, CBO>& v)
        : m_values(make_sequence<base_container_type>(s, v.value())), m_flags(make_sequence<flag_container_type>(s, v.has_value()))
    {
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::empty() const noexcept -> bool
    {
        return m_values.empty();
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::size() const noexcept -> size_type
    {
        return m_values.size();
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::max_size() const noexcept -> size_type
    {
        return m_values.max_size();
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::at(size_type i) -> reference
    {
        return reference(m_values.at(i), m_flags.at(i));
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::at(size_type i) const -> const_reference
    {
        return const_reference(m_values.at(i), m_flags.at(i));
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::operator[](size_type i) -> reference
    {
        return reference(m_values[i], m_flags[i]);
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::operator[](size_type i) const -> const_reference
    {
        return const_reference(m_values[i], m_flags[i]);
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::front() -> reference
    {
        return reference(m_values.front(), m_flags.front());
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::front() const -> const_reference
    {
        return const_reference(m_values.front(), m_flags.front());
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::back() -> reference
    {
        return reference(m_values.back(), m_flags.back());
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::back() const -> const_reference
    {
        return const_reference(m_values.back(), m_flags.back());
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::begin() noexcept -> iterator
    {
        return iterator(m_values.begin(), m_flags.begin());
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::end() noexcept -> iterator
    {
        return iterator(m_values.end(), m_flags.end());
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::begin() const noexcept -> const_iterator
    {
        return cbegin();
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::end() const noexcept -> const_iterator
    {
        return cend();
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::cbegin() const noexcept -> const_iterator
    {
        return const_iterator(m_values.cbegin(), m_flags.cbegin());
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::cend() const noexcept -> const_iterator
    {
        return const_iterator(m_values.cend(), m_flags.cend());
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::rbegin() noexcept -> reverse_iterator
    {
        return reverse_iterator(m_values.rbegin(), m_flags.rbegin());
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::rend() noexcept -> reverse_iterator
    {
        return reverse_iterator(m_values.rend(), m_flags.rend());
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::rbegin() const noexcept -> const_reverse_iterator
    {
        return crbegin();
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::rend() const noexcept -> const_reverse_iterator
    {
        return crend();
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::crbegin() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator(m_values.crbegin(), m_flags.crbegin());
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::crend() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator(m_values.crend(), m_flags.crend());
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::value() && noexcept -> base_container_type
    {
        return m_values;
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::value() & noexcept -> base_container_type&
    {
        return m_values;
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::value() const & noexcept -> const base_container_type&
    {
        return m_values;
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::has_value() && noexcept-> flag_container_type
    {
        return m_flags;
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::has_value() & noexcept -> flag_container_type&
    {
        return m_flags;
    }

    template <class BC, class FC>
    inline auto xoptional_sequence<BC, FC>::has_value() const & noexcept -> const flag_container_type&
    {
        return m_flags;
    }

    template <class BC, class FC>
    inline bool operator==(const xoptional_sequence<BC, FC>& lhs, const xoptional_sequence<BC, FC>& rhs)
    {
        return lhs.value() == rhs.value() && lhs.has_value() == rhs.has_value();
    }

    template <class BC, class FC>
    inline bool operator!=(const xoptional_sequence<BC, FC>& lhs, const xoptional_sequence<BC, FC>& rhs)
    {
        return !(lhs == rhs);
    }

    template <class BC, class FC>
    inline bool operator<(const xoptional_sequence<BC, FC>& lhs, const xoptional_sequence<BC, FC>& rhs)
    {
        return lhs.value() < rhs.value() && lhs.has_value() == rhs.has_value();
    }

    template <class BC, class FC>
    inline bool operator<=(const xoptional_sequence<BC, FC>& lhs, const xoptional_sequence<BC, FC>& rhs)
    {
        return lhs.value() <= rhs.value() && lhs.has_value() == rhs.has_value();
    }

    template <class BC, class FC>
    inline bool operator>(const xoptional_sequence<BC, FC>& lhs, const xoptional_sequence<BC, FC>& rhs)
    {
        return lhs.value() > rhs.value() && lhs.has_value() == rhs.has_value();
    }

    template <class BC, class FC>
    inline bool operator>=(const xoptional_sequence<BC, FC>& lhs, const xoptional_sequence<BC, FC>& rhs)
    {
        return lhs.value() >= rhs.value() && lhs.has_value() == rhs.has_value();
    }

    /**********************************
     * xoptional_array implementation *
     **********************************/

    template <class T, std::size_t I, class BC>
    xoptional_array<T, I, BC>::xoptional_array(size_type s, const base_value_type& v)
        : base_type(s, v)
    {
    }

    template <class T, std::size_t I, class BC>
    template <class CTO, class CBO>
    xoptional_array<T, I, BC>::xoptional_array(size_type s, const xoptional<CTO, CBO>& v)
        : base_type(s, v)
    {
    }

    /*******************************************************
     * xoptional_array and xoptional_vector implementation *
     *******************************************************/

    template <class T, class A, class BC>
    xoptional_vector<T, A, BC>::xoptional_vector(size_type s, const base_value_type& v)
        : base_type(s, v)
    {
    }

    template <class T, class A, class BC>
    template <class CTO, class CBO>
    xoptional_vector<T, A, BC>::xoptional_vector(size_type s, const xoptional<CTO, CBO>& v)
        : base_type(s, v)
    {
    }

    template <class T, class A, class BC>
    void xoptional_vector<T, A, BC>::resize(size_type s)
    {
        // Default to missing
        this->m_values.resize(s);
        this->m_flags.resize(s, false);
    }

    template <class T, class A, class BC>
    void xoptional_vector<T, A, BC>::resize(size_type s, const base_value_type& v)
    {
        this->m_values.resize(s, v);
        this->m_flags.resize(s, true);
    }

    template <class T, class A, class BC>
    template <class CTO, class CBO>
    void xoptional_vector<T, A, BC>::resize(size_type s, const xoptional<CTO, CBO>& v)
    {
        this->m_values.resize(s, v.value());
        this->m_flags.resize(s, v.has_value());
    }

    /*************************************
     * xoptional_iterator implementation *
     *************************************/

    template <class ITV, class ITB>
    xoptional_iterator<ITV, ITB>::xoptional_iterator(ITV itv, ITB itb)
        : m_itv(itv), m_itb(itb)
    {
    }

    template <class ITV, class ITB>
    auto xoptional_iterator<ITV, ITB>::operator++() -> self_type&
    {
        ++m_itv;
        ++m_itb;
        return *this;
    }

    template <class ITV, class ITB>
    auto xoptional_iterator<ITV, ITB>::operator--() -> self_type&
    {
        --m_itv;
        --m_itb;
        return *this;
    }

    template <class ITV, class ITB>
    auto xoptional_iterator<ITV, ITB>::operator+=(difference_type n) -> self_type&
    {
        m_itv += n;
        m_itb += n;
        return *this;
    }

    template <class ITV, class ITB>
    auto xoptional_iterator<ITV, ITB>::operator-=(difference_type n) -> self_type&
    {
        m_itv -= n;
        m_itb -= n;
        return *this;
    }

    template <class ITV, class ITB>
    auto xoptional_iterator<ITV, ITB>::operator-(const self_type& rhs) const -> difference_type
    {
        return m_itv - rhs.m_itv;
    }

    template <class ITV, class ITB>
    auto xoptional_iterator<ITV, ITB>::operator*() const -> reference
    {
        return reference(*m_itv, *m_itb);
    }

    template <class ITV, class ITB>
    auto xoptional_iterator<ITV, ITB>::operator-> () const -> pointer
    {
        return pointer(operator*());
    }

    template <class ITV, class ITB>
    bool xoptional_iterator<ITV, ITB>::operator==(const self_type& rhs) const
    {
        return m_itv == rhs.m_itv && m_itb == rhs.m_itb;
    }

    template <class ITV, class ITB>
    bool xoptional_iterator<ITV, ITB>::operator<(const self_type& rhs) const
    {
        return m_itv < rhs.m_itv && m_itb < rhs.m_itb;
    }
}

#endif
