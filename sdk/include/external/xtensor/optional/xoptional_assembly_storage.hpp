/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XOPTIONAL_ASSEMBLY_STORAGE_HPP
#define XOPTIONAL_ASSEMBLY_STORAGE_HPP

#include "../core/xsemantic.hpp"
#include "../optional/xoptional.hpp"
#include "xtl/xiterator_base.hpp"

namespace xt
{
    template <class VE, class FE, bool is_const>
    class xoptional_assembly_linear_iterator;

    /******************************
     * xoptional_assembly_storage *
     ******************************/

    template <class VE, class FE>
    class xoptional_assembly_storage
    {
    public:

        using self_type = xoptional_assembly_storage<VE, FE>;

        using value_storage = std::remove_reference_t<VE>;
        using flag_storage = std::remove_reference_t<FE>;

        using value_type = xtl::xoptional<typename value_storage::value_type, typename flag_storage::value_type>;

        static constexpr bool is_val_const = std::is_const<value_storage>::value;
        static constexpr bool is_flag_const = std::is_const<flag_storage>::value;
        using val_reference = std::
            conditional_t<is_val_const, typename value_storage::const_reference, typename value_storage::reference>;
        using flag_reference = std::
            conditional_t<is_flag_const, typename flag_storage::const_reference, typename flag_storage::reference>;
        using reference = xtl::xoptional<val_reference, flag_reference>;
        using const_reference = xtl::xoptional<typename value_storage::const_reference, typename flag_storage::const_reference>;

        using pointer = xtl::xclosure_pointer<reference>;
        using const_pointer = xtl::xclosure_pointer<const_reference>;

        using size_type = typename value_storage::size_type;
        using difference_type = typename value_storage::difference_type;

        using iterator = xoptional_assembly_linear_iterator<VE, FE, false>;
        using const_iterator = xoptional_assembly_linear_iterator<VE, FE, true>;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        template <class VE1, class FE1>
        xoptional_assembly_storage(const VE1& value_stor, const FE1& flag_stor);

        template <class VE1, class FE1>
        xoptional_assembly_storage(VE1& value_stor, FE1& flag_stor);

        xoptional_assembly_storage(const xoptional_assembly_storage&);
        xoptional_assembly_storage& operator=(const xoptional_assembly_storage&);
        xoptional_assembly_storage(xoptional_assembly_storage&&);
        xoptional_assembly_storage& operator=(xoptional_assembly_storage&&);

        bool empty() const noexcept;
        size_type size() const noexcept;
        void resize(size_type size);

        reference operator[](size_type i);
        const_reference operator[](size_type i) const;

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

        void swap(self_type& rhs) noexcept;

        value_storage& value() noexcept;
        const value_storage& value() const noexcept;

        flag_storage& has_value() noexcept;
        const flag_storage& has_value() const noexcept;

    private:

        VE m_value;
        FE m_has_value;
    };

    template <class VE, class FE>
    bool
    operator==(const xoptional_assembly_storage<VE, FE>& lhs, const xoptional_assembly_storage<VE, FE>& rhs);

    template <class VE, class FE>
    bool
    operator!=(const xoptional_assembly_storage<VE, FE>& lhs, const xoptional_assembly_storage<VE, FE>& rhs);

    template <class VE, class FE>
    bool
    operator<(const xoptional_assembly_storage<VE, FE>& lhs, const xoptional_assembly_storage<VE, FE>& rhs);

    template <class VE, class FE>
    bool
    operator<=(const xoptional_assembly_storage<VE, FE>& lhs, const xoptional_assembly_storage<VE, FE>& rhs);

    template <class VE, class FE>
    bool
    operator>(const xoptional_assembly_storage<VE, FE>& lhs, const xoptional_assembly_storage<VE, FE>& rhs);

    template <class VE, class FE>
    bool
    operator>=(const xoptional_assembly_storage<VE, FE>& lhs, const xoptional_assembly_storage<VE, FE>& rhs);

    template <class VE, class FE>
    void swap(xoptional_assembly_storage<VE, FE>& lhs, xoptional_assembly_storage<VE, FE>& rhs) noexcept;

    /***************************************
     * xoptional_assembly_linear_iterator *
     ***************************************/

    template <class VE, class FE, bool is_const>
    class xoptional_assembly_linear_iterator;

    template <class VE, class FE, bool is_const>
    struct xoptional_assembly_linear_iterator_traits
    {
        using iterator_type = xoptional_assembly_linear_iterator<VE, FE, is_const>;
        using xoptional_assembly_storage_type = xoptional_assembly_storage<VE, FE>;
        using value_type = typename xoptional_assembly_storage_type::value_type;
        using reference = std::conditional_t<
            is_const,
            typename xoptional_assembly_storage_type::const_reference,
            typename xoptional_assembly_storage_type::reference>;
        using difference_type = typename xoptional_assembly_storage_type::difference_type;
        using pointer = std::conditional_t<
            is_const,
            typename xoptional_assembly_storage_type::const_pointer,
            typename xoptional_assembly_storage_type::pointer>;
    };

    template <class VE, class FE, bool is_const>
    class xoptional_assembly_linear_iterator
        : public xtl::xrandom_access_iterator_base2<xoptional_assembly_linear_iterator_traits<VE, FE, is_const>>
    {
    public:

        using self_type = xoptional_assembly_linear_iterator<VE, FE, is_const>;
        using base_type = xtl::xrandom_access_iterator_base2<
            xoptional_assembly_linear_iterator_traits<VE, FE, is_const>>;

        using xoptional_assembly_storage_type = xoptional_assembly_storage<VE, FE>;
        using value_iterator = std::conditional_t<
            is_const,
            typename xoptional_assembly_storage_type::value_storage::const_iterator,
            typename xoptional_assembly_storage_type::value_storage::iterator>;
        using flag_iterator = std::conditional_t<
            is_const,
            typename xoptional_assembly_storage_type::flag_storage::const_iterator,
            typename xoptional_assembly_storage_type::flag_storage::iterator>;

        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using pointer = typename base_type::pointer;
        using difference_type = typename base_type::difference_type;

        xoptional_assembly_linear_iterator(value_iterator value_it, flag_iterator flag_it);

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

        value_iterator m_value_it;
        flag_iterator m_flag_it;
    };

    /*********************************************
     * xoptional_assembly_storage implementation *
     *********************************************/

    template <class VE, class FE>
    template <class VE1, class FE1>
    inline xoptional_assembly_storage<VE, FE>::xoptional_assembly_storage(const VE1& value_stor, const FE1& flag_stor)
        : m_value(value_stor)
        , m_has_value(flag_stor)
    {
    }

    template <class VE, class FE>
    template <class VE1, class FE1>
    inline xoptional_assembly_storage<VE, FE>::xoptional_assembly_storage(VE1& value_stor, FE1& flag_stor)
        : m_value(value_stor)
        , m_has_value(flag_stor)
    {
    }

    template <class VE, class FE>
    inline xoptional_assembly_storage<VE, FE>::xoptional_assembly_storage(const self_type& rhs)
        : m_value(rhs.m_value)
        , m_has_value(rhs.m_has_value)
    {
    }

    template <class VE, class FE>
    inline auto xoptional_assembly_storage<VE, FE>::operator=(const self_type& rhs) -> self_type&
    {
        m_value = rhs.m_value;
        m_has_value = rhs.m_has_value;
        return *this;
    }

    template <class VE, class FE>
    inline xoptional_assembly_storage<VE, FE>::xoptional_assembly_storage(self_type&& rhs)
        : m_value(std::forward<VE>(rhs.m_value))
        , m_has_value(std::forward<FE>(rhs.m_has_value))
    {
    }

    template <class VE, class FE>
    inline auto xoptional_assembly_storage<VE, FE>::operator=(self_type&& rhs) -> self_type&
    {
        m_value = std::forward<VE>(rhs.m_value);
        m_has_value = std::forward<FE>(rhs.m_has_value);
        return *this;
    }

    template <class VE, class FE>
    inline bool xoptional_assembly_storage<VE, FE>::empty() const noexcept
    {
        return value().empty();
    }

    template <class VE, class FE>
    inline auto xoptional_assembly_storage<VE, FE>::size() const noexcept -> size_type
    {
        return value().size();
    }

    template <class VE, class FE>
    inline void xoptional_assembly_storage<VE, FE>::resize(size_type size)
    {
        value().resize(size);
        has_value().resize(size);
    }

    template <class VE, class FE>
    inline auto xoptional_assembly_storage<VE, FE>::operator[](size_type i) -> reference
    {
        return reference(value()[i], has_value()[i]);
    }

    template <class VE, class FE>
    inline auto xoptional_assembly_storage<VE, FE>::operator[](size_type i) const -> const_reference
    {
        return const_reference(value()[i], has_value()[i]);
    }

    template <class VE, class FE>
    inline auto xoptional_assembly_storage<VE, FE>::front() -> reference
    {
        return reference(value()[0], has_value()[0]);
    }

    template <class VE, class FE>
    inline auto xoptional_assembly_storage<VE, FE>::front() const -> const_reference
    {
        return const_reference(value()[0], has_value()[0]);
    }

    template <class VE, class FE>
    inline auto xoptional_assembly_storage<VE, FE>::back() -> reference
    {
        return reference(value()[size() - 1], has_value()[size() - 1]);
    }

    template <class VE, class FE>
    inline auto xoptional_assembly_storage<VE, FE>::back() const -> const_reference
    {
        return const_reference(value()[size() - 1], has_value()[size() - 1]);
    }

    template <class VE, class FE>
    inline auto xoptional_assembly_storage<VE, FE>::data() noexcept -> pointer
    {
        pointer(front());
    }

    template <class VE, class FE>
    inline auto xoptional_assembly_storage<VE, FE>::data() const noexcept -> const_pointer
    {
        const_pointer(front());
    }

    template <class VE, class FE>
    inline auto xoptional_assembly_storage<VE, FE>::begin() noexcept -> iterator
    {
        return iterator(value().begin(), has_value().begin());
    }

    template <class VE, class FE>
    inline auto xoptional_assembly_storage<VE, FE>::end() noexcept -> iterator
    {
        return iterator(value().end(), has_value().end());
    }

    template <class VE, class FE>
    inline auto xoptional_assembly_storage<VE, FE>::begin() const noexcept -> const_iterator
    {
        return cbegin();
    }

    template <class VE, class FE>
    inline auto xoptional_assembly_storage<VE, FE>::end() const noexcept -> const_iterator
    {
        return cend();
    }

    template <class VE, class FE>
    inline auto xoptional_assembly_storage<VE, FE>::cbegin() const noexcept -> const_iterator
    {
        return const_iterator(value().begin(), has_value().begin());
    }

    template <class VE, class FE>
    inline auto xoptional_assembly_storage<VE, FE>::cend() const noexcept -> const_iterator
    {
        return const_iterator(value().end(), has_value().end());
    }

    template <class VE, class FE>
    inline auto xoptional_assembly_storage<VE, FE>::rbegin() noexcept -> reverse_iterator
    {
        return reverse_iterator(end());
    }

    template <class VE, class FE>
    inline auto xoptional_assembly_storage<VE, FE>::rend() noexcept -> reverse_iterator
    {
        return reverse_iterator(begin());
    }

    template <class VE, class FE>
    inline auto xoptional_assembly_storage<VE, FE>::rbegin() const noexcept -> const_reverse_iterator
    {
        return crbegin();
    }

    template <class VE, class FE>
    inline auto xoptional_assembly_storage<VE, FE>::rend() const noexcept -> const_reverse_iterator
    {
        return crend();
    }

    template <class VE, class FE>
    inline auto xoptional_assembly_storage<VE, FE>::crbegin() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator(cend());
    }

    template <class VE, class FE>
    inline auto xoptional_assembly_storage<VE, FE>::crend() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator(cbegin());
    }

    template <class VE, class FE>
    inline void xoptional_assembly_storage<VE, FE>::swap(self_type& rhs) noexcept
    {
        m_value.swap(rhs.m_value);
        m_has_value.swap(rhs.m_has_value);
    }

    template <class VE, class FE>
    inline auto xoptional_assembly_storage<VE, FE>::value() noexcept -> value_storage&
    {
        return m_value;
    }

    template <class VE, class FE>
    inline auto xoptional_assembly_storage<VE, FE>::value() const noexcept -> const value_storage&
    {
        return m_value;
    }

    template <class VE, class FE>
    inline auto xoptional_assembly_storage<VE, FE>::has_value() noexcept -> flag_storage&
    {
        return m_has_value;
    }

    template <class VE, class FE>
    inline auto xoptional_assembly_storage<VE, FE>::has_value() const noexcept -> const flag_storage&
    {
        return m_has_value;
    }

    template <class VE, class FE>
    bool
    operator==(const xoptional_assembly_storage<VE, FE>& lhs, const xoptional_assembly_storage<VE, FE>& rhs)
    {
        return lhs.value() == rhs.value() && lhs.has_value() == rhs.has_value();
    }

    template <class VE, class FE>
    bool
    operator!=(const xoptional_assembly_storage<VE, FE>& lhs, const xoptional_assembly_storage<VE, FE>& rhs)
    {
        return !(lhs == rhs);
    }

    template <class VE, class FE>
    bool operator<(const xoptional_assembly_storage<VE, FE>& lhs, const xoptional_assembly_storage<VE, FE>& rhs)
    {
        return lhs.value() < rhs.value() && lhs.has_value() == rhs.has_value();
    }

    template <class VE, class FE>
    bool
    operator<=(const xoptional_assembly_storage<VE, FE>& lhs, const xoptional_assembly_storage<VE, FE>& rhs)
    {
        return lhs.value() <= rhs.value() && lhs.has_value() == rhs.has_value();
    }

    template <class VE, class FE>
    bool operator>(const xoptional_assembly_storage<VE, FE>& lhs, const xoptional_assembly_storage<VE, FE>& rhs)
    {
        return lhs.value() > rhs.value() && lhs.has_value() == rhs.has_value();
    }

    template <class VE, class FE>
    bool
    operator>=(const xoptional_assembly_storage<VE, FE>& lhs, const xoptional_assembly_storage<VE, FE>& rhs)
    {
        return lhs.value() >= rhs.value() && lhs.has_value() == rhs.has_value();
    }

    template <class VE, class FE>
    void swap(xoptional_assembly_storage<VE, FE>& lhs, xoptional_assembly_storage<VE, FE>& rhs) noexcept
    {
        lhs.swap(rhs);
    }

    /******************************************************
     * xoptional_assembly_linear_iterator implementation *
     ******************************************************/

    template <class VE, class FE, bool C>
    inline xoptional_assembly_linear_iterator<VE, FE, C>::xoptional_assembly_linear_iterator(
        value_iterator value_it,
        flag_iterator flag_it
    )
        : m_value_it(value_it)
        , m_flag_it(flag_it)
    {
    }

    template <class VE, class FE, bool C>
    inline auto xoptional_assembly_linear_iterator<VE, FE, C>::operator++() -> self_type&
    {
        ++m_value_it;
        ++m_flag_it;
        return *this;
    }

    template <class VE, class FE, bool C>
    inline auto xoptional_assembly_linear_iterator<VE, FE, C>::operator--() -> self_type&
    {
        --m_value_it;
        --m_flag_it;
        return *this;
    }

    template <class VE, class FE, bool C>
    inline auto xoptional_assembly_linear_iterator<VE, FE, C>::operator+=(difference_type n) -> self_type&
    {
        m_value_it += n;
        m_flag_it += n;
        return *this;
    }

    template <class VE, class FE, bool C>
    inline auto xoptional_assembly_linear_iterator<VE, FE, C>::operator-=(difference_type n) -> self_type&
    {
        m_value_it -= n;
        m_flag_it -= n;
        return *this;
    }

    template <class VE, class FE, bool C>
    inline auto xoptional_assembly_linear_iterator<VE, FE, C>::operator-(const self_type& rhs) const
        -> difference_type
    {
        return m_value_it - rhs.m_value_it;
    }

    template <class VE, class FE, bool C>
    inline auto xoptional_assembly_linear_iterator<VE, FE, C>::operator*() const -> reference
    {
        return reference(*m_value_it, *m_flag_it);
    }

    template <class VE, class FE, bool C>
    inline auto xoptional_assembly_linear_iterator<VE, FE, C>::operator->() const -> pointer
    {
        return &(this->operator*());
    }

    template <class VE, class FE, bool C>
    inline bool xoptional_assembly_linear_iterator<VE, FE, C>::operator==(const self_type& rhs) const
    {
        return m_value_it == rhs.m_value_it;
    }

    template <class VE, class FE, bool C>
    inline bool xoptional_assembly_linear_iterator<VE, FE, C>::operator<(const self_type& rhs) const
    {
        return m_value_it < rhs.m_value_it;
    }

    template <class VE, class FE>
    inline xoptional_assembly_storage<VE, FE> optional_assembly_storage(const VE& value, const FE& flag)
    {
        return xoptional_assembly_storage<VE, FE>(value, flag);
    }

    template <class VE, class FE>
    inline xoptional_assembly_storage<VE, FE> optional_assembly_storage(VE& value, FE& flag)
    {
        return xoptional_assembly_storage<VE, FE>(value, flag);
    }
}

#endif
