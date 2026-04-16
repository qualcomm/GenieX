/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_ACCESSIBLE_HPP
#define XTENSOR_ACCESSIBLE_HPP

#include "../core/xstrides.hpp"
#include "../core/xtensor_forward.hpp"
#include "../utils/xexception.hpp"

namespace xt
{
    /**
     * @class xconst_accessible
     * @brief Base class for implementation of common expression constant access methods.
     *
     * The xaccessible class implements constant access methods common to all expressions.
     *
     * @tparam D The derived type, i.e. the inheriting class for which xconst_accessible
     *           provides the interface.
     */
    template <class D>
    class xconst_accessible
    {
    public:

        using derived_type = D;
        using inner_types = xcontainer_inner_types<D>;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;
        using size_type = typename inner_types::size_type;

        size_type size() const noexcept;
        size_type dimension() const noexcept;
        size_type shape(size_type index) const;

        template <class... Args>
        const_reference at(Args... args) const;

        template <class S>
        disable_integral_t<S, const_reference> operator[](const S& index) const;
        template <class I>
        const_reference operator[](std::initializer_list<I> index) const;
        const_reference operator[](size_type i) const;

        template <class... Args>
        const_reference periodic(Args... args) const;

        template <class... Args>
        bool in_bounds(Args... args) const;

        const_reference front() const;
        const_reference back() const;

    protected:

        xconst_accessible() = default;
        ~xconst_accessible() = default;

        xconst_accessible(const xconst_accessible&) = default;
        xconst_accessible& operator=(const xconst_accessible&) = default;

        xconst_accessible(xconst_accessible&&) = default;
        xconst_accessible& operator=(xconst_accessible&&) = default;

    private:

        const derived_type& derived_cast() const noexcept;
    };

    /**
     * @class xaccessible
     * @brief Base class for implementation of common expression access methods.
     *
     * The xaccessible class implements access methods common to all expressions.
     *
     * @tparam D The derived type, i.e. the inheriting class for which xaccessible
     *           provides the interface.
     */
    template <class D>
    class xaccessible : public xconst_accessible<D>
    {
    public:

        using base_type = xconst_accessible<D>;
        using derived_type = typename base_type::derived_type;
        using reference = typename base_type::reference;
        using size_type = typename base_type::size_type;

        template <class... Args>
        reference at(Args... args);

        template <class S>
        disable_integral_t<S, reference> operator[](const S& index);
        template <class I>
        reference operator[](std::initializer_list<I> index);
        reference operator[](size_type i);

        template <class... Args>
        reference periodic(Args... args);

        reference front();
        reference back();

        using base_type::at;
        using base_type::operator[];
        using base_type::back;
        using base_type::front;
        using base_type::periodic;

    protected:

        xaccessible() = default;
        ~xaccessible() = default;

        xaccessible(const xaccessible&) = default;
        xaccessible& operator=(const xaccessible&) = default;

        xaccessible(xaccessible&&) = default;
        xaccessible& operator=(xaccessible&&) = default;

    private:

        derived_type& derived_cast() noexcept;
    };

    /************************************
     * xconst_accessible implementation *
     ************************************/

    /**
     * Returns the size of the expression.
     */
    template <class D>
    inline auto xconst_accessible<D>::size() const noexcept -> size_type
    {
        return compute_size(derived_cast().shape());
    }

    /**
     * Returns the number of dimensions of the expression.
     */
    template <class D>
    inline auto xconst_accessible<D>::dimension() const noexcept -> size_type
    {
        return derived_cast().shape().size();
    }

    /**
     * Returns the i-th dimension of the expression.
     */
    template <class D>
    inline auto xconst_accessible<D>::shape(size_type index) const -> size_type
    {
        return derived_cast().shape()[index];
    }

    /**
     * Returns a constant reference to the element at the specified position in the expression,
     * after dimension and bounds checking.
     * @param args a list of indices specifying the position in the expression. Indices
     * must be unsigned integers, the number of indices should be equal to the number of dimensions
     * of the expression.
     * @exception std::out_of_range if the number of argument is greater than the number of dimensions
     * or if indices are out of bounds.
     */
    template <class D>
    template <class... Args>
    inline auto xconst_accessible<D>::at(Args... args) const -> const_reference
    {
        check_access(derived_cast().shape(), args...);
        return derived_cast().operator()(args...);
    }

    /**
     * Returns a constant reference to the element at the specified position in the expression.
     * @param index a sequence of indices specifying the position in the expression. Indices
     * must be unsigned integers, the number of indices in the list should be equal or greater
     * than the number of dimensions of the expression.
     */
    template <class D>
    template <class S>
    inline auto xconst_accessible<D>::operator[](const S& index) const
        -> disable_integral_t<S, const_reference>
    {
        return derived_cast().element(index.cbegin(), index.cend());
    }

    template <class D>
    template <class I>
    inline auto xconst_accessible<D>::operator[](std::initializer_list<I> index) const -> const_reference
    {
        return derived_cast().element(index.begin(), index.end());
    }

    template <class D>
    inline auto xconst_accessible<D>::operator[](size_type i) const -> const_reference
    {
        return derived_cast().operator()(i);
    }

    /**
     * Returns a constant reference to the element at the specified position in the expression,
     * after applying periodicity to the indices (negative and 'overflowing' indices are changed).
     * @param args a list of indices specifying the position in the expression. Indices
     * must be integers, the number of indices should be equal to the number of dimensions
     * of the expression.
     */
    template <class D>
    template <class... Args>
    inline auto xconst_accessible<D>::periodic(Args... args) const -> const_reference
    {
        normalize_periodic(derived_cast().shape(), args...);
        return derived_cast()(static_cast<size_type>(args)...);
    }

    /**
     * Returns a constant reference to first the element of the expression
     */
    template <class D>
    inline auto xconst_accessible<D>::front() const -> const_reference
    {
        return *derived_cast().begin();
    }

    /**
     * Returns a constant reference to last the element of the expression
     */
    template <class D>
    inline auto xconst_accessible<D>::back() const -> const_reference
    {
        return *std::prev(derived_cast().end());
    }

    /**
     * Returns ``true`` only if the the specified position is a valid entry in the expression.
     * @param args a list of indices specifying the position in the expression.
     * @return bool
     */
    template <class D>
    template <class... Args>
    inline bool xconst_accessible<D>::in_bounds(Args... args) const
    {
        return check_in_bounds(derived_cast().shape(), args...);
    }

    template <class D>
    inline auto xconst_accessible<D>::derived_cast() const noexcept -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

    /******************************
     * xaccessible implementation *
     ******************************/

    /**
     * Returns a reference to the element at the specified position in the expression,
     * after dimension and bounds checking.
     * @param args a list of indices specifying the position in the expression. Indices
     * must be unsigned integers, the number of indices should be equal to the number of dimensions
     * of the expression.
     * @exception std::out_of_range if the number of argument is greater than the number of dimensions
     * or if indices are out of bounds.
     */
    template <class D>
    template <class... Args>
    inline auto xaccessible<D>::at(Args... args) -> reference
    {
        check_access(derived_cast().shape(), args...);
        return derived_cast().operator()(args...);
    }

    /**
     * Returns a reference to the element at the specified position in the expression.
     * @param index a sequence of indices specifying the position in the expression. Indices
     * must be unsigned integers, the number of indices in the list should be equal or greater
     * than the number of dimensions of the expression.
     */
    template <class D>
    template <class S>
    inline auto xaccessible<D>::operator[](const S& index) -> disable_integral_t<S, reference>
    {
        return derived_cast().element(index.cbegin(), index.cend());
    }

    template <class D>
    template <class I>
    inline auto xaccessible<D>::operator[](std::initializer_list<I> index) -> reference
    {
        return derived_cast().element(index.begin(), index.end());
    }

    template <class D>
    inline auto xaccessible<D>::operator[](size_type i) -> reference
    {
        return derived_cast().operator()(i);
    }

    /**
     * Returns a reference to the element at the specified position in the expression,
     * after applying periodicity to the indices (negative and 'overflowing' indices are changed).
     * @param args a list of indices specifying the position in the expression. Indices
     * must be integers, the number of indices should be equal to the number of dimensions
     * of the expression.
     */
    template <class D>
    template <class... Args>
    inline auto xaccessible<D>::periodic(Args... args) -> reference
    {
        normalize_periodic(derived_cast().shape(), args...);
        return derived_cast()(args...);
    }

    /**
     * Returns a reference to the first element of the expression.
     */
    template <class D>
    inline auto xaccessible<D>::front() -> reference
    {
        return *derived_cast().begin();
    }

    /**
     * Returns a reference to the last element of the expression.
     */
    template <class D>
    inline auto xaccessible<D>::back() -> reference
    {
        return *std::prev(derived_cast().end());
    }

    template <class D>
    inline auto xaccessible<D>::derived_cast() noexcept -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

}

#endif
