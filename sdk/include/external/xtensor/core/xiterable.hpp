/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_ITERABLE_HPP
#define XTENSOR_ITERABLE_HPP

#include "../core/xiterator.hpp"

namespace xt
{

    /*******************
     * xconst_iterable *
     *******************/

    template <class D>
    struct xiterable_inner_types;

    /**
     * @class xconst_iterable
     * @brief Base class for multidimensional iterable constant expressions
     *
     * The xconst_iterable class defines the interface for multidimensional
     * constant expressions that can be iterated.
     *
     * @tparam D The derived type, i.e. the inheriting class for which xconst_iterable
     *           provides the interface.
     */
    template <class D>
    class xconst_iterable
    {
    public:

        using derived_type = D;

        using iterable_types = xiterable_inner_types<D>;
        using inner_shape_type = typename iterable_types::inner_shape_type;

        using stepper = typename iterable_types::stepper;
        using const_stepper = typename iterable_types::const_stepper;

        template <layout_type L>
        using layout_iterator = xiterator<stepper, inner_shape_type*, L>;
        template <layout_type L>
        using const_layout_iterator = xiterator<const_stepper, inner_shape_type*, L>;
        template <layout_type L>
        using reverse_layout_iterator = std::reverse_iterator<layout_iterator<L>>;
        template <layout_type L>
        using const_reverse_layout_iterator = std::reverse_iterator<const_layout_iterator<L>>;

        using linear_iterator = layout_iterator<XTENSOR_DEFAULT_TRAVERSAL>;
        using const_linear_iterator = const_layout_iterator<XTENSOR_DEFAULT_TRAVERSAL>;
        using reverse_linear_iterator = reverse_layout_iterator<XTENSOR_DEFAULT_TRAVERSAL>;
        using const_reverse_linear_iterator = const_reverse_layout_iterator<XTENSOR_DEFAULT_TRAVERSAL>;

        template <class S, layout_type L>
        using broadcast_iterator = xiterator<stepper, S, L>;
        template <class S, layout_type L>
        using const_broadcast_iterator = xiterator<const_stepper, S, L>;
        template <class S, layout_type L>
        using reverse_broadcast_iterator = std::reverse_iterator<broadcast_iterator<S, L>>;
        template <class S, layout_type L>
        using const_reverse_broadcast_iterator = std::reverse_iterator<const_broadcast_iterator<S, L>>;

        using iterator = layout_iterator<XTENSOR_DEFAULT_TRAVERSAL>;
        using const_iterator = const_layout_iterator<XTENSOR_DEFAULT_TRAVERSAL>;
        using reverse_iterator = reverse_layout_iterator<XTENSOR_DEFAULT_TRAVERSAL>;
        using const_reverse_iterator = const_reverse_layout_iterator<XTENSOR_DEFAULT_TRAVERSAL>;

        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        const_layout_iterator<L> begin() const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        const_layout_iterator<L> end() const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        const_layout_iterator<L> cbegin() const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        const_layout_iterator<L> cend() const noexcept;

        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        const_reverse_layout_iterator<L> rbegin() const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        const_reverse_layout_iterator<L> rend() const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        const_reverse_layout_iterator<L> crbegin() const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        const_reverse_layout_iterator<L> crend() const noexcept;

        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class S>
        const_broadcast_iterator<S, L> begin(const S& shape) const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class S>
        const_broadcast_iterator<S, L> end(const S& shape) const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class S>
        const_broadcast_iterator<S, L> cbegin(const S& shape) const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class S>
        const_broadcast_iterator<S, L> cend(const S& shape) const noexcept;

        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class S>
        const_reverse_broadcast_iterator<S, L> rbegin(const S& shape) const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class S>
        const_reverse_broadcast_iterator<S, L> rend(const S& shape) const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class S>
        const_reverse_broadcast_iterator<S, L> crbegin(const S& shape) const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class S>
        const_reverse_broadcast_iterator<S, L> crend(const S& shape) const noexcept;

    protected:

        const inner_shape_type& get_shape() const;

    private:

        template <layout_type L>
        const_layout_iterator<L> get_cbegin(bool end_index) const noexcept;
        template <layout_type L>
        const_layout_iterator<L> get_cend(bool end_index) const noexcept;

        template <layout_type L, class S>
        const_broadcast_iterator<S, L> get_cbegin(const S& shape, bool end_index) const noexcept;
        template <layout_type L, class S>
        const_broadcast_iterator<S, L> get_cend(const S& shape, bool end_index) const noexcept;

        template <class S>
        const_stepper get_stepper_begin(const S& shape) const noexcept;
        template <class S>
        const_stepper get_stepper_end(const S& shape, layout_type l) const noexcept;

        const derived_type& derived_cast() const;
    };

    /*************
     * xiterable *
     *************/

    /**
     * @class xiterable
     * @brief Base class for multidimensional iterable expressions
     *
     * The xiterable class defines the interface for multidimensional
     * expressions that can be iterated.
     *
     * @tparam D The derived type, i.e. the inheriting class for which xiterable
     *           provides the interface.
     */
    template <class D>
    class xiterable : public xconst_iterable<D>
    {
    public:

        using derived_type = D;

        using base_type = xconst_iterable<D>;
        using inner_shape_type = typename base_type::inner_shape_type;

        using stepper = typename base_type::stepper;
        using const_stepper = typename base_type::const_stepper;

        using linear_iterator = typename base_type::linear_iterator;
        using reverse_linear_iterator = typename base_type::reverse_linear_iterator;

        template <layout_type L>
        using layout_iterator = typename base_type::template layout_iterator<L>;
        template <layout_type L>
        using const_layout_iterator = typename base_type::template const_layout_iterator<L>;
        template <layout_type L>
        using reverse_layout_iterator = typename base_type::template reverse_layout_iterator<L>;
        template <layout_type L>
        using const_reverse_layout_iterator = typename base_type::template const_reverse_layout_iterator<L>;

        template <class S, layout_type L>
        using broadcast_iterator = typename base_type::template broadcast_iterator<S, L>;
        template <class S, layout_type L>
        using const_broadcast_iterator = typename base_type::template const_broadcast_iterator<S, L>;
        template <class S, layout_type L>
        using reverse_broadcast_iterator = typename base_type::template reverse_broadcast_iterator<S, L>;
        template <class S, layout_type L>
        using const_reverse_broadcast_iterator = typename base_type::template const_reverse_broadcast_iterator<S, L>;

        using iterator = typename base_type::iterator;
        using const_iterator = typename base_type::const_iterator;
        using reverse_iterator = typename base_type::reverse_iterator;
        using const_reverse_iterator = typename base_type::const_reverse_iterator;

        using base_type::begin;
        using base_type::end;
        using base_type::rbegin;
        using base_type::rend;

        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        layout_iterator<L> begin() noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        layout_iterator<L> end() noexcept;

        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        reverse_layout_iterator<L> rbegin() noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        reverse_layout_iterator<L> rend() noexcept;

        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class S>
        broadcast_iterator<S, L> begin(const S& shape) noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class S>
        broadcast_iterator<S, L> end(const S& shape) noexcept;

        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class S>
        reverse_broadcast_iterator<S, L> rbegin(const S& shape) noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class S>
        reverse_broadcast_iterator<S, L> rend(const S& shape) noexcept;

    private:

        template <layout_type L>
        layout_iterator<L> get_begin(bool end_index) noexcept;
        template <layout_type L>
        layout_iterator<L> get_end(bool end_index) noexcept;

        template <layout_type L, class S>
        broadcast_iterator<S, L> get_begin(const S& shape, bool end_index) noexcept;
        template <layout_type L, class S>
        broadcast_iterator<S, L> get_end(const S& shape, bool end_index) noexcept;

        template <class S>
        stepper get_stepper_begin(const S& shape) noexcept;
        template <class S>
        stepper get_stepper_end(const S& shape, layout_type l) noexcept;

        template <class S>
        const_stepper get_stepper_begin(const S& shape) const noexcept;
        template <class S>
        const_stepper get_stepper_end(const S& shape, layout_type l) const noexcept;

        derived_type& derived_cast();
    };

    /************************
     * xcontiguous_iterable *
     ************************/

    template <class D>
    struct xcontainer_inner_types;

    namespace detail
    {
        template <class T, bool is_const>
        struct get_storage_iterator;

        template <class T>
        struct get_storage_iterator<T, true>
        {
            using iterator = typename T::const_iterator;
            using const_iterator = typename T::const_iterator;
            using reverse_iterator = typename T::const_reverse_iterator;
            using const_reverse_iterator = typename T::const_reverse_iterator;
        };

        template <class T>
        struct get_storage_iterator<T, false>
        {
            using iterator = typename T::iterator;
            using const_iterator = typename T::const_iterator;
            using reverse_iterator = typename T::reverse_iterator;
            using const_reverse_iterator = typename T::const_reverse_iterator;
        };

        template <class D, bool has_storage_type>
        struct linear_iterator_traits_impl;

        template <class D>
        struct linear_iterator_traits_impl<D, true>
        {
            using inner_types = xcontainer_inner_types<D>;
            using storage_type = typename inner_types::storage_type;
            using iterator_type = get_storage_iterator<storage_type, std::is_const<storage_type>::value>;
            using linear_iterator = typename iterator_type::iterator;
            using const_linear_iterator = typename iterator_type::const_iterator;
            using reverse_linear_iterator = typename iterator_type::reverse_iterator;
            using const_reverse_linear_iterator = typename iterator_type::const_reverse_iterator;
        };

        template <class D>
        struct linear_iterator_traits_impl<D, false>
        {
            using inner_types = xcontainer_inner_types<D>;
            using xexpression_type = typename inner_types::xexpression_type;
            using linear_iterator = typename xexpression_type::linear_iterator;
            using const_linear_iterator = typename xexpression_type::const_linear_iterator;
            using reverse_linear_iterator = typename xexpression_type::reverse_linear_iterator;
            using const_reverse_linear_iterator = typename xexpression_type::const_reverse_linear_iterator;
        };

        template <class D>
        using linear_iterator_traits = linear_iterator_traits_impl<D, has_storage_type<D>::value>;
    }

    /**
     * @class xcontiguous_iterable
     * @brief Base class for multidimensional iterable expressions with
     * contiguous storage
     *
     * The xcontiguous_iterable class defines the interface for multidimensional
     * expressions with contiguous that can be iterated.
     *
     * @tparam D The derived type, i.e. the inheriting class for which xcontiguous_iterable
     *           provides the interface.
     */
    template <class D>
    class xcontiguous_iterable : private xiterable<D>
    {
    public:

        using derived_type = D;

        using inner_types = xcontainer_inner_types<D>;

        using iterable_base = xiterable<D>;
        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        static constexpr layout_type static_layout = inner_types::layout;

#if defined(_MSC_VER) && _MSC_VER >= 1910
        // Workaround for compiler bug in Visual Studio 2017 with respect to alias templates with non-type
        // parameters.
        template <layout_type L>
        using layout_iterator = xiterator<typename iterable_base::stepper, typename iterable_base::inner_shape_type*, L>;
        template <layout_type L>
        using const_layout_iterator = xiterator<
            typename iterable_base::const_stepper,
            typename iterable_base::inner_shape_type*,
            L>;
        template <layout_type L>
        using reverse_layout_iterator = std::reverse_iterator<layout_iterator<L>>;
        template <layout_type L>
        using const_reverse_layout_iterator = std::reverse_iterator<const_layout_iterator<L>>;
#else
        template <layout_type L>
        using layout_iterator = typename iterable_base::template layout_iterator<L>;
        template <layout_type L>
        using const_layout_iterator = typename iterable_base::template const_layout_iterator<L>;
        template <layout_type L>
        using reverse_layout_iterator = typename iterable_base::template reverse_layout_iterator<L>;
        template <layout_type L>
        using const_reverse_layout_iterator = typename iterable_base::template const_reverse_layout_iterator<L>;
#endif

        template <class S, layout_type L>
        using broadcast_iterator = typename iterable_base::template broadcast_iterator<S, L>;
        template <class S, layout_type L>
        using const_broadcast_iterator = typename iterable_base::template const_broadcast_iterator<S, L>;
        template <class S, layout_type L>
        using reverse_broadcast_iterator = typename iterable_base::template reverse_broadcast_iterator<S, L>;
        template <class S, layout_type L>
        using const_reverse_broadcast_iterator = typename iterable_base::template const_reverse_broadcast_iterator<S, L>;

        using linear_traits = detail::linear_iterator_traits<D>;
        using linear_iterator = typename linear_traits::linear_iterator;
        using const_linear_iterator = typename linear_traits::const_linear_iterator;
        using reverse_linear_iterator = typename linear_traits::reverse_linear_iterator;
        using const_reverse_linear_iterator = typename linear_traits::const_reverse_linear_iterator;

        template <layout_type L, class It1, class It2>
        using select_iterator_impl = std::conditional_t<L == static_layout, It1, It2>;

        template <layout_type L>
        using select_iterator = select_iterator_impl<L, linear_iterator, layout_iterator<L>>;
        template <layout_type L>
        using select_const_iterator = select_iterator_impl<L, const_linear_iterator, const_layout_iterator<L>>;
        template <layout_type L>
        using select_reverse_iterator = select_iterator_impl<L, reverse_linear_iterator, reverse_layout_iterator<L>>;
        template <layout_type L>
        using select_const_reverse_iterator = select_iterator_impl<
            L,
            const_reverse_linear_iterator,
            const_reverse_layout_iterator<L>>;

        using iterator = select_iterator<XTENSOR_DEFAULT_TRAVERSAL>;
        using const_iterator = select_const_iterator<XTENSOR_DEFAULT_TRAVERSAL>;
        using reverse_iterator = select_reverse_iterator<XTENSOR_DEFAULT_TRAVERSAL>;
        using const_reverse_iterator = select_const_reverse_iterator<XTENSOR_DEFAULT_TRAVERSAL>;

        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        select_iterator<L> begin() noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        select_iterator<L> end() noexcept;

        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        select_const_iterator<L> begin() const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        select_const_iterator<L> end() const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        select_const_iterator<L> cbegin() const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        select_const_iterator<L> cend() const noexcept;

        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        select_reverse_iterator<L> rbegin() noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        select_reverse_iterator<L> rend() noexcept;

        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        select_const_reverse_iterator<L> rbegin() const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        select_const_reverse_iterator<L> rend() const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        select_const_reverse_iterator<L> crbegin() const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>
        select_const_reverse_iterator<L> crend() const noexcept;

        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class S>
        broadcast_iterator<S, L> begin(const S& shape) noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class S>
        broadcast_iterator<S, L> end(const S& shape) noexcept;

        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class S>
        const_broadcast_iterator<S, L> begin(const S& shape) const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class S>
        const_broadcast_iterator<S, L> end(const S& shape) const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class S>
        const_broadcast_iterator<S, L> cbegin(const S& shape) const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class S>
        const_broadcast_iterator<S, L> cend(const S& shape) const noexcept;

        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class S>
        reverse_broadcast_iterator<S, L> rbegin(const S& shape) noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class S>
        reverse_broadcast_iterator<S, L> rend(const S& shape) noexcept;

        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class S>
        const_reverse_broadcast_iterator<S, L> rbegin(const S& shape) const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class S>
        const_reverse_broadcast_iterator<S, L> rend(const S& shape) const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class S>
        const_reverse_broadcast_iterator<S, L> crbegin(const S& shape) const noexcept;
        template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class S>
        const_reverse_broadcast_iterator<S, L> crend(const S& shape) const noexcept;

    private:

        derived_type& derived_cast();
        const derived_type& derived_cast() const;

        friend class xiterable<D>;
        friend class xconst_iterable<D>;
    };

    /**********************************
     * xconst_iterable implementation *
     **********************************/

    /**
     * @name Constant iterators
     */
    //@{
    /**
     * Returns a constant iterator to the first element of the expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xconst_iterable<D>::begin() const noexcept -> const_layout_iterator<L>
    {
        return this->template cbegin<L>();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xconst_iterable<D>::end() const noexcept -> const_layout_iterator<L>
    {
        return this->template cend<L>();
    }

    /**
     * Returns a constant iterator to the first element of the expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xconst_iterable<D>::cbegin() const noexcept -> const_layout_iterator<L>
    {
        return this->template get_cbegin<L>(false);
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xconst_iterable<D>::cend() const noexcept -> const_layout_iterator<L>
    {
        return this->template get_cend<L>(true);
    }

    //@}

    /**
     * @name Constant reverse iterators
     */
    //@{
    /**
     * Returns a constant iterator to the first element of the reversed expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xconst_iterable<D>::rbegin() const noexcept -> const_reverse_layout_iterator<L>
    {
        return this->template crbegin<L>();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the reversed expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xconst_iterable<D>::rend() const noexcept -> const_reverse_layout_iterator<L>
    {
        return this->template crend<L>();
    }

    /**
     * Returns a constant iterator to the first element of the reversed expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xconst_iterable<D>::crbegin() const noexcept -> const_reverse_layout_iterator<L>
    {
        return const_reverse_layout_iterator<L>(get_cend<L>(true));
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the reversed expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xconst_iterable<D>::crend() const noexcept -> const_reverse_layout_iterator<L>
    {
        return const_reverse_layout_iterator<L>(get_cbegin<L>(false));
    }

    //@}

    /**
     * @name Constant broadcast iterators
     */
    //@{
    /**
     * Returns a constant iterator to the first element of the expression. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L, class S>
    inline auto xconst_iterable<D>::begin(const S& shape) const noexcept -> const_broadcast_iterator<S, L>
    {
        return cbegin<L>(shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L, class S>
    inline auto xconst_iterable<D>::end(const S& shape) const noexcept -> const_broadcast_iterator<S, L>
    {
        return cend<L>(shape);
    }

    /**
     * Returns a constant iterator to the first element of the expression. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L, class S>
    inline auto xconst_iterable<D>::cbegin(const S& shape) const noexcept -> const_broadcast_iterator<S, L>
    {
        return get_cbegin<L, S>(shape, false);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L, class S>
    inline auto xconst_iterable<D>::cend(const S& shape) const noexcept -> const_broadcast_iterator<S, L>
    {
        return get_cend<L, S>(shape, true);
    }

    //@}

    /**
     * @name Constant reverse broadcast iterators
     */
    //@{
    /**
     * Returns a constant iterator to the first element of the reversed expression.
     * The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L, class S>
    inline auto xconst_iterable<D>::rbegin(const S& shape) const noexcept
        -> const_reverse_broadcast_iterator<S, L>
    {
        return crbegin<L, S>(shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * reversed expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L, class S>
    inline auto xconst_iterable<D>::rend(const S& shape) const noexcept
        -> const_reverse_broadcast_iterator<S, L>
    {
        return crend<L, S>(shape);
    }

    /**
     * Returns a constant iterator to the first element of the reversed expression.
     * The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L, class S>
    inline auto xconst_iterable<D>::crbegin(const S& shape) const noexcept
        -> const_reverse_broadcast_iterator<S, L>
    {
        return const_reverse_broadcast_iterator<S, L>(get_cend<L, S>(shape, true));
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * reversed expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L, class S>
    inline auto xconst_iterable<D>::crend(const S& shape) const noexcept
        -> const_reverse_broadcast_iterator<S, L>
    {
        return const_reverse_broadcast_iterator<S, L>(get_cbegin<L, S>(shape, false));
    }

    //@}

    template <class D>
    template <layout_type L>
    inline auto xconst_iterable<D>::get_cbegin(bool end_index) const noexcept -> const_layout_iterator<L>
    {
        return const_layout_iterator<L>(get_stepper_begin(get_shape()), &get_shape(), end_index);
    }

    template <class D>
    template <layout_type L>
    inline auto xconst_iterable<D>::get_cend(bool end_index) const noexcept -> const_layout_iterator<L>
    {
        return const_layout_iterator<L>(get_stepper_end(get_shape(), L), &get_shape(), end_index);
    }

    template <class D>
    template <layout_type L, class S>
    inline auto xconst_iterable<D>::get_cbegin(const S& shape, bool end_index) const noexcept
        -> const_broadcast_iterator<S, L>
    {
        return const_broadcast_iterator<S, L>(get_stepper_begin(shape), shape, end_index);
    }

    template <class D>
    template <layout_type L, class S>
    inline auto xconst_iterable<D>::get_cend(const S& shape, bool end_index) const noexcept
        -> const_broadcast_iterator<S, L>
    {
        return const_broadcast_iterator<S, L>(get_stepper_end(shape, L), shape, end_index);
    }

    template <class D>
    template <class S>
    inline auto xconst_iterable<D>::get_stepper_begin(const S& shape) const noexcept -> const_stepper
    {
        return derived_cast().stepper_begin(shape);
    }

    template <class D>
    template <class S>
    inline auto xconst_iterable<D>::get_stepper_end(const S& shape, layout_type l) const noexcept
        -> const_stepper
    {
        return derived_cast().stepper_end(shape, l);
    }

    template <class D>
    inline auto xconst_iterable<D>::get_shape() const -> const inner_shape_type&
    {
        return derived_cast().shape();
    }

    template <class D>
    inline auto xconst_iterable<D>::derived_cast() const -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

    /****************************
     * xiterable implementation *
     ****************************/

    /**
     * @name Iterators
     */
    //@{
    /**
     * Returns an iterator to the first element of the expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xiterable<D>::begin() noexcept -> layout_iterator<L>
    {
        return get_begin<L>(false);
    }

    /**
     * Returns an iterator to the element following the last element
     * of the expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xiterable<D>::end() noexcept -> layout_iterator<L>
    {
        return get_end<L>(true);
    }

    //@}

    /**
     * @name Reverse iterators
     */
    //@{
    /**
     * Returns an iterator to the first element of the reversed expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xiterable<D>::rbegin() noexcept -> reverse_layout_iterator<L>
    {
        return reverse_layout_iterator<L>(get_end<L>(true));
    }

    /**
     * Returns an iterator to the element following the last element
     * of the reversed expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xiterable<D>::rend() noexcept -> reverse_layout_iterator<L>
    {
        return reverse_layout_iterator<L>(get_begin<L>(false));
    }

    //@}

    /**
     * @name Broadcast iterators
     */
    //@{
    /**
     * Returns an iterator to the first element of the expression. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L, class S>
    inline auto xiterable<D>::begin(const S& shape) noexcept -> broadcast_iterator<S, L>
    {
        return get_begin<L, S>(shape, false);
    }

    /**
     * Returns an iterator to the element following the last element of the
     * expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L, class S>
    inline auto xiterable<D>::end(const S& shape) noexcept -> broadcast_iterator<S, L>
    {
        return get_end<L, S>(shape, true);
    }

    //@}

    /**
     * @name Reverse broadcast iterators
     */
    //@{
    /**
     * Returns an iterator to the first element of the reversed expression. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L, class S>
    inline auto xiterable<D>::rbegin(const S& shape) noexcept -> reverse_broadcast_iterator<S, L>
    {
        return reverse_broadcast_iterator<S, L>(get_end<L, S>(shape, true));
    }

    /**
     * Returns an iterator to the element following the last element of the
     * reversed expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L, class S>
    inline auto xiterable<D>::rend(const S& shape) noexcept -> reverse_broadcast_iterator<S, L>
    {
        return reverse_broadcast_iterator<S, L>(get_begin<L, S>(shape, false));
    }

    //@}

    template <class D>
    template <layout_type L>
    inline auto xiterable<D>::get_begin(bool end_index) noexcept -> layout_iterator<L>
    {
        return layout_iterator<L>(get_stepper_begin(this->get_shape()), &(this->get_shape()), end_index);
    }

    template <class D>
    template <layout_type L>
    inline auto xiterable<D>::get_end(bool end_index) noexcept -> layout_iterator<L>
    {
        return layout_iterator<L>(get_stepper_end(this->get_shape(), L), &(this->get_shape()), end_index);
    }

    template <class D>
    template <layout_type L, class S>
    inline auto xiterable<D>::get_begin(const S& shape, bool end_index) noexcept -> broadcast_iterator<S, L>
    {
        return broadcast_iterator<S, L>(get_stepper_begin(shape), shape, end_index);
    }

    template <class D>
    template <layout_type L, class S>
    inline auto xiterable<D>::get_end(const S& shape, bool end_index) noexcept -> broadcast_iterator<S, L>
    {
        return broadcast_iterator<S, L>(get_stepper_end(shape, L), shape, end_index);
    }

    template <class D>
    template <class S>
    inline auto xiterable<D>::get_stepper_begin(const S& shape) noexcept -> stepper
    {
        return derived_cast().stepper_begin(shape);
    }

    template <class D>
    template <class S>
    inline auto xiterable<D>::get_stepper_end(const S& shape, layout_type l) noexcept -> stepper
    {
        return derived_cast().stepper_end(shape, l);
    }

    template <class D>
    template <class S>
    inline auto xiterable<D>::get_stepper_begin(const S& shape) const noexcept -> const_stepper
    {
        return derived_cast().stepper_begin(shape);
    }

    template <class D>
    template <class S>
    inline auto xiterable<D>::get_stepper_end(const S& shape, layout_type l) const noexcept -> const_stepper
    {
        return derived_cast().stepper_end(shape, l);
    }

    template <class D>
    inline auto xiterable<D>::derived_cast() -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    /***************************************
     * xcontiguous_iterable implementation *
     ***************************************/

    /**
     * @name Iterators
     */
    //@{
    /**
     * Returns an iterator to the first element of the expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xcontiguous_iterable<D>::begin() noexcept -> select_iterator<L>
    {
        if constexpr (L == static_layout)
        {
            return derived_cast().linear_begin();
        }
        else
        {
            return iterable_base::template begin<L>();
        }
    }

    /**
     * Returns an iterator to the element following the last element
     * of the expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xcontiguous_iterable<D>::end() noexcept -> select_iterator<L>
    {
        if constexpr (L == static_layout)
        {
            return derived_cast().linear_end();
        }
        else
        {
            return iterable_base::template end<L>();
        }
    }

    /**
     * Returns a constant iterator to the first element of the expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xcontiguous_iterable<D>::begin() const noexcept -> select_const_iterator<L>
    {
        return this->template cbegin<L>();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xcontiguous_iterable<D>::end() const noexcept -> select_const_iterator<L>
    {
        return this->template cend<L>();
    }

    /**
     * Returns a constant iterator to the first element of the expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xcontiguous_iterable<D>::cbegin() const noexcept -> select_const_iterator<L>
    {
        if constexpr (L == static_layout)
        {
            return derived_cast().linear_cbegin();
        }
        else
        {
            return iterable_base::template cbegin<L>();
        }
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xcontiguous_iterable<D>::cend() const noexcept -> select_const_iterator<L>
    {
        if constexpr (L == static_layout)
        {
            return derived_cast().linear_cend();
        }
        else
        {
            return iterable_base::template cend<L>();
        }
    }

    //@}

    /**
     * @name Reverse iterators
     */
    //@{
    /**
     * Returns an iterator to the first element of the reversed expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xcontiguous_iterable<D>::rbegin() noexcept -> select_reverse_iterator<L>
    {
        if constexpr (L == static_layout)
        {
            return derived_cast().linear_rbegin();
        }
        else
        {
            return iterable_base::template rbegin<L>();
        }
    }

    /**
     * Returns an iterator to the element following the last element
     * of the reversed expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xcontiguous_iterable<D>::rend() noexcept -> select_reverse_iterator<L>
    {
        if constexpr (L == static_layout)
        {
            return derived_cast().linear_rend();
        }
        else
        {
            return iterable_base::template rend<L>();
        }
    }

    /**
     * Returns a constant iterator to the first element of the reversed expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xcontiguous_iterable<D>::rbegin() const noexcept -> select_const_reverse_iterator<L>
    {
        return this->template crbegin<L>();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the reversed expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xcontiguous_iterable<D>::rend() const noexcept -> select_const_reverse_iterator<L>
    {
        return this->template crend<L>();
    }

    /**
     * Returns a constant iterator to the first element of the reversed expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xcontiguous_iterable<D>::crbegin() const noexcept -> select_const_reverse_iterator<L>
    {
        if constexpr (L == static_layout)
        {
            return derived_cast().linear_crbegin();
        }
        else
        {
            return iterable_base::template crbegin<L>();
        }
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the reversed expression.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L>
    inline auto xcontiguous_iterable<D>::crend() const noexcept -> select_const_reverse_iterator<L>
    {
        if constexpr (L == static_layout)
        {
            return derived_cast().linear_crend();
        }
        else
        {
            return iterable_base::template crend<L>();
        }
    }

    //@}

    /**
     * @name Broadcast iterators
     */

    /**
     * Returns an iterator to the first element of the expression. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    //@{
    template <class D>
    template <layout_type L, class S>
    inline auto xcontiguous_iterable<D>::begin(const S& shape) noexcept -> broadcast_iterator<S, L>
    {
        return iterable_base::template begin<L, S>(shape);
    }

    /**
     * Returns an iterator to the element following the last element of the
     * expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L, class S>
    inline auto xcontiguous_iterable<D>::end(const S& shape) noexcept -> broadcast_iterator<S, L>
    {
        return iterable_base::template end<L, S>(shape);
    }

    /**
     * Returns a constant iterator to the first element of the expression. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L, class S>
    inline auto xcontiguous_iterable<D>::begin(const S& shape) const noexcept -> const_broadcast_iterator<S, L>
    {
        return iterable_base::template begin<L, S>(shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L, class S>
    inline auto xcontiguous_iterable<D>::end(const S& shape) const noexcept -> const_broadcast_iterator<S, L>
    {
        return iterable_base::template end<L, S>(shape);
    }

    /**
     * Returns a constant iterator to the first element of the expression. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L, class S>
    inline auto xcontiguous_iterable<D>::cbegin(const S& shape) const noexcept
        -> const_broadcast_iterator<S, L>
    {
        return iterable_base::template cbegin<L, S>(shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L, class S>
    inline auto xcontiguous_iterable<D>::cend(const S& shape) const noexcept -> const_broadcast_iterator<S, L>
    {
        return iterable_base::template cend<L, S>(shape);
    }

    //@}

    /**
     * @name Reverse broadcast iterators
     */
    //@{
    /**
     * Returns an iterator to the first element of the reversed expression. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L, class S>
    inline auto xcontiguous_iterable<D>::rbegin(const S& shape) noexcept -> reverse_broadcast_iterator<S, L>
    {
        return iterable_base::template rbegin<L, S>(shape);
    }

    /**
     * Returns an iterator to the element following the last element of the
     * reversed expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L, class S>
    inline auto xcontiguous_iterable<D>::rend(const S& shape) noexcept -> reverse_broadcast_iterator<S, L>
    {
        return iterable_base::template rend<L, S>(shape);
    }

    /**
     * Returns a constant iterator to the first element of the reversed expression.
     * The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L, class S>
    inline auto xcontiguous_iterable<D>::rbegin(const S& shape) const noexcept
        -> const_reverse_broadcast_iterator<S, L>
    {
        return iterable_base::template rbegin<L, S>(shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * reversed expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L, class S>
    inline auto xcontiguous_iterable<D>::rend(const S& shape) const noexcept
        -> const_reverse_broadcast_iterator<S, L>
    {
        return iterable_base::template rend<L, S>(shape);
    }

    /**
     * Returns a constant iterator to the first element of the reversed expression.
     * The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L, class S>
    inline auto xcontiguous_iterable<D>::crbegin(const S& shape) const noexcept
        -> const_reverse_broadcast_iterator<S, L>
    {
        return iterable_base::template crbegin<L, S>(shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * reversed expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     * @tparam S type of the \c shape parameter.
     * @tparam L order used for the traversal. Default value is \c XTENSOR_DEFAULT_TRAVERSAL.
     */
    template <class D>
    template <layout_type L, class S>
    inline auto xcontiguous_iterable<D>::crend(const S& shape) const noexcept
        -> const_reverse_broadcast_iterator<S, L>
    {
        return iterable_base::template crend<L, S>(shape);
    }

    //@}

    template <class D>
    inline auto xcontiguous_iterable<D>::derived_cast() -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D>
    inline auto xcontiguous_iterable<D>::derived_cast() const -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

}

#endif
