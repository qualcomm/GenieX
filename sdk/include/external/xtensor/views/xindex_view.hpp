/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_INDEX_VIEW_HPP
#define XTENSOR_INDEX_VIEW_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../core/xexpression.hpp"
#include "../core/xiterable.hpp"
#include "../core/xoperation.hpp"
#include "../core/xsemantic.hpp"
#include "../core/xstrides.hpp"
#include "../utils/xutils.hpp"

namespace xt
{

    /*************************
     * xindex_view extension *
     *************************/

    namespace extension
    {
        template <class Tag, class CT, class I>
        struct xindex_view_base_impl;

        template <class CT, class I>
        struct xindex_view_base_impl<xtensor_expression_tag, CT, I>
        {
            using type = xtensor_empty_base;
        };

        template <class CT, class I>
        struct xindex_view_base : xindex_view_base_impl<xexpression_tag_t<CT>, CT, I>
        {
        };

        template <class CT, class I>
        using xindex_view_base_t = typename xindex_view_base<CT, I>::type;
    }

    /***************
     * xindex_view *
     ***************/

    template <class CT, class I>
    class xindex_view;

    template <class CT, class I>
    struct xcontainer_inner_types<xindex_view<CT, I>>
    {
        using xexpression_type = std::decay_t<CT>;
        using temporary_type = xarray<typename xexpression_type::value_type, xexpression_type::static_layout>;
    };

    template <class CT, class I>
    struct xiterable_inner_types<xindex_view<CT, I>>
    {
        using inner_shape_type = std::array<std::size_t, 1>;
        using const_stepper = xindexed_stepper<xindex_view<CT, I>, true>;
        using stepper = xindexed_stepper<xindex_view<CT, I>, false>;
    };

    /**
     * @class xindex_view
     * @brief View of an xexpression from vector of indices.
     *
     * The xindex_view class implements a flat (1D) view into a multidimensional
     * xexpression yielding the values at the indices of the index array.
     * xindex_view is not meant to be used directly, but only with the \ref index_view
     * and \ref filter helper functions.
     *
     * @tparam CT the closure type of the \ref xexpression type underlying this view
     * @tparam I the index array type of the view
     *
     * @sa index_view, filter
     */
    template <class CT, class I>
    class xindex_view : public xview_semantic<xindex_view<CT, I>>,
                        public xiterable<xindex_view<CT, I>>,
                        public extension::xindex_view_base_t<CT, I>
    {
    public:

        using self_type = xindex_view<CT, I>;
        using xexpression_type = std::decay_t<CT>;
        using semantic_base = xview_semantic<self_type>;

        using extension_base = extension::xindex_view_base_t<CT, I>;
        using expression_tag = typename extension_base::expression_tag;

        using value_type = typename xexpression_type::value_type;
        using reference = inner_reference_t<CT>;
        using const_reference = typename xexpression_type::const_reference;
        using pointer = typename xexpression_type::pointer;
        using const_pointer = typename xexpression_type::const_pointer;
        using size_type = typename xexpression_type::size_type;
        using difference_type = typename xexpression_type::difference_type;

        using iterable_base = xiterable<self_type>;
        using inner_shape_type = typename iterable_base::inner_shape_type;
        using shape_type = inner_shape_type;

        using indices_type = I;

        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        using temporary_type = typename xcontainer_inner_types<self_type>::temporary_type;
        using base_index_type = xindex_type_t<shape_type>;

        using bool_load_type = typename xexpression_type::bool_load_type;

        static constexpr layout_type static_layout = layout_type::dynamic;
        static constexpr bool contiguous_layout = false;

        template <class CTA, class I2>
        xindex_view(CTA&& e, I2&& indices) noexcept;

        template <class E>
        self_type& operator=(const xexpression<E>& e);

        template <class E>
        disable_xexpression<E, self_type>& operator=(const E& e);

        size_type size() const noexcept;
        size_type dimension() const noexcept;
        const inner_shape_type& shape() const noexcept;
        size_type shape(size_type index) const;
        layout_type layout() const noexcept;
        bool is_contiguous() const noexcept;

        template <class T>
        void fill(const T& value);

        reference operator()(size_type idx = size_type(0));
        template <class... Args>
        reference operator()(size_type idx0, size_type idx1, Args... args);
        reference unchecked(size_type idx);
        template <class S>
        disable_integral_t<S, reference> operator[](const S& index);
        template <class OI>
        reference operator[](std::initializer_list<OI> index);
        reference operator[](size_type i);

        template <class It>
        reference element(It first, It last);

        const_reference operator()(size_type idx = size_type(0)) const;
        template <class... Args>
        const_reference operator()(size_type idx0, size_type idx1, Args... args) const;
        const_reference unchecked(size_type idx) const;
        template <class S>
        disable_integral_t<S, const_reference> operator[](const S& index) const;
        template <class OI>
        const_reference operator[](std::initializer_list<OI> index) const;
        const_reference operator[](size_type i) const;

        template <class It>
        const_reference element(It first, It last) const;

        xexpression_type& expression() noexcept;
        const xexpression_type& expression() const noexcept;

        template <class O>
        bool broadcast_shape(O& shape, bool reuse_cache = false) const;

        template <class O>
        bool has_linear_assign(const O& /*strides*/) const noexcept;

        template <class ST>
        stepper stepper_begin(const ST& shape);
        template <class ST>
        stepper stepper_end(const ST& shape, layout_type);

        template <class ST>
        const_stepper stepper_begin(const ST& shape) const;
        template <class ST>
        const_stepper stepper_end(const ST& shape, layout_type) const;

        template <class E>
        using rebind_t = xindex_view<E, I>;

        template <class E>
        rebind_t<E> build_index_view(E&& e) const;

    private:

        CT m_e;
        const indices_type m_indices;
        const inner_shape_type m_shape;

        void assign_temporary_impl(temporary_type&& tmp);

        friend class xview_semantic<xindex_view<CT, I>>;
    };

    /***************
     * xfiltration *
     ***************/

    /**
     * @class xfiltration
     * @brief Filter of a xexpression for fast scalar assign.
     *
     * The xfiltration class implements a lazy filtration of a multidimentional
     * \ref xexpression, optimized for scalar and computed scalar assignments.
     * Actually, the \ref xfiltration class IS NOT an \ref xexpression and the
     * scalar and computed scalar assignments are the only method it provides.
     * The filtering condition is not evaluated until the filtration is assigned.
     *
     * xfiltration is not meant to be used directly, but only with the \ref filtration
     * helper function.
     *
     * @tparam ECT the closure type of the \ref xexpression type underlying this filtration
     * @tparam CCR the closure type of the filtering \ref xexpression type
     *
     * @sa filtration
     */
    template <class ECT, class CCT>
    class xfiltration
    {
    public:

        using self_type = xfiltration<ECT, CCT>;
        using xexpression_type = std::decay_t<ECT>;
        using const_reference = typename xexpression_type::const_reference;

        template <class ECTA, class CCTA>
        xfiltration(ECTA&& e, CCTA&& condition);

        template <class E>
        disable_xexpression<E, self_type&> operator=(const E&);

        template <class E>
        disable_xexpression<E, self_type&> operator+=(const E&);

        template <class E>
        disable_xexpression<E, self_type&> operator-=(const E&);

        template <class E>
        disable_xexpression<E, self_type&> operator*=(const E&);

        template <class E>
        disable_xexpression<E, self_type&> operator/=(const E&);

        template <class E>
        disable_xexpression<E, self_type&> operator%=(const E&);

    private:

        template <class F>
        self_type& apply(F&& func);

        ECT m_e;
        CCT m_condition;
    };

    /******************************
     * xindex_view implementation *
     ******************************/

    /**
     * @name Constructor
     */
    //@{
    /**
     * Constructs an xindex_view, selecting the indices specified by \a indices.
     * The resulting xexpression has a 1D shape with a length of n for n indices.
     *
     * @param e the underlying xexpression for this view
     * @param indices the indices to select
     */
    template <class CT, class I>
    template <class CTA, class I2>
    inline xindex_view<CT, I>::xindex_view(CTA&& e, I2&& indices) noexcept
        : m_e(std::forward<CTA>(e))
        , m_indices(std::forward<I2>(indices))
        , m_shape({m_indices.size()})
    {
    }

    //@}

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended assignment operator.
     */
    template <class CT, class I>
    template <class E>
    inline auto xindex_view<CT, I>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }

    //@}

    template <class CT, class I>
    template <class E>
    inline auto xindex_view<CT, I>::operator=(const E& e) -> disable_xexpression<E, self_type>&
    {
        std::fill(this->begin(), this->end(), e);
        return *this;
    }

    template <class CT, class I>
    inline void xindex_view<CT, I>::assign_temporary_impl(temporary_type&& tmp)
    {
        std::copy(tmp.cbegin(), tmp.cend(), this->begin());
    }

    /**
     * @name Size and shape
     */
    //@{
    /**
     * Returns the size of the xindex_view.
     */
    template <class CT, class I>
    inline auto xindex_view<CT, I>::size() const noexcept -> size_type
    {
        return compute_size(shape());
    }

    /**
     * Returns the number of dimensions of the xindex_view.
     */
    template <class CT, class I>
    inline auto xindex_view<CT, I>::dimension() const noexcept -> size_type
    {
        return 1;
    }

    /**
     * Returns the shape of the xindex_view.
     */
    template <class CT, class I>
    inline auto xindex_view<CT, I>::shape() const noexcept -> const inner_shape_type&
    {
        return m_shape;
    }

    /**
     * Returns the i-th dimension of the expression.
     */
    template <class CT, class I>
    inline auto xindex_view<CT, I>::shape(size_type i) const -> size_type
    {
        return m_shape[i];
    }

    template <class CT, class I>
    inline layout_type xindex_view<CT, I>::layout() const noexcept
    {
        return static_layout;
    }

    template <class CT, class I>
    inline bool xindex_view<CT, I>::is_contiguous() const noexcept
    {
        return false;
    }

    //@}

    /**
     * @name Data
     */
    //@{

    /**
     * Fills the view with the given value.
     * @param value the value to fill the view with.
     */
    template <class CT, class I>
    template <class T>
    inline void xindex_view<CT, I>::fill(const T& value)
    {
        std::fill(this->begin(), this->end(), value);
    }

    /**
     * Returns a reference to the element at the specified position in the xindex_view.
     * @param idx index specifying the position in the index_view. More indices may be provided,
     * only the last one will be used.
     */
    template <class CT, class I>
    inline auto xindex_view<CT, I>::operator()(size_type idx) -> reference
    {
        return m_e[m_indices[idx]];
    }

    template <class CT, class I>
    template <class... Args>
    inline auto xindex_view<CT, I>::operator()(size_type, size_type idx1, Args... args) -> reference
    {
        return this->operator()(idx1, static_cast<size_type>(args)...);
    }

    /**
     * Returns a reference to the element at the specified position in the xindex_view.
     * @param idx index specifying the position in the index_view.
     */
    template <class CT, class I>
    inline auto xindex_view<CT, I>::unchecked(size_type idx) -> reference
    {
        return this->operator()(idx);
    }

    /**
     * Returns a constant reference to the element at the specified position in the xindex_view.
     * @param idx index specifying the position in the index_view. More indices may be provided,
     * only the last one will be used.
     */
    template <class CT, class I>
    inline auto xindex_view<CT, I>::operator()(size_type idx) const -> const_reference
    {
        return m_e[m_indices[idx]];
    }

    template <class CT, class I>
    template <class... Args>
    inline auto xindex_view<CT, I>::operator()(size_type, size_type idx1, Args... args) const -> const_reference
    {
        return this->operator()(idx1, args...);
    }

    /**
     * Returns a constant reference to the element at the specified position in the xindex_view.
     * @param idx index specifying the position in the index_view.
     */
    template <class CT, class I>
    inline auto xindex_view<CT, I>::unchecked(size_type idx) const -> const_reference
    {
        return this->operator()(idx);
    }

    /**
     * Returns a reference to the element at the specified position in the container.
     * @param index a sequence of indices specifying the position in the container. Indices
     * must be unsigned integers, the number of indices in the list should be equal or greater
     * than the number of dimensions of the container.
     */
    template <class CT, class I>
    template <class S>
    inline auto xindex_view<CT, I>::operator[](const S& index) -> disable_integral_t<S, reference>
    {
        return m_e[m_indices[index[0]]];
    }

    template <class CT, class I>
    template <class OI>
    inline auto xindex_view<CT, I>::operator[](std::initializer_list<OI> index) -> reference
    {
        return m_e[m_indices[*(index.begin())]];
    }

    template <class CT, class I>
    inline auto xindex_view<CT, I>::operator[](size_type i) -> reference
    {
        return operator()(i);
    }

    /**
     * Returns a constant reference to the element at the specified position in the container.
     * @param index a sequence of indices specifying the position in the container. Indices
     * must be unsigned integers, the number of indices in the list should be equal or greater
     * than the number of dimensions of the container.
     */
    template <class CT, class I>
    template <class S>
    inline auto xindex_view<CT, I>::operator[](const S& index) const -> disable_integral_t<S, const_reference>
    {
        return m_e[m_indices[index[0]]];
    }

    template <class CT, class I>
    template <class OI>
    inline auto xindex_view<CT, I>::operator[](std::initializer_list<OI> index) const -> const_reference
    {
        return m_e[m_indices[*(index.begin())]];
    }

    template <class CT, class I>
    inline auto xindex_view<CT, I>::operator[](size_type i) const -> const_reference
    {
        return operator()(i);
    }

    /**
     * Returns a reference to the element at the specified position in the xindex_view.
     * @param first iterator starting the sequence of indices
     * The number of indices in the sequence should be equal to or greater 1.
     */
    template <class CT, class I>
    template <class It>
    inline auto xindex_view<CT, I>::element(It first, It /*last*/) -> reference
    {
        return m_e[m_indices[(*first)]];
    }

    /**
     * Returns a reference to the element at the specified position in the xindex_view.
     * @param first iterator starting the sequence of indices
     * The number of indices in the sequence should be equal to or greater 1.
     */
    template <class CT, class I>
    template <class It>
    inline auto xindex_view<CT, I>::element(It first, It /*last*/) const -> const_reference
    {
        return m_e[m_indices[(*first)]];
    }

    /**
     * Returns a reference to the underlying expression of the view.
     */
    template <class CT, class I>
    inline auto xindex_view<CT, I>::expression() noexcept -> xexpression_type&
    {
        return m_e;
    }

    /**
     * Returns a constant reference to the underlying expression of the view.
     */
    template <class CT, class I>
    inline auto xindex_view<CT, I>::expression() const noexcept -> const xexpression_type&
    {
        return m_e;
    }

    //@}

    /**
     * @name Broadcasting
     */
    //@{
    /**
     * Broadcast the shape of the xindex_view to the specified parameter.
     * @param shape the result shape
     * @param reuse_cache parameter for internal optimization
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class CT, class I>
    template <class O>
    inline bool xindex_view<CT, I>::broadcast_shape(O& shape, bool) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    /**
     * Checks whether the xindex_view can be linearly assigned to an expression
     * with the specified strides.
     * @return a boolean indicating whether a linear assign is possible
     */
    template <class CT, class I>
    template <class O>
    inline bool xindex_view<CT, I>::has_linear_assign(const O& /*strides*/) const noexcept
    {
        return false;
    }

    //@}

    /***************
     * stepper api *
     ***************/

    template <class CT, class I>
    template <class ST>
    inline auto xindex_view<CT, I>::stepper_begin(const ST& shape) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(this, offset);
    }

    template <class CT, class I>
    template <class ST>
    inline auto xindex_view<CT, I>::stepper_end(const ST& shape, layout_type) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(this, offset, true);
    }

    template <class CT, class I>
    template <class ST>
    inline auto xindex_view<CT, I>::stepper_begin(const ST& shape) const -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, offset);
    }

    template <class CT, class I>
    template <class ST>
    inline auto xindex_view<CT, I>::stepper_end(const ST& shape, layout_type) const -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, offset, true);
    }

    template <class CT, class I>
    template <class E>
    inline auto xindex_view<CT, I>::build_index_view(E&& e) const -> rebind_t<E>
    {
        return rebind_t<E>(std::forward<E>(e), indices_type(m_indices));
    }

    /******************************
     * xfiltration implementation *
     ******************************/

    /**
     * @name Constructor
     */
    //@{
    /**
     * Constructs a xfiltration on the given expression \c e, selecting
     * the elements matching the specified \c condition.
     *
     * @param e the \ref xexpression to filter.
     * @param condition the filtering \ref xexpression to apply.
     */
    template <class ECT, class CCT>
    template <class ECTA, class CCTA>
    inline xfiltration<ECT, CCT>::xfiltration(ECTA&& e, CCTA&& condition)
        : m_e(std::forward<ECTA>(e))
        , m_condition(std::forward<CCTA>(condition))
    {
    }

    //@}

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * Assigns the scalar \c e to \c *this.
     * @param e the scalar to assign.
     * @return a reference to \ *this.
     */
    template <class ECT, class CCT>
    template <class E>
    inline auto xfiltration<ECT, CCT>::operator=(const E& e) -> disable_xexpression<E, self_type&>
    {
        return apply(
            [this, &e](const_reference v, bool cond)
            {
                return cond ? e : v;
            }
        );
    }

    //@}

    /**
     * @name Computed assignement
     */
    //@{
    /**
     * Adds the scalar \c e to \c *this.
     * @param e the scalar to add.
     * @return a reference to \c *this.
     */
    template <class ECT, class CCT>
    template <class E>
    inline auto xfiltration<ECT, CCT>::operator+=(const E& e) -> disable_xexpression<E, self_type&>
    {
        return apply(
            [&e](const_reference v, bool cond)
            {
                return cond ? v + e : v;
            }
        );
    }

    /**
     * Subtracts the scalar \c e from \c *this.
     * @param e the scalar to subtract.
     * @return a reference to \c *this.
     */
    template <class ECT, class CCT>
    template <class E>
    inline auto xfiltration<ECT, CCT>::operator-=(const E& e) -> disable_xexpression<E, self_type&>
    {
        return apply(
            [&e](const_reference v, bool cond)
            {
                return cond ? v - e : v;
            }
        );
    }

    /**
     * Multiplies \c *this with the scalar \c e.
     * @param e the scalar involved in the operation.
     * @return a reference to \c *this.
     */
    template <class ECT, class CCT>
    template <class E>
    inline auto xfiltration<ECT, CCT>::operator*=(const E& e) -> disable_xexpression<E, self_type&>
    {
        return apply(
            [&e](const_reference v, bool cond)
            {
                return cond ? v * e : v;
            }
        );
    }

    /**
     * Divides \c *this by the scalar \c e.
     * @param e the scalar involved in the operation.
     * @return a reference to \c *this.
     */
    template <class ECT, class CCT>
    template <class E>
    inline auto xfiltration<ECT, CCT>::operator/=(const E& e) -> disable_xexpression<E, self_type&>
    {
        return apply(
            [&e](const_reference v, bool cond)
            {
                return cond ? v / e : v;
            }
        );
    }

    /**
     * Computes the remainder of \c *this after division by the scalar \c e.
     * @param e the scalar involved in the operation.
     * @return a reference to \c *this.
     */
    template <class ECT, class CCT>
    template <class E>
    inline auto xfiltration<ECT, CCT>::operator%=(const E& e) -> disable_xexpression<E, self_type&>
    {
        return apply(
            [&e](const_reference v, bool cond)
            {
                return cond ? v % e : v;
            }
        );
    }

    template <class ECT, class CCT>
    template <class F>
    inline auto xfiltration<ECT, CCT>::apply(F&& func) -> self_type&
    {
        std::transform(m_e.cbegin(), m_e.cend(), m_condition.cbegin(), m_e.begin(), func);
        return *this;
    }

    /**
     * @brief creates an indexview from a container of indices.
     *
     * Returns a 1D view with the elements at \a indices selected.
     *
     * @param e the underlying xexpression
     * @param indices the indices to select
     *
     * @code{.cpp}
     * xarray<double> a = {{1,5,3}, {4,5,6}};
     * b = index_view(a, {{0, 0}, {1, 0}, {1, 1}});
     * std::cout << b << std::endl; // {1, 4, 5}
     * b += 100;
     * std::cout << a << std::endl; // {{101, 5, 3}, {104, 105, 6}}
     * @endcode
     */
    template <class E, class I>
    inline auto index_view(E&& e, I&& indices) noexcept
    {
        using view_type = xindex_view<xclosure_t<E>, std::decay_t<I>>;
        return view_type(std::forward<E>(e), std::forward<I>(indices));
    }

    template <class E, std::size_t L>
    inline auto index_view(E&& e, const xindex (&indices)[L]) noexcept
    {
        using view_type = xindex_view<xclosure_t<E>, std::array<xindex, L>>;
        return view_type(std::forward<E>(e), xt::to_array(indices));
    }

    /**
     * @brief creates a view into \a e filtered by \a condition.
     *
     * Returns a 1D view with the elements selected where \a condition evaluates to \em true.
     * This is equivalent to \verbatim{index_view(e, argwhere(condition));}\endverbatim
     * The returned view is not optimal if you just want to assign a scalar to the filtered
     * elements. In that case, you should consider using the \ref filtration function
     * instead.
     *
     * @tparam L the traversal order
     * @param e the underlying xexpression
     * @param condition xexpression with shape of \a e which selects indices
     *
     * @code{.cpp}
     * xarray<double> a = {{1,5,3}, {4,5,6}};
     * b = filter(a, a >= 5);
     * std::cout << b << std::endl; // {5, 5, 6}
     * @endcode
     *
     * \sa filtration
     */
    template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class E, class O>
    inline auto filter(E&& e, O&& condition) noexcept
    {
        auto indices = argwhere<L>(std::forward<O>(condition));
        using view_type = xindex_view<xclosure_t<E>, decltype(indices)>;
        return view_type(std::forward<E>(e), std::move(indices));
    }

    /**
     * @brief creates a filtration of \c e filtered by \a condition.
     *
     * Returns a lazy filtration optimized for scalar assignment.
     * Actually, scalar assignment and computed scalar assignments
     * are the only available methods of the filtration, the filtration
     * IS NOT an \ref xexpression.
     *
     * @param e the \ref xexpression to filter
     * @param condition the filtering \ref xexpression
     *
     * @code{.cpp}
     * xarray<double> a = {{1,5,3}, {4,5,6}};
     * filtration(a, a >= 5) += 2;
     * std::cout << a << std::endl; // {{1, 7, 3}, {4, 7, 8}}
     * @endcode
     */
    template <class E, class C>
    inline auto filtration(E&& e, C&& condition) noexcept
    {
        using filtration_type = xfiltration<xclosure_t<E>, xclosure_t<C>>;
        return filtration_type(std::forward<E>(e), std::forward<C>(condition));
    }
}

#endif
