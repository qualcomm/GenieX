/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_BROADCAST_HPP
#define XTENSOR_BROADCAST_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>

#include <xtl/xsequence.hpp>

#include "../containers/xscalar.hpp"
#include "../core/xaccessible.hpp"
#include "../core/xexpression.hpp"
#include "../core/xiterable.hpp"
#include "../core/xstrides.hpp"
#include "../core/xtensor_config.hpp"
#include "../utils/xutils.hpp"

namespace xt
{

    /*************
     * broadcast *
     *************/

    template <class E, class S>
    auto broadcast(E&& e, const S& s);

    template <class E, class I, std::size_t L>
    auto broadcast(E&& e, const I (&s)[L]);

    /*************************
     * xbroadcast extensions *
     *************************/

    namespace extension
    {
        template <class Tag, class CT, class X>
        struct xbroadcast_base_impl;

        template <class CT, class X>
        struct xbroadcast_base_impl<xtensor_expression_tag, CT, X>
        {
            using type = xtensor_empty_base;
        };

        template <class CT, class X>
        struct xbroadcast_base : xbroadcast_base_impl<xexpression_tag_t<CT>, CT, X>
        {
        };

        template <class CT, class X>
        using xbroadcast_base_t = typename xbroadcast_base<CT, X>::type;
    }

    /**************
     * xbroadcast *
     **************/

    template <class CT, class X>
    class xbroadcast;

    template <class CT, class X>
    struct xiterable_inner_types<xbroadcast<CT, X>>
    {
        using xexpression_type = std::decay_t<CT>;
        using inner_shape_type = promote_shape_t<typename xexpression_type::shape_type, X>;
        using const_stepper = typename xexpression_type::const_stepper;
        using stepper = const_stepper;
    };

    template <class CT, class X>
    struct xcontainer_inner_types<xbroadcast<CT, X>>
    {
        using xexpression_type = std::decay_t<CT>;
        using reference = typename xexpression_type::const_reference;
        using const_reference = typename xexpression_type::const_reference;
        using size_type = typename xexpression_type::size_type;
    };

    /*****************************
     * linear_begin / linear_end *
     *****************************/

    template <class CT, class X>
    XTENSOR_CONSTEXPR_RETURN auto linear_begin(xbroadcast<CT, X>& c) noexcept
    {
        return linear_begin(c.expression());
    }

    template <class CT, class X>
    XTENSOR_CONSTEXPR_RETURN auto linear_end(xbroadcast<CT, X>& c) noexcept
    {
        return linear_end(c.expression());
    }

    template <class CT, class X>
    XTENSOR_CONSTEXPR_RETURN auto linear_begin(const xbroadcast<CT, X>& c) noexcept
    {
        return linear_begin(c.expression());
    }

    template <class CT, class X>
    XTENSOR_CONSTEXPR_RETURN auto linear_end(const xbroadcast<CT, X>& c) noexcept
    {
        return linear_end(c.expression());
    }

    /*************************************
     * overlapping_memory_checker_traits *
     *************************************/

    template <class E>
    struct overlapping_memory_checker_traits<
        E,
        std::enable_if_t<!has_memory_address<E>::value && is_specialization_of<xbroadcast, E>::value>>
    {
        static bool check_overlap(const E& expr, const memory_range& dst_range)
        {
            if (expr.size() == 0)
            {
                return false;
            }
            else
            {
                using ChildE = std::decay_t<decltype(expr.expression())>;
                return overlapping_memory_checker_traits<ChildE>::check_overlap(expr.expression(), dst_range);
            }
        }
    };

    /**
     * @class xbroadcast
     * @brief Broadcasted xexpression to a specified shape.
     *
     * The xbroadcast class implements the broadcasting of an \ref xexpression
     * to a specified shape. xbroadcast is not meant to be used directly, but
     * only with the \ref broadcast helper functions.
     *
     * @tparam CT the closure type of the \ref xexpression to broadcast
     * @tparam X the type of the specified shape.
     *
     * @sa broadcast
     */
    template <class CT, class X>
    class xbroadcast : public xsharable_expression<xbroadcast<CT, X>>,
                       public xconst_iterable<xbroadcast<CT, X>>,
                       public xconst_accessible<xbroadcast<CT, X>>,
                       public extension::xbroadcast_base_t<CT, X>
    {
    public:

        using self_type = xbroadcast<CT, X>;
        using xexpression_type = std::decay_t<CT>;
        using accessible_base = xconst_accessible<self_type>;
        using extension_base = extension::xbroadcast_base_t<CT, X>;
        using expression_tag = typename extension_base::expression_tag;

        using inner_types = xcontainer_inner_types<self_type>;
        using value_type = typename xexpression_type::value_type;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;
        using pointer = typename xexpression_type::const_pointer;
        using const_pointer = typename xexpression_type::const_pointer;
        using size_type = typename inner_types::size_type;
        using difference_type = typename xexpression_type::difference_type;

        using iterable_base = xconst_iterable<self_type>;
        using inner_shape_type = typename iterable_base::inner_shape_type;
        using shape_type = inner_shape_type;

        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        using bool_load_type = typename xexpression_type::bool_load_type;

        static constexpr layout_type static_layout = layout_type::dynamic;
        static constexpr bool contiguous_layout = false;

        template <class CTA, class S>
        xbroadcast(CTA&& e, const S& s);

        template <class CTA>
        xbroadcast(CTA&& e, shape_type&& s);

        using accessible_base::size;
        const inner_shape_type& shape() const noexcept;
        layout_type layout() const noexcept;
        bool is_contiguous() const noexcept;
        using accessible_base::shape;

        template <class... Args>
        const_reference operator()(Args... args) const;

        template <class... Args>
        const_reference unchecked(Args... args) const;

        template <class It>
        const_reference element(It first, It last) const;

        const xexpression_type& expression() const noexcept;

        template <class S>
        bool broadcast_shape(S& shape, bool reuse_cache = false) const;

        template <class S>
        bool has_linear_assign(const S& strides) const noexcept;

        template <class S>
        const_stepper stepper_begin(const S& shape) const noexcept;
        template <class S>
        const_stepper stepper_end(const S& shape, layout_type l) const noexcept;

        template <class E, class XCT = CT, class = std::enable_if_t<xt::is_xscalar<XCT>::value>>
        void assign_to(xexpression<E>& e) const;

        template <class E>
        using rebind_t = xbroadcast<E, X>;

        template <class E>
        rebind_t<E> build_broadcast(E&& e) const;

    private:

        CT m_e;
        inner_shape_type m_shape;
    };

    /****************************
     * broadcast implementation *
     ****************************/

    /**
     * @brief Returns an \ref xexpression broadcasting the given expression to
     * a specified shape.
     *
     * @tparam e the \ref xexpression to broadcast
     * @tparam s the specified shape to broadcast.
     *
     * The returned expression either hold a const reference to \p e or a copy
     * depending on whether \p e is an lvalue or an rvalue.
     */
    template <class E, class S>
    inline auto broadcast(E&& e, const S& s)
    {
        using shape_type = filter_fixed_shape_t<std::decay_t<S>>;
        using broadcast_type = xbroadcast<const_xclosure_t<E>, shape_type>;
        return broadcast_type(std::forward<E>(e), xtl::forward_sequence<shape_type, decltype(s)>(s));
    }

    template <class E, class I, std::size_t L>
    inline auto broadcast(E&& e, const I (&s)[L])
    {
        using broadcast_type = xbroadcast<const_xclosure_t<E>, std::array<std::size_t, L>>;
        using shape_type = typename broadcast_type::shape_type;
        return broadcast_type(std::forward<E>(e), xtl::forward_sequence<shape_type, decltype(s)>(s));
    }

    /*****************************
     * xbroadcast implementation *
     *****************************/

    /**
     * @name Constructor
     */
    //@{
    /**
     * Constructs an xbroadcast expression broadcasting the specified
     * \ref xexpression to the given shape
     *
     * @param e the expression to broadcast
     * @param s the shape to apply
     */
    template <class CT, class X>
    template <class CTA, class S>
    inline xbroadcast<CT, X>::xbroadcast(CTA&& e, const S& s)
        : m_e(std::forward<CTA>(e))
    {
        if (s.size() < m_e.dimension())
        {
            XTENSOR_THROW(xt::broadcast_error, "Broadcast shape has fewer elements than original expression.");
        }
        xt::resize_container(m_shape, s.size());
        std::copy(s.begin(), s.end(), m_shape.begin());
        xt::broadcast_shape(m_e.shape(), m_shape);
    }

    /**
     * Constructs an xbroadcast expression broadcasting the specified
     * \ref xexpression to the given shape
     *
     * @param e the expression to broadcast
     * @param s the shape to apply
     */
    template <class CT, class X>
    template <class CTA>
    inline xbroadcast<CT, X>::xbroadcast(CTA&& e, shape_type&& s)
        : m_e(std::forward<CTA>(e))
        , m_shape(std::move(s))
    {
        xt::broadcast_shape(m_e.shape(), m_shape);
    }

    //@}

    /**
     * @name Size and shape
     */
    //@{
    /**
     * Returns the shape of the expression.
     */
    template <class CT, class X>
    inline auto xbroadcast<CT, X>::shape() const noexcept -> const inner_shape_type&
    {
        return m_shape;
    }

    /**
     * Returns the layout_type of the expression.
     */
    template <class CT, class X>
    inline layout_type xbroadcast<CT, X>::layout() const noexcept
    {
        return m_e.layout();
    }

    template <class CT, class X>
    inline bool xbroadcast<CT, X>::is_contiguous() const noexcept
    {
        return false;
    }

    //@}

    /**
     * @name Data
     */
    //@{
    /**
     * Returns a constant reference to the element at the specified position in the expression.
     * @param args a list of indices specifying the position in the function. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the expression.
     */
    template <class CT, class X>
    template <class... Args>
    inline auto xbroadcast<CT, X>::operator()(Args... args) const -> const_reference
    {
        return m_e(args...);
    }

    /**
     * Returns a constant reference to the element at the specified position in the expression.
     * @param args a list of indices specifying the position in the expression. Indices
     * must be unsigned integers, the number of indices must be equal to the number of
     * dimensions of the expression, else the behavior is undefined.
     *
     * @warning This method is meant for performance, for expressions with a dynamic
     * number of dimensions (i.e. not known at compile time). Since it may have
     * undefined behavior (see parameters), operator() should be preferred whenever
     * it is possible.
     * @warning This method is NOT compatible with broadcasting, meaning the following
     * code has undefined behavior:
     * @code{.cpp}
     * xt::xarray<double> a = {{0, 1}, {2, 3}};
     * xt::xarray<double> b = {0, 1};
     * auto fd = a + b;
     * double res = fd.uncheked(0, 1);
     * @endcode
     */
    template <class CT, class X>
    template <class... Args>
    inline auto xbroadcast<CT, X>::unchecked(Args... args) const -> const_reference
    {
        return this->operator()(args...);
    }

    /**
     * Returns a constant reference to the element at the specified position in the expression.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater
     * than the number of dimensions of the function.
     */
    template <class CT, class X>
    template <class It>
    inline auto xbroadcast<CT, X>::element(It, It last) const -> const_reference
    {
        return m_e.element(last - this->dimension(), last);
    }

    /**
     * Returns a constant reference to the underlying expression of the broadcast expression.
     */
    template <class CT, class X>
    inline auto xbroadcast<CT, X>::expression() const noexcept -> const xexpression_type&
    {
        return m_e;
    }

    //@}

    /**
     * @name Broadcasting
     */
    //@{
    /**
     * Broadcast the shape of the function to the specified parameter.
     * @param shape the result shape
     * @param reuse_cache parameter for internal optimization
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class CT, class X>
    template <class S>
    inline bool xbroadcast<CT, X>::broadcast_shape(S& shape, bool) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    /**
     * Checks whether the xbroadcast can be linearly assigned to an expression
     * with the specified strides.
     * @return a boolean indicating whether a linear assign is possible
     */
    template <class CT, class X>
    template <class S>
    inline bool xbroadcast<CT, X>::has_linear_assign(const S& strides) const noexcept
    {
        return this->dimension() == m_e.dimension()
               && std::equal(m_shape.cbegin(), m_shape.cend(), m_e.shape().cbegin())
               && m_e.has_linear_assign(strides);
    }

    //@}

    template <class CT, class X>
    template <class S>
    inline auto xbroadcast<CT, X>::stepper_begin(const S& shape) const noexcept -> const_stepper
    {
        // Could check if (broadcastable(shape, m_shape)
        return m_e.stepper_begin(shape);
    }

    template <class CT, class X>
    template <class S>
    inline auto xbroadcast<CT, X>::stepper_end(const S& shape, layout_type l) const noexcept -> const_stepper
    {
        // Could check if (broadcastable(shape, m_shape)
        return m_e.stepper_end(shape, l);
    }

    template <class CT, class X>
    template <class E, class XCT, class>
    inline void xbroadcast<CT, X>::assign_to(xexpression<E>& e) const
    {
        auto& ed = e.derived_cast();
        ed.resize(m_shape);
        std::fill(ed.begin(), ed.end(), m_e());
    }

    template <class CT, class X>
    template <class E>
    inline auto xbroadcast<CT, X>::build_broadcast(E&& e) const -> rebind_t<E>
    {
        return rebind_t<E>(std::forward<E>(e), inner_shape_type(m_shape));
    }
}

#endif
