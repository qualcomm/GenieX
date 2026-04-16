/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_SEMANTIC_HPP
#define XTENSOR_SEMANTIC_HPP

#include <functional>
#include <utility>

#include "../core/xassign.hpp"
#include "../core/xexpression_traits.hpp"

namespace xt
{
    namespace detail
    {
        template <class D>
        struct is_sharable
        {
            static constexpr bool value = true;
        };

        template <class ET, class S, layout_type L, bool SH, class Tag>
        struct is_sharable<xfixed_container<ET, S, L, SH, Tag>>
        {
            static constexpr bool value = SH;
        };

        template <class ET, class S, layout_type L, bool SH, class Tag>
        struct is_sharable<xfixed_adaptor<ET, S, L, SH, Tag>>
        {
            static constexpr bool value = SH;
        };
    }

    template <class D>
    using select_expression_base_t = std::
        conditional_t<detail::is_sharable<D>::value, xsharable_expression<D>, xexpression<D>>;

    /**
     * @class xsemantic_base
     * @brief Base interface for assignable xexpressions.
     *
     * The xsemantic_base class defines the interface for assignable
     * xexpressions.
     *
     * @tparam D The derived type, i.e. the inheriting class for which xsemantic_base
     *           provides the interface.
     */
    template <class D>
    class xsemantic_base : public select_expression_base_t<D>
    {
    public:

        using base_type = select_expression_base_t<D>;
        using derived_type = typename base_type::derived_type;

        using temporary_type = typename xcontainer_inner_types<D>::temporary_type;

        template <class E>
        disable_xexpression<E, derived_type&> operator+=(const E&);

        template <class E>
        disable_xexpression<E, derived_type&> operator-=(const E&);

        template <class E>
        disable_xexpression<E, derived_type&> operator*=(const E&);

        template <class E>
        disable_xexpression<E, derived_type&> operator/=(const E&);

        template <class E>
        disable_xexpression<E, derived_type&> operator%=(const E&);

        template <class E>
        disable_xexpression<E, derived_type&> operator&=(const E&);

        template <class E>
        disable_xexpression<E, derived_type&> operator|=(const E&);

        template <class E>
        disable_xexpression<E, derived_type&> operator^=(const E&);

        template <class E>
        derived_type& operator+=(const xexpression<E>&);

        template <class E>
        derived_type& operator-=(const xexpression<E>&);

        template <class E>
        derived_type& operator*=(const xexpression<E>&);

        template <class E>
        derived_type& operator/=(const xexpression<E>&);

        template <class E>
        derived_type& operator%=(const xexpression<E>&);

        template <class E>
        derived_type& operator&=(const xexpression<E>&);

        template <class E>
        derived_type& operator|=(const xexpression<E>&);

        template <class E>
        derived_type& operator^=(const xexpression<E>&);

        template <class E>
        derived_type& assign(const xexpression<E>&);

        template <class E>
        derived_type& plus_assign(const xexpression<E>&);

        template <class E>
        derived_type& minus_assign(const xexpression<E>&);

        template <class E>
        derived_type& multiplies_assign(const xexpression<E>&);

        template <class E>
        derived_type& divides_assign(const xexpression<E>&);

        template <class E>
        derived_type& modulus_assign(const xexpression<E>&);

        template <class E>
        derived_type& bit_and_assign(const xexpression<E>&);

        template <class E>
        derived_type& bit_or_assign(const xexpression<E>&);

        template <class E>
        derived_type& bit_xor_assign(const xexpression<E>&);

    protected:

        xsemantic_base() = default;
        ~xsemantic_base() = default;

        xsemantic_base(const xsemantic_base&) = default;
        xsemantic_base& operator=(const xsemantic_base&) = default;

        xsemantic_base(xsemantic_base&&) = default;
        xsemantic_base& operator=(xsemantic_base&&) = default;

        template <class E>
        derived_type& operator=(const xexpression<E>&);
    };

    template <class E>
    using is_assignable = is_crtp_base_of<xsemantic_base, E>;

    template <class E, class R = void>
    using enable_assignable = typename std::enable_if<is_assignable<E>::value, R>::type;

    template <class E, class R = void>
    using disable_assignable = typename std::enable_if<!is_assignable<E>::value, R>::type;

    /**
     * @class xcontainer_semantic
     * @brief Implementation of the xsemantic_base interface
     * for dense multidimensional containers.
     *
     * The xcontainer_semantic class is an implementation of the
     * xsemantic_base interface for dense multidimensional
     * containers.
     *
     * @tparam D the derived type
     */
    template <class D>
    class xcontainer_semantic : public xsemantic_base<D>
    {
    public:

        using base_type = xsemantic_base<D>;
        using derived_type = D;
        using temporary_type = typename base_type::temporary_type;

        derived_type& assign_temporary(temporary_type&&);

        template <class E>
        derived_type& assign_xexpression(const xexpression<E>& e);

        template <class E>
        derived_type& computed_assign(const xexpression<E>& e);

        template <class E, class F>
        derived_type& scalar_computed_assign(const E& e, F&& f);

    protected:

        xcontainer_semantic() = default;
        ~xcontainer_semantic() = default;

        xcontainer_semantic(const xcontainer_semantic&) = default;
        xcontainer_semantic& operator=(const xcontainer_semantic&) = default;

        xcontainer_semantic(xcontainer_semantic&&) = default;
        xcontainer_semantic& operator=(xcontainer_semantic&&) = default;

        template <class E>
        derived_type& operator=(const xexpression<E>&);
    };

    template <class E>
    using has_container_semantics = is_crtp_base_of<xcontainer_semantic, E>;

    template <class E, class R = void>
    using enable_xcontainer_semantics = typename std::enable_if<has_container_semantics<E>::value, R>::type;

    template <class E, class R = void>
    using disable_xcontainer_semantics = typename std::enable_if<!has_container_semantics<E>::value, R>::type;


    template <class D>
    class xview_semantic;

    template <class E>
    struct overlapping_memory_checker_traits<
        E,
        std::enable_if_t<!has_memory_address<E>::value && is_crtp_base_of<xview_semantic, E>::value>>
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
     * @class xview_semantic
     * @brief Implementation of the xsemantic_base interface for
     * multidimensional views
     *
     * The xview_semantic is an implementation of the xsemantic_base
     * interface for multidimensional views.
     *
     * @tparam D the derived type
     */
    template <class D>
    class xview_semantic : public xsemantic_base<D>
    {
    public:

        using base_type = xsemantic_base<D>;
        using derived_type = D;
        using temporary_type = typename base_type::temporary_type;

        derived_type& assign_temporary(temporary_type&&);

        template <class E>
        derived_type& assign_xexpression(const xexpression<E>& e);

        template <class E>
        derived_type& computed_assign(const xexpression<E>& e);

        template <class E, class F>
        derived_type& scalar_computed_assign(const E& e, F&& f);

    protected:

        xview_semantic() = default;
        ~xview_semantic() = default;

        xview_semantic(const xview_semantic&) = default;
        xview_semantic& operator=(const xview_semantic&) = default;

        xview_semantic(xview_semantic&&) = default;
        xview_semantic& operator=(xview_semantic&&) = default;

        template <class E>
        derived_type& operator=(const xexpression<E>&);
    };

    template <class E>
    using has_view_semantics = is_crtp_base_of<xview_semantic, E>;

    template <class E, class R = void>
    using enable_xview_semantics = typename std::enable_if<has_view_semantics<E>::value, R>::type;

    template <class E, class R = void>
    using disable_xview_semantics = typename std::enable_if<!has_view_semantics<E>::value, R>::type;

    /*********************************
     * xsemantic_base implementation *
     *********************************/

    /**
     * @name Computed assignement
     */
    //@{
    /**
     * Adds the scalar \c e to \c *this.
     * @param e the scalar to add.
     * @return a reference to \c *this.
     */
    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator+=(const E& e) -> disable_xexpression<E, derived_type&>
    {
        return this->derived_cast().scalar_computed_assign(e, std::plus<>());
    }

    /**
     * Subtracts the scalar \c e from \c *this.
     * @param e the scalar to subtract.
     * @return a reference to \c *this.
     */
    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator-=(const E& e) -> disable_xexpression<E, derived_type&>
    {
        return this->derived_cast().scalar_computed_assign(e, std::minus<>());
    }

    /**
     * Multiplies \c *this with the scalar \c e.
     * @param e the scalar involved in the operation.
     * @return a reference to \c *this.
     */
    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator*=(const E& e) -> disable_xexpression<E, derived_type&>
    {
        return this->derived_cast().scalar_computed_assign(e, std::multiplies<>());
    }

    /**
     * Divides \c *this by the scalar \c e.
     * @param e the scalar involved in the operation.
     * @return a reference to \c *this.
     */
    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator/=(const E& e) -> disable_xexpression<E, derived_type&>
    {
        return this->derived_cast().scalar_computed_assign(e, std::divides<>());
    }

    /**
     * Computes the remainder of \c *this after division by the scalar \c e.
     * @param e the scalar involved in the operation.
     * @return a reference to \c *this.
     */
    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator%=(const E& e) -> disable_xexpression<E, derived_type&>
    {
        return this->derived_cast().scalar_computed_assign(e, std::modulus<>());
    }

    /**
     * Computes the bitwise and of \c *this and the scalar \c e and assigns it to \c *this.
     * @param e the scalar involved in the operation.
     * @return a reference to \c *this.
     */
    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator&=(const E& e) -> disable_xexpression<E, derived_type&>
    {
        return this->derived_cast().scalar_computed_assign(e, std::bit_and<>());
    }

    /**
     * Computes the bitwise or of \c *this and the scalar \c e and assigns it to \c *this.
     * @param e the scalar involved in the operation.
     * @return a reference to \c *this.
     */
    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator|=(const E& e) -> disable_xexpression<E, derived_type&>
    {
        return this->derived_cast().scalar_computed_assign(e, std::bit_or<>());
    }

    /**
     * Computes the bitwise xor of \c *this and the scalar \c e and assigns it to \c *this.
     * @param e the scalar involved in the operation.
     * @return a reference to \c *this.
     */
    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator^=(const E& e) -> disable_xexpression<E, derived_type&>
    {
        return this->derived_cast().scalar_computed_assign(e, std::bit_xor<>());
    }

    /**
     * Adds the xexpression \c e to \c *this.
     * @param e the xexpression to add.
     * @return a reference to \c *this.
     */
    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator+=(const xexpression<E>& e) -> derived_type&
    {
        return this->derived_cast() = this->derived_cast() + e.derived_cast();
    }

    /**
     * Subtracts the xexpression \c e from \c *this.
     * @param e the xexpression to subtract.
     * @return a reference to \c *this.
     */
    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator-=(const xexpression<E>& e) -> derived_type&
    {
        return this->derived_cast() = this->derived_cast() - e.derived_cast();
    }

    /**
     * Multiplies \c *this with the xexpression \c e.
     * @param e the xexpression involved in the operation.
     * @return a reference to \c *this.
     */
    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator*=(const xexpression<E>& e) -> derived_type&
    {
        return this->derived_cast() = this->derived_cast() * e.derived_cast();
    }

    /**
     * Divides \c *this by the xexpression \c e.
     * @param e the xexpression involved in the operation.
     * @return a reference to \c *this.
     */
    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator/=(const xexpression<E>& e) -> derived_type&
    {
        return this->derived_cast() = this->derived_cast() / e.derived_cast();
    }

    /**
     * Computes the remainder of \c *this after division by the xexpression \c e.
     * @param e the xexpression involved in the operation.
     * @return a reference to \c *this.
     */
    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator%=(const xexpression<E>& e) -> derived_type&
    {
        return this->derived_cast() = this->derived_cast() % e.derived_cast();
    }

    /**
     * Computes the bitwise and of \c *this and the xexpression \c e and assigns it to \c *this.
     * @param e the xexpression involved in the operation.
     * @return a reference to \c *this.
     */
    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator&=(const xexpression<E>& e) -> derived_type&
    {
        return this->derived_cast() = this->derived_cast() & e.derived_cast();
    }

    /**
     * Computes the bitwise or of \c *this and the xexpression \c e and assigns it to \c *this.
     * @param e the xexpression involved in the operation.
     * @return a reference to \c *this.
     */
    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator|=(const xexpression<E>& e) -> derived_type&
    {
        return this->derived_cast() = this->derived_cast() | e.derived_cast();
    }

    /**
     * Computes the bitwise xor of \c *this and the xexpression \c e and assigns it to \c *this.
     * @param e the xexpression involved in the operation.
     * @return a reference to \c *this.
     */
    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator^=(const xexpression<E>& e) -> derived_type&
    {
        return this->derived_cast() = this->derived_cast() ^ e.derived_cast();
    }

    //@}

    /**
     * @name Assign functions
     */

    /**
     * Assigns the xexpression \c e to \c *this. Ensures no temporary
     * will be used to perform the assignment.
     * @param e the xexpression to assign.
     * @return a reference to \c *this.
     */
    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::assign(const xexpression<E>& e) -> derived_type&
    {
        return this->derived_cast().assign_xexpression(e);
    }

    /**
     * Adds the xexpression \c e to \c *this. Ensures no temporary
     * will be used to perform the assignment.
     * @param e the xexpression to add.
     * @return a reference to \c *this.
     */
    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::plus_assign(const xexpression<E>& e) -> derived_type&
    {
        return this->derived_cast().computed_assign(this->derived_cast() + e.derived_cast());
    }

    /**
     * Subtracts the xexpression \c e to \c *this. Ensures no temporary
     * will be used to perform the assignment.
     * @param e the xexpression to subtract.
     * @return a reference to \c *this.
     */
    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::minus_assign(const xexpression<E>& e) -> derived_type&
    {
        return this->derived_cast().computed_assign(this->derived_cast() - e.derived_cast());
    }

    /**
     * Multiplies \c *this with the xexpression \c e. Ensures no temporary
     * will be used to perform the assignment.
     * @param e the xexpression involved in the operation.
     * @return a reference to \c *this.
     */
    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::multiplies_assign(const xexpression<E>& e) -> derived_type&
    {
        return this->derived_cast().computed_assign(this->derived_cast() * e.derived_cast());
    }

    /**
     * Divides \c *this by the xexpression \c e. Ensures no temporary
     * will be used to perform the assignment.
     * @param e the xexpression involved in the operation.
     * @return a reference to \c *this.
     */
    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::divides_assign(const xexpression<E>& e) -> derived_type&
    {
        return this->derived_cast().computed_assign(this->derived_cast() / e.derived_cast());
    }

    /**
     * Computes the remainder of \c *this after division by the xexpression \c e.
     * Ensures no temporary will be used to perform the assignment.
     * @param e the xexpression involved in the operation.
     * @return a reference to \c *this.
     */
    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::modulus_assign(const xexpression<E>& e) -> derived_type&
    {
        return this->derived_cast().computed_assign(this->derived_cast() % e.derived_cast());
    }

    /**
     * Computes the bitwise and of \c e to \c *this. Ensures no temporary
     * will be used to perform the assignment.
     * @param e the xexpression to add.
     * @return a reference to \c *this.
     */
    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::bit_and_assign(const xexpression<E>& e) -> derived_type&
    {
        return this->derived_cast().computed_assign(this->derived_cast() & e.derived_cast());
    }

    /**
     * Computes the bitwise or of \c e to \c *this. Ensures no temporary
     * will be used to perform the assignment.
     * @param e the xexpression to add.
     * @return a reference to \c *this.
     */
    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::bit_or_assign(const xexpression<E>& e) -> derived_type&
    {
        return this->derived_cast().computed_assign(this->derived_cast() | e.derived_cast());
    }

    /**
     * Computes the bitwise xor of \c e to \c *this. Ensures no temporary
     * will be used to perform the assignment.
     * @param e the xexpression to add.
     * @return a reference to \c *this.
     */
    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::bit_xor_assign(const xexpression<E>& e) -> derived_type&
    {
        return this->derived_cast().computed_assign(this->derived_cast() ^ e.derived_cast());
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator=(const xexpression<E>& e) -> derived_type&
    {
#ifdef XTENSOR_FORCE_TEMPORARY_MEMORY_IN_ASSIGNMENTS
        temporary_type tmp(e);
        return this->derived_cast().assign_temporary(std::move(tmp));
#else
        auto&& this_derived = this->derived_cast();
        auto memory_checker = make_overlapping_memory_checker(this_derived);
        if (memory_checker.check_overlap(e.derived_cast()))
        {
            temporary_type tmp(e);
            return this_derived.assign_temporary(std::move(tmp));
        }
        else
        {
            return this->assign(e);
        }
#endif
    }

    /**************************************
     * xcontainer_semantic implementation *
     **************************************/

    /**
     * Assigns the temporary \c tmp to \c *this.
     * @param tmp the temporary to assign.
     * @return a reference to \c *this.
     */
    template <class D>
    inline auto xcontainer_semantic<D>::assign_temporary(temporary_type&& tmp) -> derived_type&
    {
        return (this->derived_cast() = std::move(tmp));
    }

    template <class D>
    template <class E>
    inline auto xcontainer_semantic<D>::assign_xexpression(const xexpression<E>& e) -> derived_type&
    {
        xt::assign_xexpression(*this, e);
        return this->derived_cast();
    }

    template <class D>
    template <class E>
    inline auto xcontainer_semantic<D>::computed_assign(const xexpression<E>& e) -> derived_type&
    {
        xt::computed_assign(*this, e);
        return this->derived_cast();
    }

    template <class D>
    template <class E, class F>
    inline auto xcontainer_semantic<D>::scalar_computed_assign(const E& e, F&& f) -> derived_type&
    {
        xt::scalar_computed_assign(*this, e, std::forward<F>(f));
        return this->derived_cast();
    }

    template <class D>
    template <class E>
    inline auto xcontainer_semantic<D>::operator=(const xexpression<E>& e) -> derived_type&
    {
        return base_type::operator=(e);
    }

    /*********************************
     * xview_semantic implementation *
     *********************************/

    /**
     * Assigns the temporary \c tmp to \c *this.
     * @param tmp the temporary to assign.
     * @return a reference to \c *this.
     */
    template <class D>
    inline auto xview_semantic<D>::assign_temporary(temporary_type&& tmp) -> derived_type&
    {
        this->derived_cast().assign_temporary_impl(std::move(tmp));
        return this->derived_cast();
    }

    namespace detail
    {
        template <class F>
        bool get_rhs_triviality(const F&)
        {
            return true;
        }

        template <class F, class R, class... CT>
        bool get_rhs_triviality(const xfunction<F, R, CT...>& rhs)
        {
            using index_type = xindex_type_t<typename xfunction<F, R, CT...>::shape_type>;
            using size_type = typename index_type::size_type;
            size_type size = rhs.dimension();
            index_type shape = uninitialized_shape<index_type>(size);
            bool trivial_broadcast = rhs.broadcast_shape(shape, true);
            return trivial_broadcast;
        }
    }

    template <class D>
    template <class E>
    inline auto xview_semantic<D>::assign_xexpression(const xexpression<E>& e) -> derived_type&
    {
        xt::assert_compatible_shape(*this, e);
        xt::assign_data(*this, e, detail::get_rhs_triviality(e.derived_cast()));
        return this->derived_cast();
    }

    template <class D>
    template <class E>
    inline auto xview_semantic<D>::computed_assign(const xexpression<E>& e) -> derived_type&
    {
        xt::assert_compatible_shape(*this, e);
        xt::assign_data(*this, e, detail::get_rhs_triviality(e.derived_cast()));
        return this->derived_cast();
    }

    namespace xview_semantic_detail
    {
        template <class D>
        auto get_begin(D&& lhs, std::true_type)
        {
            return lhs.linear_begin();
        }

        template <class D>
        auto get_begin(D&& lhs, std::false_type)
        {
            return lhs.begin();
        }
    }

    template <class D>
    template <class E, class F>
    inline auto xview_semantic<D>::scalar_computed_assign(const E& e, F&& f) -> derived_type&
    {
        D& d = this->derived_cast();

        using size_type = typename D::size_type;
        auto dst = xview_semantic_detail::get_begin(d, std::integral_constant<bool, D::contiguous_layout>());
        for (size_type i = d.size(); i > 0; --i)
        {
            *dst = f(*dst, e);
            ++dst;
        }
        return this->derived_cast();
    }

    template <class D>
    template <class E>
    inline auto xview_semantic<D>::operator=(const xexpression<E>& rhs) -> derived_type&
    {
        bool cond = (rhs.derived_cast().shape().size() == this->derived_cast().dimension())
                    && std::equal(
                        this->derived_cast().shape().begin(),
                        this->derived_cast().shape().end(),
                        rhs.derived_cast().shape().begin()
                    );

        if (!cond)
        {
            base_type::operator=(broadcast(rhs.derived_cast(), this->derived_cast().shape()));
        }
        else
        {
            base_type::operator=(rhs);
        }
        return this->derived_cast();
    }
}

#endif
