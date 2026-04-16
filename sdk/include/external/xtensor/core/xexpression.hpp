/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_EXPRESSION_HPP
#define XTENSOR_EXPRESSION_HPP

#include <cstddef>
#include <type_traits>
#include <vector>

#include <xtl/xclosure.hpp>
#include <xtl/xmeta_utils.hpp>
#include <xtl/xtype_traits.hpp>

#include "../core/xlayout.hpp"
#include "../core/xshape.hpp"
#include "../core/xtensor_forward.hpp"
#include "../utils/xutils.hpp"

namespace xt
{

    /***************************
     * xexpression declaration *
     ***************************/

    /**
     * @class xexpression
     * @brief Base class for xexpressions
     *
     * The xexpression class is the base class for all classes representing an expression
     * that can be evaluated to a multidimensional container with tensor semantic.
     * Functions that can apply to any xexpression regardless of its specific type should take a
     * xexpression argument.
     *
     * @tparam E The derived type.
     *
     */
    template <class D>
    class xexpression
    {
    public:

        using derived_type = D;

        derived_type& derived_cast() & noexcept;
        const derived_type& derived_cast() const& noexcept;
        derived_type derived_cast() && noexcept;

    protected:

        xexpression() = default;
        ~xexpression() = default;

        xexpression(const xexpression&) = default;
        xexpression& operator=(const xexpression&) = default;

        xexpression(xexpression&&) = default;
        xexpression& operator=(xexpression&&) = default;
    };

    /************************************
     * xsharable_expression declaration *
     ************************************/

    template <class E>
    class xshared_expression;

    template <class E>
    class xsharable_expression;

    namespace detail
    {
        template <class E>
        xshared_expression<E> make_xshared_impl(xsharable_expression<E>&&);
    }

    template <class D>
    class xsharable_expression : public xexpression<D>
    {
    protected:

        xsharable_expression();
        ~xsharable_expression() = default;

        xsharable_expression(const xsharable_expression&) = default;
        xsharable_expression& operator=(const xsharable_expression&) = default;

        xsharable_expression(xsharable_expression&&) = default;
        xsharable_expression& operator=(xsharable_expression&&) = default;

    private:

        std::shared_ptr<D> p_shared;

        friend xshared_expression<D> detail::make_xshared_impl<D>(xsharable_expression<D>&&);
    };

    /******************************
     * xexpression implementation *
     ******************************/

    /**
     * @name Downcast functions
     */
    //@{
    /**
     * Returns a reference to the actual derived type of the xexpression.
     */
    template <class D>
    inline auto xexpression<D>::derived_cast() & noexcept -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    /**
     * Returns a constant reference to the actual derived type of the xexpression.
     */
    template <class D>
    inline auto xexpression<D>::derived_cast() const& noexcept -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

    /**
     * Returns a constant reference to the actual derived type of the xexpression.
     */
    template <class D>
    inline auto xexpression<D>::derived_cast() && noexcept -> derived_type
    {
        return *static_cast<derived_type*>(this);
    }

    //@}

    /***************************************
     * xsharable_expression implementation *
     ***************************************/

    template <class D>
    inline xsharable_expression<D>::xsharable_expression()
        : p_shared(nullptr)
    {
    }

    /**
     * is_crtp_base_of<B, E>
     * Resembles std::is_base_of, but adresses the problem of whether _some_ instantiation
     * of a CRTP templated class B is a base of class E. A CRTP templated class is correctly
     * templated with the most derived type in the CRTP hierarchy. Using this assumption,
     * this implementation deals with either CRTP final classes (checks for inheritance
     * with E as the CRTP parameter of B) or CRTP base classes (which are singly templated
     * by the most derived class, and that's pulled out to use as a templete parameter for B).
     */

    namespace detail
    {
        template <template <class> class B, class E>
        struct is_crtp_base_of_impl : std::is_base_of<B<E>, E>
        {
        };

        template <template <class> class B, class E, template <class> class F>
        struct is_crtp_base_of_impl<B, F<E>>
            : std::disjunction<std::is_base_of<B<E>, F<E>>, std::is_base_of<B<F<E>>, F<E>>>
        {
        };
    }

    template <template <class> class B, class E>
    using is_crtp_base_of = detail::is_crtp_base_of_impl<B, std::decay_t<E>>;

    template <class E>
    using is_xexpression = is_crtp_base_of<xexpression, E>;

    template <class E>
    concept xexpression_concept = is_xexpression<E>::value;

    template <class E, class R = void>
    using enable_xexpression = typename std::enable_if<is_xexpression<E>::value, R>::type;

    template <class E, class R = void>
    using disable_xexpression = typename std::enable_if<!is_xexpression<E>::value, R>::type;

    template <class... E>
    using has_xexpression = std::disjunction<is_xexpression<E>...>;

    template <class E>
    using is_xsharable_expression = is_crtp_base_of<xsharable_expression, E>;

    template <class E, class R = void>
    using enable_xsharable_expression = typename std::enable_if<is_xsharable_expression<E>::value, R>::type;

    template <class E, class R = void>
    using disable_xsharable_expression = typename std::enable_if<!is_xsharable_expression<E>::value, R>::type;

    template <class LHS, class RHS>
    struct can_assign : std::is_assignable<LHS, RHS>
    {
    };

    template <class LHS, class RHS, class R = void>
    using enable_assignable_expression = typename std::enable_if<can_assign<LHS, RHS>::value, R>::type;

    template <class LHS, class RHS, class R = void>
    using enable_not_assignable_expression = typename std::enable_if<!can_assign<LHS, RHS>::value, R>::type;

    /***********************
     * evaluation_strategy *
     ***********************/

    namespace detail
    {
        struct option_base
        {
        };
    }

    namespace evaluation_strategy
    {

        struct immediate_type : xt::detail::option_base
        {
        };

        constexpr auto immediate = std::tuple<immediate_type>{};

        struct lazy_type : xt::detail::option_base
        {
        };

        constexpr auto lazy = std::tuple<lazy_type>{};

        /*
        struct cached {};
        */
    }

    template <class T>
    struct is_evaluation_strategy : std::is_base_of<detail::option_base, std::decay_t<T>>
    {
    };

    /************
     * xclosure *
     ************/

    template <class T>
    class xscalar;

    template <class E, class EN = void>
    struct xclosure
    {
        using type = xtl::closure_type_t<E>;
    };

    template <class E>
    struct xclosure<xshared_expression<E>, std::enable_if_t<true>>
    {
        using type = xshared_expression<E>;  // force copy
    };

    template <class E>
    struct xclosure<E, disable_xexpression<std::decay_t<E>>>
    {
        using type = xscalar<xtl::closure_type_t<E>>;
    };

    template <class E>
    using xclosure_t = typename xclosure<E>::type;

    template <class E, class EN = void>
    struct const_xclosure
    {
        using type = xtl::const_closure_type_t<E>;
    };

    template <class E>
    struct const_xclosure<E, disable_xexpression<std::decay_t<E>>>
    {
        using type = xscalar<xtl::const_closure_type_t<E>>;
    };

    template <class E>
    struct const_xclosure<xshared_expression<E>&, std::enable_if_t<true>>
    {
        using type = xshared_expression<E>;  // force copy
    };

    template <class E>
    using const_xclosure_t = typename const_xclosure<E>::type;

    /*************************
     * expression tag system *
     *************************/

    struct xtensor_expression_tag
    {
    };

    struct xoptional_expression_tag
    {
    };

    namespace extension
    {
        template <class E, class = void_t<int>>
        struct get_expression_tag_impl
        {
            using type = xtensor_expression_tag;
        };

        template <class E>
        struct get_expression_tag_impl<E, void_t<typename std::decay_t<E>::expression_tag>>
        {
            using type = typename std::decay_t<E>::expression_tag;
        };

        template <class E>
        struct get_expression_tag : get_expression_tag_impl<E>
        {
        };

        template <class E>
        using get_expression_tag_t = typename get_expression_tag<E>::type;

        template <class... T>
        struct expression_tag_and;

        template <>
        struct expression_tag_and<>
        {
            using type = xtensor_expression_tag;
        };

        template <class T>
        struct expression_tag_and<T>
        {
            using type = T;
        };

        template <class T>
        struct expression_tag_and<T, T>
        {
            using type = T;
        };

        template <class T>
        struct expression_tag_and<xtensor_expression_tag, T>
        {
            using type = T;
        };

        template <class T>
        struct expression_tag_and<T, xtensor_expression_tag> : expression_tag_and<xtensor_expression_tag, T>
        {
        };

        template <>
        struct expression_tag_and<xtensor_expression_tag, xtensor_expression_tag>
        {
            using type = xtensor_expression_tag;
        };

        template <class T1, class... T>
        struct expression_tag_and<T1, T...> : expression_tag_and<T1, typename expression_tag_and<T...>::type>
        {
        };

        template <class... T>
        using expression_tag_and_t = typename expression_tag_and<T...>::type;

        struct xtensor_empty_base
        {
            using expression_tag = xtensor_expression_tag;
        };
    }

    template <class... T>
    struct xexpression_tag
    {
        using type = extension::expression_tag_and_t<
            extension::get_expression_tag_t<std::decay_t<const_xclosure_t<T>>>...>;
    };

    template <class... T>
    using xexpression_tag_t = typename xexpression_tag<T...>::type;

    template <class E>
    struct is_xtensor_expression : std::is_same<xexpression_tag_t<E>, xtensor_expression_tag>
    {
    };

    template <class E>
    struct is_xoptional_expression : std::is_same<xexpression_tag_t<E>, xoptional_expression_tag>
    {
    };

    /********************************
     * xoptional_comparable concept *
     ********************************/

    template <class... E>
    struct xoptional_comparable
        : std::conjunction<std::disjunction<is_xtensor_expression<E>, is_xoptional_expression<E>>...>
    {
    };

#define XTENSOR_FORWARD_CONST_METHOD(name)                                   \
    auto name() const -> decltype(std::declval<xtl::constify_t<E>>().name()) \
    {                                                                        \
        return m_ptr->name();                                                \
    }

#define XTENSOR_FORWARD_METHOD(name)                  \
    auto name() -> decltype(std::declval<E>().name()) \
    {                                                 \
        return m_ptr->name();                         \
    }

#define XTENSOR_FORWARD_CONST_ITERATOR_METHOD(name)                                               \
    template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>                                          \
    auto name() const noexcept -> decltype(std::declval<xtl::constify_t<E>>().template name<L>()) \
    {                                                                                             \
        return m_ptr->template name<L>();                                                         \
    }                                                                                             \
    template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class S>                                 \
    auto name(const S& shape) const noexcept                                                      \
        -> decltype(std::declval<xtl::constify_t<E>>().template name<L>(shape))                   \
    {                                                                                             \
        return m_ptr->template name<L>();                                                         \
    }

#define XTENSOR_FORWARD_ITERATOR_METHOD(name)                                                 \
    template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL, class S>                             \
    auto name(const S& shape) noexcept -> decltype(std::declval<E>().template name<L>(shape)) \
    {                                                                                         \
        return m_ptr->template name<L>();                                                     \
    }                                                                                         \
    template <layout_type L = XTENSOR_DEFAULT_TRAVERSAL>                                      \
    auto name() noexcept -> decltype(std::declval<E>().template name<L>())                    \
    {                                                                                         \
        return m_ptr->template name<L>();                                                     \
    }

    namespace detail
    {
        template <class E>
        struct expr_strides_type
        {
            using type = typename E::strides_type;
        };

        template <class E>
        struct expr_inner_strides_type
        {
            using type = typename E::inner_strides_type;
        };

        template <class E>
        struct expr_backstrides_type
        {
            using type = typename E::backstrides_type;
        };

        template <class E>
        struct expr_inner_backstrides_type
        {
            using type = typename E::inner_backstrides_type;
        };

        template <class E>
        struct expr_storage_type
        {
            using type = typename E::storage_type;
        };
    }

    /**
     * @class xshared_expression
     * @brief Shared xexpressions
     *
     * Due to C++ lifetime constraints it's sometimes necessary to create shared
     * expressions (akin to a shared pointer).
     *
     * For example, when a temporary expression needs to be used twice in another
     * expression, shared expressions can come to the rescue:
     *
     * @code{.cpp}
     * template <class E>
     * auto cos_plus_sin(xexpression<E>&& expr)
     * {
     *     // THIS IS WRONG: forwarding rvalue twice not permitted!
     *     // return xt::sin(std::forward<E>(expr)) + xt::cos(std::forward<E>(expr));
     *     // THIS IS WRONG TOO: because second `expr` is taken as reference (which will be invalid)
     *     // return xt::sin(std::forward<E>(expr)) + xt::cos(expr)
     *     auto shared_expr = xt::make_xshared(std::forward<E>(expr));
     *     auto result = xt::sin(shared_expr) + xt::cos(shared_expr);
     *     std::cout << shared_expr.use_count() << std::endl; // Will print 3 because used twice in expression
     *     return result; // all valid because expr lifetime managed by xshared_expression / shared_ptr.
     * }
     * @endcode
     */
    template <class E>
    class xshared_expression : public xexpression<xshared_expression<E>>
    {
    public:

        using base_class = xexpression<xshared_expression<E>>;

        using value_type = typename E::value_type;
        using reference = typename E::reference;
        using const_reference = typename E::const_reference;
        using pointer = typename E::pointer;
        using const_pointer = typename E::const_pointer;
        using size_type = typename E::size_type;
        using difference_type = typename E::difference_type;

        using inner_shape_type = typename E::inner_shape_type;
        using shape_type = typename E::shape_type;

        using strides_type = xtl::mpl::
            eval_if_t<has_strides<E>, detail::expr_strides_type<E>, get_strides_type<shape_type>>;
        using backstrides_type = xtl::mpl::
            eval_if_t<has_strides<E>, detail::expr_backstrides_type<E>, get_strides_type<shape_type>>;
        using inner_strides_type = xtl::mpl::
            eval_if_t<has_strides<E>, detail::expr_inner_strides_type<E>, get_strides_type<shape_type>>;
        using inner_backstrides_type = xtl::mpl::
            eval_if_t<has_strides<E>, detail::expr_inner_backstrides_type<E>, get_strides_type<shape_type>>;
        using storage_type = xtl::mpl::eval_if_t<has_storage_type<E>, detail::expr_storage_type<E>, make_invalid_type<>>;

        using stepper = typename E::stepper;
        using const_stepper = typename E::const_stepper;

        using linear_iterator = typename E::linear_iterator;
        using const_linear_iterator = typename E::const_linear_iterator;

        using bool_load_type = typename E::bool_load_type;

        static constexpr layout_type static_layout = E::static_layout;
        static constexpr bool contiguous_layout = static_layout != layout_type::dynamic;

        explicit xshared_expression(const std::shared_ptr<E>& ptr);
        long use_count() const noexcept;

        template <class... Args>
        auto operator()(Args... args) -> decltype(std::declval<E>()(args...))
        {
            return m_ptr->operator()(args...);
        }

        XTENSOR_FORWARD_CONST_METHOD(shape)
        XTENSOR_FORWARD_CONST_METHOD(dimension)
        XTENSOR_FORWARD_CONST_METHOD(size)
        XTENSOR_FORWARD_CONST_METHOD(layout)
        XTENSOR_FORWARD_CONST_METHOD(is_contiguous)

        XTENSOR_FORWARD_ITERATOR_METHOD(begin)
        XTENSOR_FORWARD_ITERATOR_METHOD(end)
        XTENSOR_FORWARD_CONST_ITERATOR_METHOD(begin)
        XTENSOR_FORWARD_CONST_ITERATOR_METHOD(end)
        XTENSOR_FORWARD_CONST_ITERATOR_METHOD(cbegin)
        XTENSOR_FORWARD_CONST_ITERATOR_METHOD(cend)

        XTENSOR_FORWARD_ITERATOR_METHOD(rbegin)
        XTENSOR_FORWARD_ITERATOR_METHOD(rend)
        XTENSOR_FORWARD_CONST_ITERATOR_METHOD(rbegin)
        XTENSOR_FORWARD_CONST_ITERATOR_METHOD(rend)
        XTENSOR_FORWARD_CONST_ITERATOR_METHOD(crbegin)
        XTENSOR_FORWARD_CONST_ITERATOR_METHOD(crend)

        XTENSOR_FORWARD_METHOD(linear_begin)
        XTENSOR_FORWARD_METHOD(linear_end)
        XTENSOR_FORWARD_CONST_METHOD(linear_begin)
        XTENSOR_FORWARD_CONST_METHOD(linear_end)
        XTENSOR_FORWARD_CONST_METHOD(linear_cbegin)
        XTENSOR_FORWARD_CONST_METHOD(linear_cend)

        XTENSOR_FORWARD_METHOD(linear_rbegin)
        XTENSOR_FORWARD_METHOD(linear_rend)
        XTENSOR_FORWARD_CONST_METHOD(linear_rbegin)
        XTENSOR_FORWARD_CONST_METHOD(linear_rend)
        XTENSOR_FORWARD_CONST_METHOD(linear_crbegin)
        XTENSOR_FORWARD_CONST_METHOD(linear_crend)

        template <class T = E>
        std::enable_if_t<has_strides<T>::value, const inner_strides_type&> strides() const
        {
            return m_ptr->strides();
        }

        template <class T = E>
        std::enable_if_t<has_strides<T>::value, const inner_strides_type&> backstrides() const
        {
            return m_ptr->backstrides();
        }

        template <class T = E>
        std::enable_if_t<has_data_interface<T>::value, pointer> data() noexcept
        {
            return m_ptr->data();
        }

        template <class T = E>
        std::enable_if_t<has_data_interface<T>::value, pointer> data() const noexcept
        {
            return m_ptr->data();
        }

        template <class T = E>
        std::enable_if_t<has_data_interface<T>::value, size_type> data_offset() const noexcept
        {
            return m_ptr->data_offset();
        }

        template <class T = E>
        std::enable_if_t<has_data_interface<T>::value, typename T::storage_type&> storage() noexcept
        {
            return m_ptr->storage();
        }

        template <class T = E>
        std::enable_if_t<has_data_interface<T>::value, const typename T::storage_type&> storage() const noexcept
        {
            return m_ptr->storage();
        }

        template <class It>
        reference element(It first, It last)
        {
            return m_ptr->element(first, last);
        }

        template <class It>
        const_reference element(It first, It last) const
        {
            return m_ptr->element(first, last);
        }

        template <class S>
        bool broadcast_shape(S& shape, bool reuse_cache = false) const
        {
            return m_ptr->broadcast_shape(shape, reuse_cache);
        }

        template <class S>
        bool has_linear_assign(const S& strides) const noexcept
        {
            return m_ptr->has_linear_assign(strides);
        }

        template <class S>
        auto stepper_begin(const S& shape) noexcept -> decltype(std::declval<E>().stepper_begin(shape))
        {
            return m_ptr->stepper_begin(shape);
        }

        template <class S>
        auto stepper_end(const S& shape, layout_type l) noexcept
            -> decltype(std::declval<E>().stepper_end(shape, l))
        {
            return m_ptr->stepper_end(shape, l);
        }

        template <class S>
        auto stepper_begin(const S& shape) const noexcept
            -> decltype(std::declval<const E>().stepper_begin(shape))
        {
            return static_cast<const E*>(m_ptr.get())->stepper_begin(shape);
        }

        template <class S>
        auto stepper_end(const S& shape, layout_type l) const noexcept
            -> decltype(std::declval<const E>().stepper_end(shape, l))
        {
            return static_cast<const E*>(m_ptr.get())->stepper_end(shape, l);
        }

    private:

        std::shared_ptr<E> m_ptr;
    };

    /**
     * Constructor for xshared expression (note: usually the free function
     * `make_xshared` is recommended).
     *
     * @param ptr shared ptr that contains the expression
     * @sa make_xshared
     */
    template <class E>
    inline xshared_expression<E>::xshared_expression(const std::shared_ptr<E>& ptr)
        : m_ptr(ptr)
    {
    }

    /**
     * Return the number of times this expression is referenced.
     * Internally calls the use_count() function of the std::shared_ptr.
     */
    template <class E>
    inline long xshared_expression<E>::use_count() const noexcept
    {
        return m_ptr.use_count();
    }

    namespace detail
    {
        template <class E>
        inline xshared_expression<E> make_xshared_impl(xsharable_expression<E>&& expr)
        {
            if (expr.p_shared == nullptr)
            {
                expr.p_shared = std::make_shared<E>(std::move(expr).derived_cast());
            }
            return xshared_expression<E>(expr.p_shared);
        }
    }

    /**
     * Helper function to create shared expression from any xexpression
     *
     * @param expr rvalue expression that will be shared
     * @return xshared expression
     */
    template <class E>
    inline xshared_expression<E> make_xshared(xexpression<E>&& expr)
    {
        static_assert(
            is_xsharable_expression<E>::value,
            "make_shared requires E to inherit from xsharable_expression"
        );
        return detail::make_xshared_impl(std::move(expr.derived_cast()));
    }

    /**
     * Helper function to create shared expression from any xexpression
     *
     * @param expr rvalue expression that will be shared
     * @return xshared expression
     * @sa make_xshared
     */
    template <class E>
    inline auto share(xexpression<E>& expr)
    {
        return make_xshared(std::move(expr));
    }

    /**
     * Helper function to create shared expression from any xexpression
     *
     * @param expr rvalue expression that will be shared
     * @return xshared expression
     * @sa make_xshared
     */
    template <class E>
    inline auto share(xexpression<E>&& expr)
    {
        return make_xshared(std::move(expr));
    }

#undef XTENSOR_FORWARD_METHOD

}

#endif
