/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_GENERATOR_HPP
#define XTENSOR_GENERATOR_HPP

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>

#include <xtl/xsequence.hpp>

#include "../core/xaccessible.hpp"
#include "../core/xexpression.hpp"
#include "../core/xiterable.hpp"
#include "../core/xstrides.hpp"
#include "../utils/xutils.hpp"
#include "../views/xstrided_view.hpp"

namespace xt
{

    /************************
     * xgenerator extension *
     ************************/

    namespace extension
    {
        template <class Tag, class F, class R, class S>
        struct xgenerator_base_impl;

        template <class F, class R, class S>
        struct xgenerator_base_impl<xtensor_expression_tag, F, R, S>
        {
            using type = xtensor_empty_base;
        };

        template <class F, class R, class S>
        struct xgenerator_base : xgenerator_base_impl<xexpression_tag_t<R>, F, R, S>
        {
        };

        template <class F, class R, class S>
        using xgenerator_base_t = typename xgenerator_base<F, R, S>::type;
    }

    /**************
     * xgenerator *
     **************/

    template <class F, class R, class S>
    class xgenerator;

    template <typename T>
    concept xgenerator_concept = is_specialization_of<xgenerator, std::decay_t<T>>::value;

    template <class C, class R, class S>
    struct xiterable_inner_types<xgenerator<C, R, S>>
    {
        using inner_shape_type = S;
        using const_stepper = xindexed_stepper<xgenerator<C, R, S>, true>;
        using stepper = const_stepper;
    };

    template <class C, class R, class S>
    struct xcontainer_inner_types<xgenerator<C, R, S>>
    {
        using reference = R;
        using const_reference = R;
        using size_type = std::size_t;
    };

    /*************************************
     * overlapping_memory_checker_traits *
     *************************************/

    template <xgenerator_concept E>
        requires(without_memory_address_concept<E>)
    struct overlapping_memory_checker_traits<E>
    {
        static bool check_overlap(const E&, const memory_range&)
        {
            return false;
        }
    };

    /**
     * @class xgenerator
     * @brief Multidimensional function operating on indices.
     *
     * The xgenerator class implements a multidimensional function,
     * generating a value from the supplied indices.
     *
     * @tparam F the function type
     * @tparam R the return type of the function
     * @tparam S the shape type of the generator
     */
    template <class F, class R, class S>
    class xgenerator : public xsharable_expression<xgenerator<F, R, S>>,
                       public xconst_iterable<xgenerator<F, R, S>>,
                       public xconst_accessible<xgenerator<F, R, S>>,
                       public extension::xgenerator_base_t<F, R, S>
    {
    public:

        using self_type = xgenerator<F, R, S>;
        using functor_type = typename std::remove_reference<F>::type;

        using accessible_base = xconst_accessible<self_type>;
        using extension_base = extension::xgenerator_base_t<F, R, S>;
        using expression_tag = typename extension_base::expression_tag;

        using inner_types = xcontainer_inner_types<self_type>;
        using value_type = R;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using size_type = typename inner_types::size_type;
        using difference_type = std::ptrdiff_t;

        using iterable_base = xconst_iterable<self_type>;
        using inner_shape_type = typename iterable_base::inner_shape_type;
        using shape_type = inner_shape_type;

        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        using bool_load_type = xt::bool_load_type<R>;

        static constexpr layout_type static_layout = layout_type::dynamic;
        static constexpr bool contiguous_layout = false;

        template <class Func>
        xgenerator(Func&& f, const S& shape) noexcept;

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

        template <class O>
        bool broadcast_shape(O& shape, bool reuse_cache = false) const;

        template <class O>
        bool has_linear_assign(const O& /*strides*/) const noexcept;

        template <class O>
        const_stepper stepper_begin(const O& shape) const noexcept;
        template <class O>
        const_stepper stepper_end(const O& shape, layout_type) const noexcept;

        template <class E, class FE = F>
        void assign_to(xexpression<E>& e) const noexcept
            requires(has_assign_to_v<E, FE>);

        const functor_type& functor() const noexcept;

        template <class OR, class OF>
        using rebind_t = xgenerator<OF, OR, S>;

        template <class OR, class OF>
        rebind_t<OR, OF> build_generator(OF&& func) const;

        template <class O = xt::dynamic_shape<typename shape_type::value_type>>
        auto reshape(O&& shape) const&;

        template <class O = xt::dynamic_shape<typename shape_type::value_type>>
        auto reshape(O&& shape) &&;

        template <class T>
        auto reshape(std::initializer_list<T> shape) const&;

        template <class T>
        auto reshape(std::initializer_list<T> shape) &&;

    private:

        template <class O>
        decltype(auto) compute_shape(O&& shape, std::false_type /*signed*/) const;

        template <class O>
        auto compute_shape(O&& shape, std::true_type /*signed*/) const;

        template <class T>
        auto compute_shape(std::initializer_list<T> shape) const;

        template <std::size_t dim>
        void adapt_index() const;

        template <std::size_t dim, class I, class... Args>
        void adapt_index(I& arg, Args&... args) const;

        functor_type m_f;
        inner_shape_type m_shape;
    };

    /*****************************
     * xgenerator implementation *
     *****************************/

    /**
     * @name Constructor
     */
    //@{
    /**
     * Constructs an xgenerator applying the specified function over the
     * given shape.
     * @param f the function to apply
     * @param shape the shape of the xgenerator
     */
    template <class F, class R, class S>
    template <class Func>
    inline xgenerator<F, R, S>::xgenerator(Func&& f, const S& shape) noexcept
        : m_f(std::forward<Func>(f))
        , m_shape(shape)
    {
    }

    //@}

    /**
     * @name Size and shape
     */
    //@{
    /**
     * Returns the shape of the xgenerator.
     */
    template <class F, class R, class S>
    inline auto xgenerator<F, R, S>::shape() const noexcept -> const inner_shape_type&
    {
        return m_shape;
    }

    template <class F, class R, class S>
    inline layout_type xgenerator<F, R, S>::layout() const noexcept
    {
        return static_layout;
    }

    template <class F, class R, class S>
    inline bool xgenerator<F, R, S>::is_contiguous() const noexcept
    {
        return false;
    }

    //@}

    /**
     * @name Data
     */

    /**
     * Returns the evaluated element at the specified position in the function.
     * @param args a list of indices specifying the position in the function. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the function.
     */
    template <class F, class R, class S>
    template <class... Args>
    inline auto xgenerator<F, R, S>::operator()(Args... args) const -> const_reference
    {
        XTENSOR_TRY(check_index(shape(), args...));
        adapt_index<0>(args...);
        return m_f(args...);
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
     *  code has undefined behavior:
     * @code{.cpp}
     * xt::xarray<double> a = {{0, 1}, {2, 3}};
     * xt::xarray<double> b = {0, 1};
     * auto fd = a + b;
     * double res = fd.uncheked(0, 1);
     * @endcode
     */
    template <class F, class R, class S>
    template <class... Args>
    inline auto xgenerator<F, R, S>::unchecked(Args... args) const -> const_reference
    {
        return m_f(args...);
    }

    /**
     * Returns a constant reference to the element at the specified position in the function.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater
     * than the number of dimensions of the container.
     */
    template <class F, class R, class S>
    template <class It>
    inline auto xgenerator<F, R, S>::element(It first, It last) const -> const_reference
    {
        using bounded_iterator = xbounded_iterator<It, typename shape_type::const_iterator>;
        XTENSOR_TRY(check_element_index(shape(), first, last));
        return m_f.element(bounded_iterator(first, shape().cbegin()), bounded_iterator(last, shape().cend()));
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
    template <class F, class R, class S>
    template <class O>
    inline bool xgenerator<F, R, S>::broadcast_shape(O& shape, bool) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    /**
     * Checks whether the xgenerator can be linearly assigned to an expression
     * with the specified strides.
     * @return a boolean indicating whether a linear assign is possible
     */
    template <class F, class R, class S>
    template <class O>
    inline bool xgenerator<F, R, S>::has_linear_assign(const O& /*strides*/) const noexcept
    {
        return false;
    }

    //@}

    template <class F, class R, class S>
    template <class O>
    inline auto xgenerator<F, R, S>::stepper_begin(const O& shape) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - this->dimension();
        return const_stepper(this, offset);
    }

    template <class F, class R, class S>
    template <class O>
    inline auto xgenerator<F, R, S>::stepper_end(const O& shape, layout_type) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - this->dimension();
        return const_stepper(this, offset, true);
    }

    template <class F, class R, class S>
    template <class E, class FE>
    inline void xgenerator<F, R, S>::assign_to(xexpression<E>& e) const noexcept
        requires(has_assign_to_v<E, FE>)
    {
        e.derived_cast().resize(m_shape);
        m_f.assign_to(e);
    }

    template <class F, class R, class S>
    inline auto xgenerator<F, R, S>::functor() const noexcept -> const functor_type&
    {
        return m_f;
    }

    template <class F, class R, class S>
    template <class OR, class OF>
    inline auto xgenerator<F, R, S>::build_generator(OF&& func) const -> rebind_t<OR, OF>
    {
        return rebind_t<OR, OF>(std::move(func), shape_type(m_shape));
    }

    /**
     * Reshapes the generator and keeps old elements. The `shape` argument can have one of its value
     * equal to `-1`, in this case the value is inferred from the number of elements in the generator
     * and the remaining values in the `shape`.
     * @code{.cpp}
     * auto a = xt::arange<double>(50).reshape({-1, 10});
     * //a.shape() is {5, 10}
     * @endcode
     * @param shape the new shape (has to have same number of elements as the original generator)
     */
    template <class F, class R, class S>
    template <class O>
    inline auto xgenerator<F, R, S>::reshape(O&& shape) const&
    {
        return reshape_view(*this, compute_shape(shape, xtl::is_signed<typename std::decay_t<O>::value_type>()));
    }

    template <class F, class R, class S>
    template <class O>
    inline auto xgenerator<F, R, S>::reshape(O&& shape) &&
    {
        return reshape_view(
            std::move(*this),
            compute_shape(shape, xtl::is_signed<typename std::decay_t<O>::value_type>())
        );
    }

    template <class F, class R, class S>
    template <class T>
    inline auto xgenerator<F, R, S>::reshape(std::initializer_list<T> shape) const&
    {
        return reshape_view(*this, compute_shape(shape));
    }

    template <class F, class R, class S>
    template <class T>
    inline auto xgenerator<F, R, S>::reshape(std::initializer_list<T> shape) &&
    {
        return reshape_view(std::move(*this), compute_shape(shape));
    }

    template <class F, class R, class S>
    template <class O>
    inline decltype(auto) xgenerator<F, R, S>::compute_shape(O&& shape, std::false_type) const
    {
        return xtl::forward_sequence<xt::dynamic_shape<typename shape_type::value_type>, O>(shape);
    }

    template <class F, class R, class S>
    template <class O>
    inline auto xgenerator<F, R, S>::compute_shape(O&& shape, std::true_type) const
    {
        using vtype = typename shape_type::value_type;
        xt::dynamic_shape<vtype> sh(shape.size());
        using int_type = typename std::decay_t<O>::value_type;
        int_type accumulator(1);
        std::size_t neg_idx = 0;
        std::size_t i = 0;
        for (std::size_t j = 0; j != shape.size(); ++j, ++i)
        {
            auto dim = shape[j];
            if (dim < 0)
            {
                XTENSOR_ASSERT(dim == -1 && !neg_idx);
                neg_idx = i;
            }
            else
            {
                sh[j] = static_cast<vtype>(dim);
            }
            accumulator *= dim;
        }
        if (accumulator < 0)
        {
            sh[neg_idx] = this->size()
                          / static_cast<size_type>(std::make_unsigned_t<int_type>(std::abs(accumulator)));
        }
        return sh;
    }

    template <class F, class R, class S>
    template <class T>
    inline auto xgenerator<F, R, S>::compute_shape(std::initializer_list<T> shape) const
    {
        using sh_type = xt::dynamic_shape<T>;
        sh_type sh = xtl::make_sequence<sh_type>(shape.size());
        std::copy(shape.begin(), shape.end(), sh.begin());
        return compute_shape(std::move(sh), xtl::is_signed<T>());
    }

    template <class F, class R, class S>
    template <std::size_t dim>
    inline void xgenerator<F, R, S>::adapt_index() const
    {
    }

    template <class F, class R, class S>
    template <std::size_t dim, class I, class... Args>
    inline void xgenerator<F, R, S>::adapt_index(I& arg, Args&... args) const
    {
        using tmp_value_type = typename decltype(m_shape)::value_type;
        if (sizeof...(Args) + 1 > m_shape.size())
        {
            adapt_index<dim>(args...);
        }
        else
        {
            if (static_cast<tmp_value_type>(arg) >= m_shape[dim] && m_shape[dim] == 1)
            {
                arg = 0;
            }
            adapt_index<dim + 1>(args...);
        }
    }

    namespace detail
    {
        template <class Functor, class I, std::size_t L>
        inline auto make_xgenerator(Functor&& f, const I (&shape)[L]) noexcept
        {
            using shape_type = std::array<std::size_t, L>;
            using type = xgenerator<Functor, typename Functor::value_type, shape_type>;
            return type(std::forward<Functor>(f), xtl::forward_sequence<shape_type, decltype(shape)>(shape));
        }

        template <class Functor, class S>
        inline auto make_xgenerator(Functor&& f, S&& shape) noexcept
        {
            using type = xgenerator<Functor, typename Functor::value_type, std::decay_t<S>>;
            return type(std::forward<Functor>(f), std::forward<S>(shape));
        }
    }
}

#endif
