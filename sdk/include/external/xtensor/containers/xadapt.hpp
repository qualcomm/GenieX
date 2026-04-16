/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_ADAPT_HPP
#define XTENSOR_ADAPT_HPP

#include <array>
#include <cstddef>
#include <memory>
#include <type_traits>

#include <xtl/xsequence.hpp>

#include "../containers/xarray.hpp"
#include "../containers/xbuffer_adaptor.hpp"
#include "../containers/xfixed.hpp"
#include "../containers/xtensor.hpp"

namespace xt
{
    /**
     * @defgroup xt_xadapt Adaptors of STL-like containers
     */

    namespace detail
    {
        template <class>
        struct array_size_impl;

        template <class T, std::size_t N>
        struct array_size_impl<std::array<T, N>>
        {
            static constexpr std::size_t value = N;
        };

        template <class C>
        using array_size = array_size_impl<std::decay_t<C>>;

        template <class P>
        struct default_allocator_for_ptr
        {
            using type = std::allocator<std::remove_const_t<std::remove_pointer_t<std::remove_reference_t<P>>>>;
        };

        template <class P>
        using default_allocator_for_ptr_t = typename default_allocator_for_ptr<P>::type;

        template <class T>
        using not_an_array = std::negation<is_array<T>>;

        template <class T>
        using not_a_pointer = std::negation<std::is_pointer<T>>;

        template <class T>
        using not_a_layout = std::negation<std::is_same<layout_type, T>>;
    }

#ifndef IN_DOXYGEN

    /**************************
     * xarray_adaptor builder *
     **************************/

    /**
     * Constructs an xarray_adaptor of the given stl-like container,
     * with the specified shape and layout.
     *
     * @ingroup xt_xadapt
     * @param container the container to adapt
     * @param shape the shape of the xarray_adaptor
     * @param l the layout_type of the xarray_adaptor
     */
    template <
        layout_type L = XTENSOR_DEFAULT_LAYOUT,
        class C,
        class SC,
        XTL_REQUIRES(detail::not_an_array<std::decay_t<SC>>, detail::not_a_pointer<C>)>
    inline xarray_adaptor<xtl::closure_type_t<C>, L, std::decay_t<SC>>
    adapt(C&& container, const SC& shape, layout_type l = L)
    {
        static_assert(!xtl::is_integral<SC>::value, "shape cannot be a integer");
        using return_type = xarray_adaptor<xtl::closure_type_t<C>, L, std::decay_t<SC>>;
        return return_type(std::forward<C>(container), shape, l);
    }

    /**
     * Constructs an non-owning xarray_adaptor from a pointer with the specified shape and layout.
     *
     * @ingroup xt_xadapt
     * @param pointer the container to adapt
     * @param shape the shape of the xarray_adaptor
     * @param l the layout_type of the xarray_adaptor
     */
    template <
        layout_type L = XTENSOR_DEFAULT_LAYOUT,
        class C,
        class SC,
        XTL_REQUIRES(detail::not_an_array<std::decay_t<SC>>, std::is_pointer<std::remove_reference_t<C>>)>
    inline auto adapt(C&& pointer, const SC& shape, layout_type l = L)
    {
        static_assert(!xtl::is_integral<SC>::value, "shape cannot be a integer");
        using buffer_type = xbuffer_adaptor<C, xt::no_ownership, detail::default_allocator_for_ptr_t<C>>;
        using return_type = xarray_adaptor<buffer_type, L, std::decay_t<SC>>;
        std::size_t size = compute_size(shape);
        return return_type(buffer_type(pointer, size), shape, l);
    }

    /**
     * Constructs an xarray_adaptor of the given stl-like container,
     * with the specified shape and strides.
     *
     * @ingroup xt_xadapt
     * @param container the container to adapt
     * @param shape the shape of the xarray_adaptor
     * @param strides the strides of the xarray_adaptor
     */
    template <
        class C,
        class SC,
        class SS,
        XTL_REQUIRES(detail::not_an_array<std::decay_t<SC>>, detail::not_a_layout<std::decay_t<SS>>)>
    inline xarray_adaptor<xtl::closure_type_t<C>, layout_type::dynamic, std::decay_t<SC>>
    adapt(C&& container, SC&& shape, SS&& strides)
    {
        static_assert(!xtl::is_integral<std::decay_t<SC>>::value, "shape cannot be a integer");
        using return_type = xarray_adaptor<xtl::closure_type_t<C>, layout_type::dynamic, std::decay_t<SC>>;
        return return_type(
            std::forward<C>(container),
            xtl::forward_sequence<typename return_type::inner_shape_type, SC>(shape),
            xtl::forward_sequence<typename return_type::inner_strides_type, SS>(strides)
        );
    }

    /**
     * Constructs an xarray_adaptor of the given dynamically allocated C array,
     * with the specified shape and layout.
     *
     * @ingroup xt_xadapt
     * @param pointer the pointer to the beginning of the dynamic array
     * @param size the size of the dynamic array
     * @param ownership indicates whether the adaptor takes ownership of the array.
     *        Possible values are ``no_ownership()`` or ``acquire_ownership()``
     * @param shape the shape of the xarray_adaptor
     * @param l the layout_type of the xarray_adaptor
     * @param alloc the allocator used for allocating / deallocating the dynamic array
     */
    template <
        layout_type L = XTENSOR_DEFAULT_LAYOUT,
        class P,
        class O,
        class SC,
        class A = detail::default_allocator_for_ptr_t<P>,
        XTL_REQUIRES(detail::not_an_array<std::decay_t<SC>>)>
    inline xarray_adaptor<xbuffer_adaptor<xtl::closure_type_t<P>, O, A>, L, SC> adapt(
        P&& pointer,
        typename A::size_type size,
        O ownership,
        const SC& shape,
        layout_type l = L,
        const A& alloc = A()
    )
    {
        static_assert(!xtl::is_integral<SC>::value, "shape cannot be a integer");
        (void) ownership;
        using buffer_type = xbuffer_adaptor<xtl::closure_type_t<P>, O, A>;
        using return_type = xarray_adaptor<buffer_type, L, SC>;
        buffer_type buf(std::forward<P>(pointer), size, alloc);
        return return_type(std::move(buf), shape, l);
    }

    /**
     * Constructs an xarray_adaptor of the given dynamically allocated C array,
     * with the specified shape and strides.
     *
     * @ingroup xt_xadapt
     * @param pointer the pointer to the beginning of the dynamic array
     * @param size the size of the dynamic array
     * @param ownership indicates whether the adaptor takes ownership of the array.
     *        Possible values are ``no_ownership()`` or ``acquire_ownership()``
     * @param shape the shape of the xarray_adaptor
     * @param strides the strides of the xarray_adaptor
     * @param alloc the allocator used for allocating / deallocating the dynamic array
     */
    template <
        class P,
        class O,
        class SC,
        class SS,
        class A = detail::default_allocator_for_ptr_t<P>,
        XTL_REQUIRES(detail::not_an_array<std::decay_t<SC>>, detail::not_a_layout<std::decay_t<SS>>)>
    inline xarray_adaptor<xbuffer_adaptor<xtl::closure_type_t<P>, O, A>, layout_type::dynamic, std::decay_t<SC>>
    adapt(P&& pointer, typename A::size_type size, O ownership, SC&& shape, SS&& strides, const A& alloc = A())
    {
        static_assert(!xtl::is_integral<std::decay_t<SC>>::value, "shape cannot be a integer");
        (void) ownership;
        using buffer_type = xbuffer_adaptor<xtl::closure_type_t<P>, O, A>;
        using return_type = xarray_adaptor<buffer_type, layout_type::dynamic, std::decay_t<SC>>;
        buffer_type buf(std::forward<P>(pointer), size, alloc);
        return return_type(
            std::move(buf),
            xtl::forward_sequence<typename return_type::inner_shape_type, SC>(shape),
            xtl::forward_sequence<typename return_type::inner_strides_type, SS>(strides)
        );
    }

    /**
     * Constructs an xarray_adaptor of the given C array allocated on the stack, with the
     * specified shape and layout.
     *
     * @ingroup xt_xadapt
     * @param c_array the C array allocated on the stack
     * @param shape the shape of the xarray_adaptor
     * @param l the layout_type of the xarray_adaptor
     */
    template <
        layout_type L = XTENSOR_DEFAULT_LAYOUT,
        class T,
        std::size_t N,
        class SC,
        XTL_REQUIRES(detail::not_an_array<std::decay_t<SC>>)>
    inline auto adapt(T (&c_array)[N], const SC& shape, layout_type l = L)
    {
        return adapt(&c_array[0], N, xt::no_ownership(), shape, l);
    }

    /**
     * Constructs an xarray_adaptor of the given C array allocated on the stack, with the
     * specified shape and stirdes.
     *
     * @ingroup xt_xadapt
     * @param c_array the C array allocated on the stack
     * @param shape the shape of the xarray_adaptor
     * @param strides the strides of the xarray_adaptor
     */
    template <
        class T,
        std::size_t N,
        class SC,
        class SS,
        XTL_REQUIRES(detail::not_an_array<std::decay_t<SC>>, detail::not_a_layout<std::decay_t<SS>>)>
    inline auto adapt(T (&c_array)[N], SC&& shape, SS&& strides)
    {
        return adapt(&c_array[0], N, xt::no_ownership(), std::forward<SC>(shape), std::forward<SS>(strides));
    }

    /***************************
     * xtensor_adaptor builder *
     ***************************/

    /**
     * Constructs a 1-D xtensor_adaptor of the given stl-like container,
     * with the specified layout_type.
     *
     * @ingroup xt_xadapt
     * @param container the container to adapt
     * @param l the layout_type of the xtensor_adaptor
     */
    template <layout_type L = XTENSOR_DEFAULT_LAYOUT, class C>
    inline xtensor_adaptor<C, 1, L> adapt(C&& container, layout_type l = L)
    {
        const std::array<typename std::decay_t<C>::size_type, 1> shape{container.size()};
        using return_type = xtensor_adaptor<xtl::closure_type_t<C>, 1, L>;
        return return_type(std::forward<C>(container), shape, l);
    }

    /**
     * Constructs an xtensor_adaptor of the given stl-like container,
     * with the specified shape and layout_type.
     *
     * @ingroup xt_xadapt
     * @param container the container to adapt
     * @param shape the shape of the xtensor_adaptor
     * @param l the layout_type of the xtensor_adaptor
     */
    template <
        layout_type L = XTENSOR_DEFAULT_LAYOUT,
        class C,
        class SC,
        XTL_REQUIRES(detail::is_array<std::decay_t<SC>>, detail::not_a_pointer<C>)>
    inline xtensor_adaptor<C, detail::array_size<SC>::value, L>
    adapt(C&& container, const SC& shape, layout_type l = L)
    {
        static_assert(!xtl::is_integral<SC>::value, "shape cannot be a integer");
        constexpr std::size_t N = detail::array_size<SC>::value;
        using return_type = xtensor_adaptor<xtl::closure_type_t<C>, N, L>;
        return return_type(std::forward<C>(container), shape, l);
    }

    /**
     * Constructs an non-owning xtensor_adaptor from a pointer with the specified shape and layout.
     *
     * @ingroup xt_xadapt
     * @param pointer the pointer to adapt
     * @param shape the shape of the xtensor_adaptor
     * @param l the layout_type of the xtensor_adaptor
     */
    template <
        layout_type L = XTENSOR_DEFAULT_LAYOUT,
        class C,
        class SC,
        XTL_REQUIRES(detail::is_array<std::decay_t<SC>>, std::is_pointer<std::remove_reference_t<C>>)>
    inline auto adapt(C&& pointer, const SC& shape, layout_type l = L)
    {
        static_assert(!xtl::is_integral<SC>::value, "shape cannot be a integer");
        using buffer_type = xbuffer_adaptor<C, xt::no_ownership, detail::default_allocator_for_ptr_t<C>>;
        constexpr std::size_t N = detail::array_size<SC>::value;
        using return_type = xtensor_adaptor<buffer_type, N, L>;
        return return_type(buffer_type(pointer, compute_size(shape)), shape, l);
    }

    /**
     * Constructs an xtensor_adaptor of the given stl-like container,
     * with the specified shape and strides.
     *
     * @ingroup xt_xadapt
     * @param container the container to adapt
     * @param shape the shape of the xtensor_adaptor
     * @param strides the strides of the xtensor_adaptor
     */
    template <
        class C,
        class SC,
        class SS,
        XTL_REQUIRES(detail::is_array<std::decay_t<SC>>, detail::not_a_layout<std::decay_t<SS>>)>
    inline xtensor_adaptor<C, detail::array_size<SC>::value, layout_type::dynamic>
    adapt(C&& container, SC&& shape, SS&& strides)
    {
        static_assert(!xtl::is_integral<std::decay_t<SC>>::value, "shape cannot be a integer");
        constexpr std::size_t N = detail::array_size<SC>::value;
        using return_type = xtensor_adaptor<xtl::closure_type_t<C>, N, layout_type::dynamic>;
        return return_type(
            std::forward<C>(container),
            xtl::forward_sequence<typename return_type::inner_shape_type, SC>(shape),
            xtl::forward_sequence<typename return_type::inner_strides_type, SS>(strides)
        );
    }

    /**
     * Constructs a 1-D xtensor_adaptor of the given dynamically allocated C array,
     * with the specified layout.
     *
     * @ingroup xt_xadapt
     * @param pointer the pointer to the beginning of the dynamic array
     * @param size the size of the dynamic array
     * @param ownership indicates whether the adaptor takes ownership of the array.
     *        Possible values are ``no_ownership()`` or ``acquire_ownership()``
     * @param l the layout_type of the xtensor_adaptor
     * @param alloc the allocator used for allocating / deallocating the dynamic array
     */
    template <layout_type L = XTENSOR_DEFAULT_LAYOUT, class P, class O, class A = detail::default_allocator_for_ptr_t<P>>
    inline xtensor_adaptor<xbuffer_adaptor<xtl::closure_type_t<P>, O, A>, 1, L>
    adapt(P&& pointer, typename A::size_type size, O ownership, layout_type l = L, const A& alloc = A())
    {
        (void) ownership;
        using buffer_type = xbuffer_adaptor<xtl::closure_type_t<P>, O, A>;
        using return_type = xtensor_adaptor<buffer_type, 1, L>;
        buffer_type buf(std::forward<P>(pointer), size, alloc);
        const std::array<typename A::size_type, 1> shape{size};
        return return_type(std::move(buf), shape, l);
    }

    /**
     * Constructs an xtensor_adaptor of the given dynamically allocated C array,
     * with the specified shape and layout.
     *
     * @ingroup xt_xadapt
     * @param pointer the pointer to the beginning of the dynamic array
     * @param size the size of the dynamic array
     * @param ownership indicates whether the adaptor takes ownership of the array.
     *        Possible values are ``no_ownership()`` or ``acquire_ownership()``
     * @param shape the shape of the xtensor_adaptor
     * @param l the layout_type of the xtensor_adaptor
     * @param alloc the allocator used for allocating / deallocating the dynamic array
     */
    template <
        layout_type L = XTENSOR_DEFAULT_LAYOUT,
        class P,
        class O,
        class SC,
        class A = detail::default_allocator_for_ptr_t<P>,
        XTL_REQUIRES(detail::is_array<std::decay_t<SC>>)>
    inline xtensor_adaptor<xbuffer_adaptor<xtl::closure_type_t<P>, O, A>, detail::array_size<SC>::value, L>
    adapt(
        P&& pointer,
        typename A::size_type size,
        O ownership,
        const SC& shape,
        layout_type l = L,
        const A& alloc = A()
    )
    {
        static_assert(!xtl::is_integral<SC>::value, "shape cannot be a integer");
        (void) ownership;
        using buffer_type = xbuffer_adaptor<xtl::closure_type_t<P>, O, A>;
        constexpr std::size_t N = detail::array_size<SC>::value;
        using return_type = xtensor_adaptor<buffer_type, N, L>;
        buffer_type buf(std::forward<P>(pointer), size, alloc);
        return return_type(std::move(buf), shape, l);
    }

    /**
     * Constructs an xtensor_adaptor of the given dynamically allocated C array,
     * with the specified shape and strides.
     *
     * @ingroup xt_xadapt
     * @param pointer the pointer to the beginning of the dynamic array
     * @param size the size of the dynamic array
     * @param ownership indicates whether the adaptor takes ownership of the array.
     *        Possible values are ``no_ownership()`` or ``acquire_ownership()``
     * @param shape the shape of the xtensor_adaptor
     * @param strides the strides of the xtensor_adaptor
     * @param alloc the allocator used for allocating / deallocating the dynamic array
     */
    template <
        class P,
        class O,
        class SC,
        class SS,
        class A = detail::default_allocator_for_ptr_t<P>,
        XTL_REQUIRES(detail::is_array<std::decay_t<SC>>, detail::not_a_layout<std::decay_t<SS>>)>
    inline xtensor_adaptor<xbuffer_adaptor<xtl::closure_type_t<P>, O, A>, detail::array_size<SC>::value, layout_type::dynamic>
    adapt(P&& pointer, typename A::size_type size, O ownership, SC&& shape, SS&& strides, const A& alloc = A())
    {
        static_assert(!xtl::is_integral<std::decay_t<SC>>::value, "shape cannot be a integer");
        (void) ownership;
        using buffer_type = xbuffer_adaptor<xtl::closure_type_t<P>, O, A>;
        constexpr std::size_t N = detail::array_size<SC>::value;
        using return_type = xtensor_adaptor<buffer_type, N, layout_type::dynamic>;
        buffer_type buf(std::forward<P>(pointer), size, alloc);
        return return_type(
            std::move(buf),
            xtl::forward_sequence<typename return_type::inner_shape_type, SC>(shape),
            xtl::forward_sequence<typename return_type::inner_strides_type, SS>(strides)
        );
    }

    /**
     * Constructs an xtensor_adaptor of the given C array allocated on the stack, with the
     * specified shape and layout.
     *
     * @ingroup xt_xadapt
     * @param c_array the C array allocated on the stack
     * @param shape the shape of the xarray_adaptor
     * @param l the layout_type of the xarray_adaptor
     */
    template <
        layout_type L = XTENSOR_DEFAULT_LAYOUT,
        class T,
        std::size_t N,
        class SC,
        XTL_REQUIRES(detail::is_array<std::decay_t<SC>>)>
    inline auto adapt(T (&c_array)[N], const SC& shape, layout_type l = L)
    {
        return adapt(&c_array[0], N, xt::no_ownership(), shape, l);
    }

    /**
     * Constructs an xtensor_adaptor of the given C array allocated on the stack, with the
     * specified shape and strides.
     *
     * @ingroup xt_xadapt
     * @param c_array the C array allocated on the stack
     * @param shape the shape of the xarray_adaptor
     * @param strides the strides of the xarray_adaptor
     */
    template <
        class T,
        std::size_t N,
        class SC,
        class SS,
        XTL_REQUIRES(detail::is_array<std::decay_t<SC>>, detail::not_a_layout<std::decay_t<SS>>)>
    inline auto adapt(T (&c_array)[N], SC&& shape, SS&& strides)
    {
        return adapt(&c_array[0], N, xt::no_ownership(), std::forward<SC>(shape), std::forward<SS>(strides));
    }

    /**
     * Constructs an non-owning xtensor_fixed_adaptor from a pointer with the
     * specified shape and layout.
     *
     * @ingroup xt_xadapt
     * @param pointer the pointer to adapt
     * @param shape the shape of the xtensor_fixed_adaptor
     */
    template <
        layout_type L = XTENSOR_DEFAULT_LAYOUT,
        class C,
        std::size_t... X,
        XTL_REQUIRES(std::is_pointer<std::remove_reference_t<C>>)>
    inline auto adapt(C&& pointer, const fixed_shape<X...>& /*shape*/)
    {
        using buffer_type = xbuffer_adaptor<C, xt::no_ownership, detail::default_allocator_for_ptr_t<C>>;
        using return_type = xfixed_adaptor<buffer_type, fixed_shape<X...>, L>;
        return return_type(buffer_type(pointer, detail::fixed_compute_size<fixed_shape<X...>>::value));
    }

    template <layout_type L = XTENSOR_DEFAULT_LAYOUT, class C, class T, std::size_t N>
    inline auto adapt(C&& ptr, const T (&shape)[N])
    {
        using shape_type = std::array<std::size_t, N>;
        return adapt(std::forward<C>(ptr), xtl::forward_sequence<shape_type, decltype(shape)>(shape));
    }

#else  // IN_DOXYGEN

    /**
     * Constructs:
     * - an xarray_adaptor if SC is not an array type
     * - an xtensor_adaptor if SC is an array type
     *
     * from the given stl-like container or pointer, with the specified shape and layout.
     * If the adaptor is built from a pointer, it does not take its ownership.
     *
     * @ingroup xt_xadapt
     * @param container the container or pointer to adapt
     * @param shape the shape of the adaptor
     * @param l the layout_type of the adaptor
     */
    template <layout_type L = XTENSOR_DEFAULT_LAYOUT, class C, class SC>
    inline auto adapt(C&& container, const SC& shape, layout_type l = L);

    /**
     * Constructs:
     * - an xarray_adaptor if SC is not an array type
     * - an xtensor_adaptor if SC is an array type
     *
     * from the given stl-like container with the specified shape and strides.
     *
     * @ingroup xt_xadapt
     * @param container the container to adapt
     * @param shape the shape of the adaptor
     * @param strides the strides of the adaptor
     */
    template <class C, class SC, class SS>
    inline auto adapt(C&& container, SC&& shape, SS&& strides);

    /**
     * Constructs:
     * - an xarray_adaptor if SC is not an array type
     * - an xtensor_adaptor if SC is an array type
     *
     * of the given dynamically allocated C array, with the specified shape and layout.
     *
     * @ingroup xt_xadapt
     * @param pointer the pointer to the beginning of the dynamic array
     * @param size the size of the dynamic array
     * @param ownership indicates whether the adaptor takes ownership of the array.
     *        Possible values are ``no_ownership()`` or ``acquire_ownership()``
     * @param shape the shape of the adaptor
     * @param l the layout_type of the adaptor
     * @param alloc the allocator used for allocating / deallocating the dynamic array
     */
    template <layout_type L = XTENSOR_DEFAULT_LAYOUT, class P, class O, class SC, class A = detail::default_allocator_for_ptr_t<P>>
    inline auto adapt(
        P&& pointer,
        typename A::size_type size,
        O ownership,
        const SC& shape,
        layout_type l = L,
        const A& alloc = A()
    );

    /**
     * Constructs:
     * - an xarray_adaptor if SC is not an array type
     * - an xtensor_adaptor if SC is an array type
     *
     * of the given dynamically allocated C array, with the specified shape and strides.
     *
     * @ingroup xt_xadapt
     * @param pointer the pointer to the beginning of the dynamic array
     * @param size the size of the dynamic array
     * @param ownership indicates whether the adaptor takes ownership of the array.
     *        Possible values are ``no_ownership()`` or ``acquire_ownership()``
     * @param shape the shape of the adaptor
     * @param strides the strides of the adaptor
     * @param alloc the allocator used for allocating / deallocating the dynamic array
     */
    template <class P, class O, class SC, class SS, class A = detail::default_allocator_for_ptr_t<P>>
    inline auto
    adapt(P&& pointer, typename A::size_type size, O ownership, SC&& shape, SS&& strides, const A& alloc = A());

    /**
     * Constructs:
     * - an xarray_adaptor if SC is not an array type
     * - an xtensor_adaptor if SC is an array type
     *
     * of the given C array allocated on the stack, with the specified shape and layout.
     *
     * @ingroup xt_xadapt
     * @param c_array the C array allocated on the stack
     * @param shape the shape of the adaptor
     * @param l the layout_type of the adaptor
     */
    template <layout_type L = XTENSOR_DEFAULT_LAYOUT, class T, std::size_t N, class SC>
    inline auto adapt(T (&c_array)[N], const SC& shape, layout_type l = L);

    /**
     * Constructs:
     * - an xarray_adaptor if SC is not an array type
     * - an xtensor_adaptor if SC is an array type
     *
     * of the given C array allocated on the stack, with the
     * specified shape and strides.
     *
     * @ingroup xt_xadapt
     * @param c_array the C array allocated on the stack
     * @param shape the shape of the adaptor
     * @param strides the strides of the adaptor
     */
    template <class T, std::size_t N, class SC, class SS>
    inline auto adapt(T (&c_array)[N], SC&& shape, SS&& strides);

    /**
     * Constructs an non-owning xtensor_fixed_adaptor from a pointer with the
     * specified shape and layout.
     *
     * @ingroup xt_xadapt
     * @param pointer the pointer to adapt
     * @param shape the shape of the xtensor_fixed_adaptor
     */
    template <layout_type L = XTENSOR_DEFAULT_LAYOUT, class C, std::size_t... X>
    inline auto adapt(C&& pointer, const fixed_shape<X...>& /*shape*/);

    /**
     * Constructs a 1-D xtensor_adaptor of the given stl-like container,
     * with the specified layout_type.
     *
     * @ingroup xt_xadapt
     * @param container the container to adapt
     * @param l the layout_type of the xtensor_adaptor
     */
    template <layout_type L = XTENSOR_DEFAULT_LAYOUT, class C>
    inline xtensor_adaptor<C, 1, L> adapt(C&& container, layout_type l = L);

    /**
     * Constructs a 1-D xtensor_adaptor of the given dynamically allocated C array,
     * with the specified layout.
     *
     * @ingroup xt_xadapt
     * @param pointer the pointer to the beginning of the dynamic array
     * @param size the size of the dynamic array
     * @param ownership indicates whether the adaptor takes ownership of the array.
     *        Possible values are ``no_ownership()`` or ``acquire_ownership()``
     * @param l the layout_type of the xtensor_adaptor
     * @param alloc the allocator used for allocating / deallocating the dynamic array
     */
    template <layout_type L = XTENSOR_DEFAULT_LAYOUT, class P, class O, class A = detail::default_allocator_for_ptr_t<P>>
    inline xtensor_adaptor<xbuffer_adaptor<xtl::closure_type_t<P>, O, A>, 1, L>
    adapt(P&& pointer, typename A::size_type size, O ownership, layout_type l = L, const A& alloc = A());

#endif  // IN_DOXYGEN

    /*****************************
     * smart_ptr adapter builder *
     *****************************/

    /**
     * Adapt a smart pointer to a typed memory block (unique_ptr or shared_ptr)
     *
     * @code{.cpp}
     * #include <xtensor/xadapt.hpp>
     * #include <xtensor/xio.hpp>
     *
     * std::shared_ptr<double> sptr(new double[8], std::default_delete<double[]>());
     * sptr.get()[2] = 321.;
     * std::vector<size_t> shape = {4, 2};
     * auto xptr = adapt_smart_ptr(sptr, shape);
     * xptr(1, 3) = 123.;
     * std::cout << xptr;
     * @endcode
     *
     * @ingroup xt_xadapt
     * @param smart_ptr a smart pointer to a memory block of T[]
     * @param shape The desired shape
     * @param l The desired memory layout
     *
     * @return xarray_adaptor for memory
     */
    template <layout_type L = XTENSOR_DEFAULT_LAYOUT, class P, class SC, XTL_REQUIRES(detail::not_an_array<std::decay_t<SC>>)>
    auto adapt_smart_ptr(P&& smart_ptr, const SC& shape, layout_type l = L)
    {
        using buffer_adaptor = xbuffer_adaptor<decltype(smart_ptr.get()), smart_ownership, std::decay_t<P>>;
        return xarray_adaptor<buffer_adaptor, L, std::decay_t<SC>>(
            buffer_adaptor(smart_ptr.get(), compute_size(shape), std::forward<P>(smart_ptr)),
            shape,
            l
        );
    }

    /**
     * Adapt a smart pointer (shared_ptr or unique_ptr)
     *
     * This function allows to automatically adapt a shared or unique pointer to
     * a given shape and operate naturally on it. Memory will be automatically
     * handled by the smart pointer implementation.
     *
     * @code{.cpp}
     * #include <xtensor/xadapt.hpp>
     * #include <xtensor/xio.hpp>
     *
     * struct Buffer {
     *     Buffer(std::vector<double>& buf) : m_buf(buf) {}
     *     ~Buffer() { std::cout << "deleted" << std::endl; }
     *     std::vector<double> m_buf;
     * };
     *
     * auto data = std::vector<double>{1,2,3,4,5,6,7,8};
     * auto shared_buf = std::make_shared<Buffer>(data);
     * auto unique_buf = std::make_unique<Buffer>(data);
     *
     * std::cout << shared_buf.use_count() << std::endl;
     * {
     *     std::vector<size_t> shape = {2, 4};
     *     auto obj = adapt_smart_ptr(shared_buf.get()->m_buf.data(),
     *                                shape, shared_buf);
     *     // Use count increased to 2
     *     std::cout << shared_buf.use_count() << std::endl;
     *     std::cout << obj << std::endl;
     * }
     * // Use count reset to 1
     * std::cout << shared_buf.use_count() << std::endl;
     *
     * {
     *     std::vector<size_t> shape = {2, 4};
     *     auto obj = adapt_smart_ptr(unique_buf.get()->m_buf.data(),
     *                                shape, std::move(unique_buf));
     *     std::cout << obj << std::endl;
     * }
     * @endcode
     *
     * @ingroup xt_xadapt
     * @param data_ptr A pointer to a typed data block (e.g. double*)
     * @param shape The desired shape
     * @param smart_ptr A smart pointer to move or copy, in order to manage memory
     * @param l The desired memory layout
     *
     * @return xarray_adaptor on the memory
     */
    template <
        layout_type L = XTENSOR_DEFAULT_LAYOUT,
        class P,
        class SC,
        class D,
        XTL_REQUIRES(detail::not_an_array<std::decay_t<SC>>, detail::not_a_layout<std::decay_t<D>>)>
    auto adapt_smart_ptr(P&& data_ptr, const SC& shape, D&& smart_ptr, layout_type l = L)
    {
        using buffer_adaptor = xbuffer_adaptor<P, smart_ownership, std::decay_t<D>>;

        return xarray_adaptor<buffer_adaptor, L, std::decay_t<SC>>(
            buffer_adaptor(data_ptr, compute_size(shape), std::forward<D>(smart_ptr)),
            shape,
            l
        );
    }

    /**
     * Adapt a smart pointer to a typed memory block (unique_ptr or shared_ptr)
     *
     * @code{.cpp}
     * #include <xtensor/xadapt.hpp>
     * #include <xtensor/xio.hpp>
     *
     * std::shared_ptr<double> sptr(new double[8], std::default_delete<double[]>());
     * sptr.get()[2] = 321.;
     * auto xptr = adapt_smart_ptr(sptr, {4, 2});
     * xptr(1, 3) = 123.;
     * std::cout << xptr;
     * @endcode
     *
     * @ingroup xt_xadapt
     * @param smart_ptr a smart pointer to a memory block of T[]
     * @param shape The desired shape
     * @param l The desired memory layout
     *
     * @return xtensor_adaptor for memory
     */
    template <layout_type L = XTENSOR_DEFAULT_LAYOUT, class P, class I, std::size_t N>
    auto adapt_smart_ptr(P&& smart_ptr, const I (&shape)[N], layout_type l = L)
    {
        using buffer_adaptor = xbuffer_adaptor<decltype(smart_ptr.get()), smart_ownership, std::decay_t<P>>;
        std::array<std::size_t, N> fshape = xtl::forward_sequence<std::array<std::size_t, N>, decltype(shape)>(
            shape
        );
        return xtensor_adaptor<buffer_adaptor, N, L>(
            buffer_adaptor(smart_ptr.get(), compute_size(fshape), std::forward<P>(smart_ptr)),
            std::move(fshape),
            l
        );
    }

    /**
     * Adapt a smart pointer (shared_ptr or unique_ptr)
     *
     * This function allows to automatically adapt a shared or unique pointer to
     * a given shape and operate naturally on it. Memory will be automatically
     * handled by the smart pointer implementation.
     *
     * @code{.cpp}
     * #include <xtensor/xadapt.hpp>
     * #include <xtensor/xio.hpp>
     *
     * struct Buffer {
     *     Buffer(std::vector<double>& buf) : m_buf(buf) {}
     *     ~Buffer() { std::cout << "deleted" << std::endl; }
     *     std::vector<double> m_buf;
     * };
     *
     * auto data = std::vector<double>{1,2,3,4,5,6,7,8};
     * auto shared_buf = std::make_shared<Buffer>(data);
     * auto unique_buf = std::make_unique<Buffer>(data);
     *
     * std::cout << shared_buf.use_count() << std::endl;
     * {
     *     auto obj = adapt_smart_ptr(shared_buf.get()->m_buf.data(),
     *                                {2, 4}, shared_buf);
     *     // Use count increased to 2
     *     std::cout << shared_buf.use_count() << std::endl;
     *     std::cout << obj << std::endl;
     * }
     * // Use count reset to 1
     * std::cout << shared_buf.use_count() << std::endl;
     *
     * {
     *     auto obj = adapt_smart_ptr(unique_buf.get()->m_buf.data(),
     *                                {2, 4}, std::move(unique_buf));
     *     std::cout << obj << std::endl;
     * }
     * @endcode
     *
     * @ingroup xt_xadapt
     * @param data_ptr A pointer to a typed data block (e.g. double*)
     * @param shape The desired shape
     * @param smart_ptr A smart pointer to move or copy, in order to manage memory
     * @param l The desired memory layout
     *
     * @return xtensor_adaptor on the memory
     */
    template <
        layout_type L = XTENSOR_DEFAULT_LAYOUT,
        class P,
        class I,
        std::size_t N,
        class D,
        XTL_REQUIRES(detail::not_a_layout<std::decay_t<D>>)>
    auto adapt_smart_ptr(P&& data_ptr, const I (&shape)[N], D&& smart_ptr, layout_type l = L)
    {
        using buffer_adaptor = xbuffer_adaptor<P, smart_ownership, std::decay_t<D>>;
        std::array<std::size_t, N> fshape = xtl::forward_sequence<std::array<std::size_t, N>, decltype(shape)>(
            shape
        );

        return xtensor_adaptor<buffer_adaptor, N, L>(
            buffer_adaptor(data_ptr, compute_size(fshape), std::forward<D>(smart_ptr)),
            std::move(fshape),
            l
        );
    }

    /**
     * @brief xtensor adaptor for a pointer.
     *
     * Construct for example with:
     *
     * @code{.cpp}
     * #include <xtensor/xadapt.hpp>
     *
     * std::array<size_t, 2> shape = {2, 2};
     * std::vector<double> data = {1, 2, 3, 4};
     *
     * xt::xtensor_pointer<double, 2> a = xt::adapt(data.data(), 4, xt::no_ownership(), shape);
     * @endcode
     *
     * @ingroup xt_xadapt
     * @tparam T The data type (e.g. ``double``).
     * @tparam N The number of dimensions.
     * @tparam L The xt::layout_type() of the xtensor.
     */
    template <class T, std::size_t N, layout_type L = XTENSOR_DEFAULT_LAYOUT>
    using xtensor_pointer = xtensor_adaptor<
        xbuffer_adaptor<xtl::closure_type_t<T*>, xt::no_ownership, detail::default_allocator_for_ptr_t<T>>,
        N,
        L>;

    /**
     * @brief xarray adaptor for a pointer.
     *
     * Construct for example with:
     *
     * @code{.cpp}
     * #include <xtensor/xadapt.hpp>
     *
     * std::vector<int> data(4, 0);
     * xt::svector<size_t> shape({2, 2});
     *
     * xt::xarray_pointer<int> a = xt::adapt(data.data(), data.size(), xt::no_ownership(), shape);
     * @endcode
     *
     * @ingroup xt_xadapt
     * @tparam T The data type (e.g. ``double``).
     * @tparam L The xt::layout_type() of the xarray.
     * @tparam SC The shape container type (e.g. ``xt::svector<size_t>``). Default matches
     *      xt::adapt(P&&, typename A::size_type, O, const SC&, layout_type, const A& alloc)
     */
    template <
        class T,
        layout_type L = XTENSOR_DEFAULT_LAYOUT,
        class SC = XTENSOR_DEFAULT_SHAPE_CONTAINER(T, std::allocator<std::size_t>, std::allocator<std::size_t>)>
    using xarray_pointer = xarray_adaptor<
        xbuffer_adaptor<xtl::closure_type_t<T*>, xt::no_ownership, detail::default_allocator_for_ptr_t<T>>,
        L,
        SC>;
}

#endif
