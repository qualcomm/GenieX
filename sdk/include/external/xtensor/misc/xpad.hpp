/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_PAD_HPP
#define XTENSOR_PAD_HPP

#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../views/xstrided_view.hpp"
#include "../views/xview.hpp"

using namespace xt::placeholders;  // to enable _ syntax

namespace xt
{
    /**
     * @brief Defines different algorithms to be used in ``xt::pad``:
     * - ``constant``: Pads with a constant value.
     * - ``symmetric``: Pads with the reflection of the vector mirrored along the edge of the array.
     * - ``reflect``: Pads with the reflection of the vector mirrored on the first and last values
     *   of the vector along each axis.
     * - ``wrap``: Pads with the wrap of the vector along the axis. The first values are used to pad
     *   the end and the end values are used to pad the beginning.
     * - ``periodic`` : ``== wrap`` (pads with periodic repetitions of the vector).
     *
     * OpenCV to xtensor:
     * - ``BORDER_CONSTANT == constant``
     * - ``BORDER_REFLECT == symmetric``
     * - ``BORDER_REFLECT_101 == reflect``
     * - ``BORDER_WRAP == wrap``
     */
    enum class pad_mode
    {
        constant,
        symmetric,
        reflect,
        wrap,
        periodic,
        edge
    };

    namespace detail
    {
        template <class S, class T>
        inline bool check_pad_width(const std::vector<std::vector<S>>& pad_width, const T& shape)
        {
            if (pad_width.size() != shape.size())
            {
                return false;
            }

            return true;
        }
    }

    /**
     * @brief Pad an array.
     *
     * @param e The array.
     * @param pad_width Number of values padded to the edges of each axis:
     * `{{before_1, after_1}, ..., {before_N, after_N}}`.
     * @param mode The type of algorithm to use. [default: `xt::pad_mode::constant`].
     * @param constant_value The value to set the padded values for each axis
     * (used in `xt::pad_mode::constant`).
     * @return The padded array.
     */
    template <class E, class S = typename std::decay_t<E>::size_type, class V = typename std::decay_t<E>::value_type>
    inline auto
    pad(E&& e,
        const std::vector<std::vector<S>>& pad_width,
        pad_mode mode = pad_mode::constant,
        V constant_value = 0)
    {
        XTENSOR_ASSERT(detail::check_pad_width(pad_width, e.shape()));

        using size_type = typename std::decay_t<E>::size_type;
        using return_type = temporary_type_t<E>;

        // place the original array in the center

        auto new_shape = e.shape();
        xt::xstrided_slice_vector sv;
        sv.reserve(e.shape().size());
        for (size_type axis = 0; axis < e.shape().size(); ++axis)
        {
            size_type nb = static_cast<size_type>(pad_width[axis][0]);
            size_type ne = static_cast<size_type>(pad_width[axis][1]);
            size_type ns = nb + e.shape(axis) + ne;
            new_shape[axis] = ns;
            sv.push_back(xt::range(nb, nb + e.shape(axis)));
        }

        if (mode == pad_mode::constant)
        {
            return_type out(new_shape, constant_value);
            xt::strided_view(out, sv) = e;
            return out;
        }

        return_type out(new_shape);
        xt::strided_view(out, sv) = e;

        // construct padded regions based on original image

        xt::xstrided_slice_vector svs(e.shape().size(), xt::all());
        xt::xstrided_slice_vector svt(e.shape().size(), xt::all());

        for (size_type axis = 0; axis < e.shape().size(); ++axis)
        {
            size_type nb = static_cast<size_type>(pad_width[axis][0]);
            size_type ne = static_cast<size_type>(pad_width[axis][1]);

            if (nb > static_cast<size_type>(0))
            {
                svt[axis] = xt::range(0, nb);

                if (mode == pad_mode::wrap || mode == pad_mode::periodic)
                {
                    XTENSOR_ASSERT(nb <= e.shape(axis));
                    svs[axis] = xt::range(e.shape(axis), nb + e.shape(axis));
                    xt::strided_view(out, svt) = xt::strided_view(out, svs);
                }
                else if (mode == pad_mode::symmetric)
                {
                    XTENSOR_ASSERT(nb <= e.shape(axis));
                    svs[axis] = xt::range(2 * nb - 1, nb - 1, -1);
                    xt::strided_view(out, svt) = xt::strided_view(out, svs);
                }
                else if (mode == pad_mode::reflect)
                {
                    XTENSOR_ASSERT(nb <= e.shape(axis) - 1);
                    svs[axis] = xt::range(2 * nb, nb, -1);
                    xt::strided_view(out, svt) = xt::strided_view(out, svs);
                }
                else if (mode == pad_mode::edge)
                {
                    svs[axis] = xt::range(nb, nb + 1);
                    xt::strided_view(out, svt) = xt::broadcast(
                        xt::strided_view(out, svs),
                        xt::strided_view(out, svt).shape()
                    );
                }
            }

            if (ne > static_cast<size_type>(0))
            {
                svt[axis] = xt::range(out.shape(axis) - ne, out.shape(axis));

                if (mode == pad_mode::wrap || mode == pad_mode::periodic)
                {
                    XTENSOR_ASSERT(ne <= e.shape(axis));
                    svs[axis] = xt::range(nb, nb + ne);
                    xt::strided_view(out, svt) = xt::strided_view(out, svs);
                }
                else if (mode == pad_mode::symmetric)
                {
                    XTENSOR_ASSERT(ne <= e.shape(axis));
                    if (ne == nb + e.shape(axis))
                    {
                        svs[axis] = xt::range(nb + e.shape(axis) - 1, _, -1);
                    }
                    else
                    {
                        svs[axis] = xt::range(nb + e.shape(axis) - 1, nb + e.shape(axis) - ne - 1, -1);
                    }
                    xt::strided_view(out, svt) = xt::strided_view(out, svs);
                }
                else if (mode == pad_mode::reflect)
                {
                    XTENSOR_ASSERT(ne <= e.shape(axis) - 1);
                    if (ne == nb + e.shape(axis) - 1)
                    {
                        svs[axis] = xt::range(nb + e.shape(axis) - 2, _, -1);
                    }
                    else
                    {
                        svs[axis] = xt::range(nb + e.shape(axis) - 2, nb + e.shape(axis) - ne - 2, -1);
                    }
                    xt::strided_view(out, svt) = xt::strided_view(out, svs);
                }
                else if (mode == pad_mode::edge)
                {
                    svs[axis] = xt::range(out.shape(axis) - ne - 1, out.shape(axis) - ne);
                    xt::strided_view(out, svt) = xt::broadcast(
                        xt::strided_view(out, svs),
                        xt::strided_view(out, svt).shape()
                    );
                }
            }

            svs[axis] = xt::all();
            svt[axis] = xt::all();
        }

        return out;
    }

    /**
     * @brief Pad an array.
     *
     * @param e The array.
     * @param pad_width Number of values padded to the edges of each axis:
     * `{before, after}`.
     * @param mode The type of algorithm to use. [default: `xt::pad_mode::constant`].
     * @param constant_value The value to set the padded values for each axis
     * (used in `xt::pad_mode::constant`).
     * @return The padded array.
     */
    template <class E, class S = typename std::decay_t<E>::size_type, class V = typename std::decay_t<E>::value_type>
    inline auto
    pad(E&& e, const std::vector<S>& pad_width, pad_mode mode = pad_mode::constant, V constant_value = 0)
    {
        std::vector<std::vector<S>> pw(e.shape().size(), pad_width);

        return pad(e, pw, mode, constant_value);
    }

    /**
     * @brief Pad an array.
     *
     * @param e The array.
     * @param pad_width Number of values padded to the edges of each axis.
     * @param mode The type of algorithm to use. [default: `xt::pad_mode::constant`].
     * @param constant_value The value to set the padded values for each axis
     * (used in `xt::pad_mode::constant`).
     * @return The padded array.
     */
    template <class E, class S = typename std::decay_t<E>::size_type, class V = typename std::decay_t<E>::value_type>
    inline auto pad(E&& e, S pad_width, pad_mode mode = pad_mode::constant, V constant_value = 0)
    {
        std::vector<std::vector<S>> pw(e.shape().size(), {pad_width, pad_width});

        return pad(e, pw, mode, constant_value);
    }

    namespace detail
    {

        template <class E, class S>
        inline auto tile(E&& e, const S& reps)
        {
            using size_type = typename std::decay_t<E>::size_type;

            using return_type = temporary_type_t<E>;

            XTENSOR_ASSERT(e.shape().size() == reps.size());

            using new_shape_type = typename return_type::shape_type;
            auto new_shape = xtl::make_sequence<new_shape_type>(e.shape().size());

            xt::xstrided_slice_vector sv(reps.size());

            for (size_type axis = 0; axis < reps.size(); ++axis)
            {
                new_shape[axis] = e.shape(axis) * reps[axis];
                sv[axis] = xt::range(0, e.shape(axis));
            }
            return_type out(new_shape);

            xt::strided_view(out, sv) = e;

            xt::xstrided_slice_vector svs(e.shape().size(), xt::all());
            xt::xstrided_slice_vector svt(e.shape().size(), xt::all());

            for (size_type axis = 0; axis < e.shape().size(); ++axis)
            {
                for (size_type i = 1; i < static_cast<size_type>(reps[axis]); ++i)
                {
                    svs[axis] = xt::range(0, e.shape(axis));
                    svt[axis] = xt::range(i * e.shape(axis), (i + 1) * e.shape(axis));
                    xt::strided_view(out, svt) = xt::strided_view(out, svs);
                    svs[axis] = xt::all();
                    svt[axis] = xt::all();
                }
            }

            return out;
        }
    }

    /**
     * @brief Tile an array.
     *
     * @param e The array.
     * @param reps The number of repetitions of A along each axis.
     * @return The tiled array.
     */

    template <class E, class S = typename std::decay_t<E>::size_type>
    inline auto tile(E&& e, std::initializer_list<S> reps)
    {
        return detail::tile(std::forward<E>(e), std::vector<S>{reps});
    }

    template <class E, class C, XTL_REQUIRES(std::negation<xtl::is_integral<C>>)>
    inline auto tile(E&& e, const C& reps)
    {
        return detail::tile(std::forward<E>(e), reps);
    }

    /**
     * @brief Tile an array.
     *
     * @param e The array.
     * @param reps The number of repetitions of A along the first axis.
     * @return The tiled array.
     */
    template <class E, class S = typename std::decay_t<E>::size_type, XTL_REQUIRES(xtl::is_integral<S>)>
    inline auto tile(E&& e, S reps)
    {
        std::vector<S> tw(e.shape().size(), static_cast<S>(1));
        tw[0] = reps;
        return detail::tile(std::forward<E>(e), tw);
    }
}

#endif
