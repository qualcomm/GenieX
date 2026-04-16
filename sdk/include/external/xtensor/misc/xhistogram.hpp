/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

/**
 * @brief construct histogram
 */

#ifndef XTENSOR_HISTOGRAM_HPP
#define XTENSOR_HISTOGRAM_HPP

#include "../containers/xtensor.hpp"
#include "../misc/xset_operation.hpp"
#include "../misc/xsort.hpp"
#include "../views/xview.hpp"

using namespace xt::placeholders;

namespace xt
{
    /**
     * @ingroup digitize
     * @brief Return the indices of the bins to which each value in input array belongs.
     *
     * @param data The data.
     * @param bin_edges The bin-edges. It has to be 1-dimensional and monotonic.
     * @param right Indicating whether the intervals include the right or the left bin edge.
     * @return Output array of indices, of same shape as x.
     */
    template <class E1, class E2>
    inline auto digitize(E1&& data, E2&& bin_edges, bool right = false)
    {
        XTENSOR_ASSERT(bin_edges.dimension() == 1);
        XTENSOR_ASSERT(bin_edges.size() >= 2);
        XTENSOR_ASSERT(std::is_sorted(bin_edges.cbegin(), bin_edges.cend()));
        XTENSOR_ASSERT(xt::amin(data)[0] >= bin_edges[0]);
        XTENSOR_ASSERT(xt::amax(data)[0] <= bin_edges[bin_edges.size() - 1]);

        return xt::searchsorted(std::forward<E2>(bin_edges), std::forward<E1>(data), right);
    }

    namespace detail
    {
        template <class R = double, class E1, class E2, class E3>
        inline auto histogram_imp(E1&& data, E2&& bin_edges, E3&& weights, bool density, bool equal_bins)
        {
            using size_type = common_size_type_t<std::decay_t<E1>, std::decay_t<E2>, std::decay_t<E3>>;
            using value_type = typename std::decay_t<E3>::value_type;

            XTENSOR_ASSERT(data.dimension() == 1);
            XTENSOR_ASSERT(weights.dimension() == 1);
            XTENSOR_ASSERT(bin_edges.dimension() == 1);
            XTENSOR_ASSERT(weights.size() == data.size());
            XTENSOR_ASSERT(bin_edges.size() >= 2);
            XTENSOR_ASSERT(std::is_sorted(bin_edges.cbegin(), bin_edges.cend()));

            size_t n_bins = bin_edges.size() - 1;
            xt::xtensor<value_type, 1> count = xt::zeros<value_type>({n_bins});

            if (equal_bins)
            {
                std::array<typename std::decay_t<E2>::value_type, 2> bounds = xt::minmax(bin_edges)();
                auto left = static_cast<double>(bounds[0]);
                auto right = static_cast<double>(bounds[1]);
                double norm = 1. / (right - left);
                for (size_t i = 0; i < data.size(); ++i)
                {
                    auto v = static_cast<double>(data(i));
                    // left and right are not bounds of data
                    if (v >= left && v < right)
                    {
                        auto i_bin = static_cast<size_t>(static_cast<double>(n_bins) * (v - left) * norm);
                        count(i_bin) += weights(i);
                    }
                    else if (v == right)
                    {
                        count(n_bins - 1) += weights(i);
                    }
                }
            }
            else
            {
                auto sorter = xt::argsort(data);

                size_type ibin = 0;

                for (auto& idx : sorter)
                {
                    auto item = data[idx];

                    if (item < bin_edges[0])
                    {
                        continue;
                    }

                    if (item > bin_edges[n_bins])
                    {
                        break;
                    }

                    while (item >= bin_edges[ibin + 1] && ibin < n_bins - 1)
                    {
                        ++ibin;
                    }

                    count[ibin] += weights[idx];
                }
            }

            xt::xtensor<R, 1> prob = xt::cast<R>(count);

            if (density)
            {
                R n = static_cast<R>(data.size());
                for (size_type i = 0; i < bin_edges.size() - 1; ++i)
                {
                    prob[i] /= (static_cast<R>(bin_edges[i + 1] - bin_edges[i]) * n);
                }
            }

            return prob;
        }

    }  // detail

    /**
     * @ingroup histogram
     * @brief Compute the histogram of a set of data.
     *
     * @param data The data.
     * @param bin_edges The bin-edges. It has to be 1-dimensional and monotonic.
     * @param weights Weight factors corresponding to each data-point.
     * @param density If true the resulting integral is normalized to 1. [default: false]
     * @return An one-dimensional xarray<double>, length: bin_edges.size()-1.
     */
    template <class R = double, class E1, class E2, class E3>
    inline auto histogram(E1&& data, E2&& bin_edges, E3&& weights, bool density = false)
    {
        return detail::histogram_imp<R>(
            std::forward<E1>(data),
            std::forward<E2>(bin_edges),
            std::forward<E3>(weights),
            density,
            false
        );
    }

    /**
     * @ingroup histogram
     * @brief Compute the histogram of a set of data.
     *
     * @param data The data.
     * @param bin_edges The bin-edges.
     * @param density If true the resulting integral is normalized to 1. [default: false]
     * @return An one-dimensional xarray<double>, length: bin_edges.size()-1.
     */
    template <class R = double, class E1, class E2>
    inline auto histogram(E1&& data, E2&& bin_edges, bool density = false)
    {
        using value_type = typename std::decay_t<E1>::value_type;

        auto n = data.size();

        return detail::histogram_imp<R>(
            std::forward<E1>(data),
            std::forward<E2>(bin_edges),
            xt::ones<value_type>({n}),
            density,
            false
        );
    }

    /**
     * @ingroup histogram
     * @brief Compute the histogram of a set of data.
     *
     * @param data The data.
     * @param bins The number of bins. [default: 10]
     * @param density If true the resulting integral is normalized to 1. [default: false]
     * @return An one-dimensional xarray<double>, length: bin_edges.size()-1.
     */
    template <class R = double, class E1>
    inline auto histogram(E1&& data, std::size_t bins = 10, bool density = false)
    {
        using value_type = typename std::decay_t<E1>::value_type;

        auto n = data.size();

        return detail::histogram_imp<R>(
            std::forward<E1>(data),
            histogram_bin_edges(data, xt::ones<value_type>({n}), bins),
            xt::ones<value_type>({n}),
            density,
            true
        );
    }

    /**
     * @ingroup histogram
     * @brief Compute the histogram of a set of data.
     *
     * @param data The data.
     * @param bins The number of bins.
     * @param left The lower-most edge.
     * @param right The upper-most edge.
     * @param density If true the resulting integral is normalized to 1. [default: false]
     * @return An one-dimensional xarray<double>, length: bin_edges.size()-1.
     */
    template <class R = double, class E1, class E2>
    inline auto histogram(E1&& data, std::size_t bins, E2 left, E2 right, bool density = false)
    {
        using value_type = typename std::decay_t<E1>::value_type;

        auto n = data.size();

        return detail::histogram_imp<R>(
            std::forward<E1>(data),
            histogram_bin_edges(data, left, right, bins),
            xt::ones<value_type>({n}),
            density,
            true
        );
    }

    /**
     * @ingroup histogram
     * @brief Compute the histogram of a set of data.
     *
     * @param data The data.
     * @param bins The number of bins.
     * @param weights Weight factors corresponding to each data-point.
     * @param density If true the resulting integral is normalized to 1. [default: false]
     * @return An one-dimensional xarray<double>, length: bin_edges.size()-1.
     */
    template <class R = double, class E1, class E2>
    inline auto histogram(E1&& data, std::size_t bins, E2&& weights, bool density = false)
    {
        return detail::histogram_imp<R>(
            std::forward<E1>(data),
            histogram_bin_edges(data, weights, bins),
            std::forward<E2>(weights),
            density,
            true
        );
    }

    /**
     * @ingroup histogram
     * @brief Compute the histogram of a set of data.
     *
     * @param data The data.
     * @param bins The number of bins.
     * @param left The lower-most edge.
     * @param right The upper-most edge.
     * @param weights Weight factors corresponding to each data-point.
     * @param density If true the resulting integral is normalized to 1. [default: false]
     * @return An one-dimensional xarray<double>, length: bin_edges.size()-1.
     */
    template <class R = double, class E1, class E2, class E3>
    inline auto histogram(E1&& data, std::size_t bins, E2&& weights, E3 left, E3 right, bool density = false)
    {
        return detail::histogram_imp<R>(
            std::forward<E1>(data),
            histogram_bin_edges(data, weights, left, right, bins),
            std::forward<E2>(weights),
            density,
            true
        );
    }

    /**
     * @ingroup histogram
     * @brief Defines different algorithms to be used in "histogram_bin_edges"
     */
    enum class histogram_algorithm
    {
        automatic,
        linspace,
        logspace,
        uniform
    };

    /**
     * @ingroup histogram
     * @brief Compute the bin-edges of a histogram of a set of data using different algorithms.
     *
     * @param data The data.
     * @param weights Weight factors corresponding to each data-point.
     * @param left The lower-most edge.
     * @param right The upper-most edge.
     * @param bins The number of bins. [default: 10]
     * @param mode The type of algorithm to use. [default: "auto"]
     * @return An one-dimensional xarray<double>, length: bins+1.
     */
    template <class E1, class E2, class E3>
    inline auto histogram_bin_edges(
        E1&& data,
        E2&& weights,
        E3 left,
        E3 right,
        std::size_t bins = 10,
        histogram_algorithm mode = histogram_algorithm::automatic
    )
    {
        // counter and return type
        using size_type = common_size_type_t<std::decay_t<E1>, std::decay_t<E2>>;
        using value_type = typename std::decay_t<E1>::value_type;
        using weights_type = typename std::decay_t<E2>::value_type;

        // basic checks
        // - rank
        XTENSOR_ASSERT(data.dimension() == 1);
        XTENSOR_ASSERT(weights.dimension() == 1);
        // - size
        XTENSOR_ASSERT(weights.size() == data.size());
        // - bounds
        XTENSOR_ASSERT(left <= right);
        // - non-empty
        XTENSOR_ASSERT(bins > std::size_t(0));

        // act on different modes
        switch (mode)
        {
            // bins of equal width
            case histogram_algorithm::automatic:
            {
                xt::xtensor<value_type, 1> bin_edges = xt::linspace<value_type>(left, right, bins + 1);
                return bin_edges;
            }

            // bins of equal width
            case histogram_algorithm::linspace:
            {
                xt::xtensor<value_type, 1> bin_edges = xt::linspace<value_type>(left, right, bins + 1);
                return bin_edges;
            }

            // bins of logarithmically increasing size
            case histogram_algorithm::logspace:
            {
                using rhs_value_type = std::conditional_t<xtl::is_integral<value_type>::value, double, value_type>;

                xtensor<value_type, 1> bin_edges = xt::cast<value_type>(
                    xt::logspace<rhs_value_type>(std::log10(left), std::log10(right), bins + 1)
                );

                // TODO: replace above with below after 'xsimd' fix
                // xt::xtensor<value_type,1> bin_edges = xt::logspace<value_type>(
                //     std::log10(left), std::log10(right), bins+1);

                return bin_edges;
            }

            // same amount of data in each bin
            case histogram_algorithm::uniform:
            {
                // indices that sort "data"
                auto sorter = xt::argsort(data);

                // histogram: all of equal 'height'
                // - height
                weights_type w = xt::sum<weights_type>(weights)[0] / static_cast<weights_type>(bins);
                // - apply to all bins
                xt::xtensor<weights_type, 1> count = w * xt::ones<weights_type>({bins});

                // take cumulative sum, to act as easy look-up
                count = xt::cumsum(count);

                // edges
                // - allocate
                std::vector<size_t> shape = {bins + 1};
                xt::xtensor<value_type, 1> bin_edges = xtensor<value_type, 1>::from_shape(shape);
                // - first/last edge
                bin_edges[0] = left;
                bin_edges[bins] = right;
                // - cumulative weight
                weights_type cum_weight = static_cast<weights_type>(0);
                // - current bin
                size_type ibin = 0;
                // - loop to find interior bin-edges
                for (size_type i = 0; i < weights.size(); ++i)
                {
                    if (cum_weight >= count[ibin])
                    {
                        bin_edges[ibin + 1] = data[sorter[i]];
                        ++ibin;
                    }
                    cum_weight += weights[sorter[i]];
                }
                return bin_edges;
            }

            // bins of equal width
            default:
            {
                xt::xtensor<value_type, 1> bin_edges = xt::linspace<value_type>(left, right, bins + 1);
                return bin_edges;
            }
        }
    }

    /**
     * @ingroup histogram
     * @brief Compute the bin-edges of a histogram of a set of data using different algorithms.
     *
     * @param data The data.
     * @param weights Weight factors corresponding to each data-point.
     * @param bins The number of bins. [default: 10]
     * @param mode The type of algorithm to use. [default: "auto"]
     * @return An one-dimensional xarray<double>, length: bins+1.
     */
    template <class E1, class E2>
    inline auto histogram_bin_edges(
        E1&& data,
        E2&& weights,
        std::size_t bins = 10,
        histogram_algorithm mode = histogram_algorithm::automatic
    )
    {
        using value_type = typename std::decay_t<E1>::value_type;
        std::array<value_type, 2> left_right;
        left_right = xt::minmax(data)();

        return histogram_bin_edges(
            std::forward<E1>(data),
            std::forward<E2>(weights),
            left_right[0],
            left_right[1],
            bins,
            mode
        );
    }

    /**
     * @ingroup histogram
     * @brief Compute the bin-edges of a histogram of a set of data using different algorithms.
     *
     * @param data The data.
     * @param bins The number of bins. [default: 10]
     * @param mode The type of algorithm to use. [default: "auto"]
     * @return An one-dimensional xarray<double>, length: bins+1.
     */
    template <class E1>
    inline auto
    histogram_bin_edges(E1&& data, std::size_t bins = 10, histogram_algorithm mode = histogram_algorithm::automatic)
    {
        using value_type = typename std::decay_t<E1>::value_type;

        auto n = data.size();
        std::array<value_type, 2> left_right;
        left_right = xt::minmax(data)();

        return histogram_bin_edges(
            std::forward<E1>(data),
            xt::ones<value_type>({n}),
            left_right[0],
            left_right[1],
            bins,
            mode
        );
    }

    /**
     * @ingroup histogram
     * @brief Compute the bin-edges of a histogram of a set of data using different algorithms.
     *
     * @param data The data.
     * @param left The lower-most edge.
     * @param right The upper-most edge.
     * @param bins The number of bins. [default: 10]
     * @param mode The type of algorithm to use. [default: "auto"]
     * @return An one-dimensional xarray<double>, length: bins+1.
     */
    template <class E1, class E2>
    inline auto histogram_bin_edges(
        E1&& data,
        E2 left,
        E2 right,
        std::size_t bins = 10,
        histogram_algorithm mode = histogram_algorithm::automatic
    )
    {
        using value_type = typename std::decay_t<E1>::value_type;

        auto n = data.size();

        return histogram_bin_edges(std::forward<E1>(data), xt::ones<value_type>({n}), left, right, bins, mode);
    }

    /**
     * Count number of occurrences of each value in array of non-negative ints.
     *
     * The number of bins (of size 1) is one larger than the largest value in x.
     * If minlength is specified, there will be at least this number of bins in
     * the output array (though it will be longer if necessary, depending on the
     * contents of x). Each bin gives the number of occurrences of its index
     * value in x. If weights is specified the input array is weighted by it,
     * i.e. if a value ``n`` is found at position ``i``, ``out[n] += weight[i]``
     * instead of ``out[n] += 1``.
     *
     * @param data the 1D container with integers to count into bins
     * @param weights a 1D container with the same number of elements as ``data``
     * @param minlength The minlength
     *
     * @return 1D container with the bincount
     */
    template <class E1, class E2, XTL_REQUIRES(is_xexpression<std::decay_t<E2>>)>
    inline auto bincount(E1&& data, E2&& weights, std::size_t minlength = 0)
    {
        using result_value_type = typename std::decay_t<E2>::value_type;
        using input_value_type = typename std::decay_t<E1>::value_type;
        using size_type = typename std::decay_t<E1>::size_type;

        static_assert(
            xtl::is_integral<typename std::decay_t<E1>::value_type>::value,
            "Bincount data has to be integral type."
        );
        XTENSOR_ASSERT(data.dimension() == 1);
        XTENSOR_ASSERT(weights.dimension() == 1);

        std::array<input_value_type, 2> left_right;
        left_right = xt::minmax(data)();

        if (left_right[0] < input_value_type(0))
        {
            XTENSOR_THROW(std::runtime_error, "Data argument for bincount can only contain positive integers!");
        }

        xt::xtensor<result_value_type, 1> res = xt::zeros<result_value_type>(
            {(std::max)(minlength, std::size_t(left_right[1] + 1))}
        );

        for (size_type i = 0; i < data.size(); ++i)
        {
            res(data(i)) += weights(i);
        }

        return res;
    }

    template <class E1>
    inline auto bincount(E1&& data, std::size_t minlength = 0)
    {
        return bincount(
            std::forward<E1>(data),
            xt::ones<typename std::decay_t<E1>::value_type>(data.shape()),
            minlength
        );
    }

    /**
     * Get the number of items in each bin, given the fraction of items per bin.
     * The output is such that the total number of items of all bins is exactly "N".
     *
     * @param N the number of items to distribute
     * @param weights fraction of items per bin: a 1D container whose size is the number of bins
     *
     * @return 1D container with the number of items per bin
     */
    template <class E>
    inline xt::xtensor<size_t, 1> bin_items(size_t N, E&& weights)
    {
        if (weights.size() <= std::size_t(1))
        {
            xt::xtensor<size_t, 1> n = N * xt::ones<size_t>({1});
            return n;
        }

#ifdef XTENSOR_ENABLE_ASSERT
        using value_type = typename std::decay_t<E>::value_type;

        XTENSOR_ASSERT(xt::all(weights >= static_cast<value_type>(0)));
        XTENSOR_ASSERT(xt::sum(weights)() > static_cast<value_type>(0));
#endif

        xt::xtensor<double, 1> P = xt::cast<double>(weights) / static_cast<double>(xt::sum(weights)());
        xt::xtensor<size_t, 1> n = xt::ceil(static_cast<double>(N) * P);

        if (xt::sum(n)() == N)
        {
            return n;
        }

        xt::xtensor<size_t, 1> d = xt::zeros<size_t>(P.shape());
        xt::xtensor<size_t, 1> sorter = xt::argsort(P);
        sorter = xt::view(sorter, xt::range(P.size(), _, -1));
        sorter = xt::view(sorter, xt::range(0, xt::sum(n)(0) - N));
        xt::view(d, xt::keep(sorter)) = 1;
        n -= d;

        return n;
    }

    /**
     * Get the number of items in each bin, with each bin having approximately the same number of
     * items in it,under the constraint that the total number of items of all bins is exactly "N".
     *
     * @param N the number of items to distribute
     * @param bins the number of bins
     *
     * @return 1D container with the number of items per bin
     */
    inline xt::xtensor<size_t, 1> bin_items(size_t N, size_t bins)
    {
        return bin_items(N, xt::ones<double>({bins}));
    }
}

#endif
