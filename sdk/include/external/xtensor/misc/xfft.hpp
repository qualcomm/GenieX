#ifdef XTENSOR_USE_TBB
#include <oneapi/tbb.h>
#endif
#include <stdexcept>

#include <xtl/xcomplex.hpp>

#include "../containers/xarray.hpp"
#include "../core/xmath.hpp"
#include "../core/xnoalias.hpp"
#include "../generators/xbuilder.hpp"
#include "../misc/xcomplex.hpp"
#include "../views/xaxis_slice_iterator.hpp"
#include "../views/xview.hpp"
#include "./xtl_concepts.hpp"

namespace xt
{
    namespace fft
    {
        namespace detail
        {
            template <xtl::complex_concept E>
            inline auto radix2(E&& e)
            {
                using namespace xt::placeholders;
                using namespace std::complex_literals;
                using value_type = typename std::decay_t<E>::value_type;
                using precision = typename value_type::value_type;
                auto N = e.size();
                const bool powerOfTwo = !(N == 0) && !(N & (N - 1));
                // check for power of 2
                if (!powerOfTwo || N == 0)
                {
                    // TODO: Replace implementation with dft
                    XTENSOR_THROW(std::runtime_error, "FFT Implementation requires power of 2");
                }
                auto pi = xt::numeric_constants<precision>::PI;
                xt::xtensor<value_type, 1> ev = e;
                if (N <= 1)
                {
                    return ev;
                }
                else
                {
#ifdef XTENSOR_USE_TBB
                    xt::xtensor<value_type, 1> even;
                    xt::xtensor<value_type, 1> odd;
                    oneapi::tbb::parallel_invoke(
                        [&]
                        {
                            even = radix2(xt::view(ev, xt::range(0, _, 2)));
                        },
                        [&]
                        {
                            odd = radix2(xt::view(ev, xt::range(1, _, 2)));
                        }
                    );
#else
                    auto even = radix2(xt::view(ev, xt::range(0, _, 2)));
                    auto odd = radix2(xt::view(ev, xt::range(1, _, 2)));
#endif

                    auto range = xt::arange<double>(N / 2);
                    auto exp = xt::exp(static_cast<value_type>(-2i) * pi * range / N);
                    auto t = exp * odd;
                    auto first_half = even + t;
                    auto second_half = even - t;
                    // TODO: should be a call to stack if performance was improved
                    auto spectrum = xt::xtensor<value_type, 1>::from_shape({N});
                    xt::view(spectrum, xt::range(0, N / 2)) = first_half;
                    xt::view(spectrum, xt::range(N / 2, N)) = second_half;
                    return spectrum;
                }
            }

            template <typename E>
            auto transform_bluestein(E&& data)
            {
                using value_type = typename std::decay_t<E>::value_type;
                using precision = typename value_type::value_type;

                // Find a power-of-2 convolution length m such that m >= n * 2 + 1
                const std::size_t n = data.size();
                size_t m = std::ceil(std::log2(n * 2 + 1));
                m = std::pow(2, m);

                // Trignometric table
                auto exp_table = xt::xtensor<std::complex<precision>, 1>::from_shape({n});
                xt::xtensor<std::size_t, 1> i = xt::pow(xt::linspace<std::size_t>(0, n - 1, n), 2);
                i %= (n * 2);

                auto angles = xt::eval(precision{3.141592653589793238463} * i / n);
                auto j = std::complex<precision>(0, 1);
                exp_table = xt::exp(-angles * j);

                // Temporary vectors and preprocessing
                auto av = xt::empty<std::complex<precision>>({m});
                xt::view(av, xt::range(0, n)) = data * exp_table;


                auto bv = xt::empty<std::complex<precision>>({m});
                xt::view(bv, xt::range(0, n)) = ::xt::conj(exp_table);
                xt::view(bv, xt::range(-n + 1, xt::placeholders::_)) = xt::view(
                    ::xt::conj(xt::flip(exp_table)),
                    xt::range(xt::placeholders::_, -1)
                );

                // Convolution
                auto xv = radix2(av);
                auto yv = radix2(bv);
                auto spectrum_k = xv * yv;
                auto complex_args = xt::conj(spectrum_k);
                auto fft_res = radix2(complex_args);
                auto cv = xt::conj(fft_res) / m;

                return xt::eval(xt::view(cv, xt::range(0, n)) * exp_table);
            }
        }  // namespace detail

        /**
         * @brief 1D FFT of an Nd array along a specified axis
         * @param e an Nd expression to be transformed to the fourier domain
         * @param axis the axis along which to perform the 1D FFT
         * @return a transformed xarray of the specified precision
         */
        template <class E>
        inline auto fft(E&& e, std::ptrdiff_t axis = -1)
        {
            using value_type = typename std::decay<E>::type::value_type;
            if constexpr (xtl::is_complex<typename std::decay<E>::type::value_type>::value)
            {
                using precision = typename value_type::value_type;
                const auto saxis = xt::normalize_axis(e.dimension(), axis);
                const size_t N = e.shape(saxis);
                const bool powerOfTwo = !(N == 0) && !(N & (N - 1));
                xt::xarray<std::complex<precision>> out = xt::eval(e);
                auto begin = xt::axis_slice_begin(out, saxis);
                auto end = xt::axis_slice_end(out, saxis);
                for (auto iter = begin; iter != end; iter++)
                {
                    if (powerOfTwo)
                    {
                        xt::noalias(*iter) = detail::radix2(*iter);
                    }
                    else
                    {
                        xt::noalias(*iter) = detail::transform_bluestein(*iter);
                    }
                }
                return out;
            }
            else
            {
                return fft(xt::cast<std::complex<value_type>>(e), axis);
            }
        }

        template <class E>
        inline auto ifft(E&& e, std::ptrdiff_t axis = -1)
        {
            if constexpr (xtl::is_complex<typename std::decay<E>::type::value_type>::value)
            {
                // check the length of the data on that axis
                const std::size_t n = e.shape(axis);
                if (n == 0)
                {
                    XTENSOR_THROW(std::runtime_error, "Cannot take the iFFT along an empty dimention");
                }
                auto complex_args = xt::conj(e);
                auto fft_res = xt::fft::fft(complex_args, axis);
                fft_res = xt::conj(fft_res);
                return fft_res;
            }
            else
            {
                using value_type = typename std::decay<E>::type::value_type;
                return ifft(xt::cast<std::complex<value_type>>(e), axis);
            }
        }

        /*
         * @brief performs a circular fft convolution xvec and yvec must
         *        be the same shape.
         * @param xvec first array of the convolution
         * @param yvec second array of the convolution
         * @param axis axis along which to perform the convolution
         */
        template <typename E1, typename E2>
        auto convolve(E1&& xvec, E2&& yvec, std::ptrdiff_t axis = -1)
        {
            // we could broadcast but that could get complicated???
            if (xvec.dimension() != yvec.dimension())
            {
                XTENSOR_THROW(std::runtime_error, "Mismatched dimentions");
            }

            auto saxis = xt::normalize_axis(xvec.dimension(), axis);
            if (xvec.shape(saxis) != yvec.shape(saxis))
            {
                XTENSOR_THROW(std::runtime_error, "Mismatched lengths along slice axis");
            }

            const std::size_t n = xvec.shape(saxis);

            auto xv = fft(xvec, axis);
            auto yv = fft(yvec, axis);

            auto begin_x = xt::axis_slice_begin(xv, saxis);
            auto end_x = xt::axis_slice_end(xv, saxis);
            auto iter_y = xt::axis_slice_begin(yv, saxis);

            for (auto iter = begin_x; iter != end_x; iter++)
            {
                (*iter) = (*iter_y++) * (*iter);
            }

            auto outvec = ifft(xv, axis);

            // Scaling (because this FFT implementation omits it)
            outvec = outvec / n;

            return outvec;
        }

    }
}  // namespace xt::fft
