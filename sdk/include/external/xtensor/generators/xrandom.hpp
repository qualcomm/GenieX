/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

/**
 * @brief functions to obtain xgenerators generating random numbers with given shape
 */

#ifndef XTENSOR_RANDOM_HPP
#define XTENSOR_RANDOM_HPP

#include <algorithm>
#include <functional>
#include <random>
#include <type_traits>
#include <utility>

#include <xtl/xspan.hpp>

#include "../containers/xtensor.hpp"
#include "../core/xmath.hpp"
#include "../core/xtensor_config.hpp"
#include "../generators/xbuilder.hpp"
#include "../generators/xgenerator.hpp"
#include "../misc/xtl_concepts.hpp"
#include "../views/xindex_view.hpp"
#include "../views/xview.hpp"

namespace xt
{
    /*********************
     * Random generators *
     *********************/

    namespace random
    {
        using default_engine_type = std::mt19937;
        using seed_type = default_engine_type::result_type;

        default_engine_type& get_default_random_engine();
        void seed(seed_type seed);

        template <class T, class S, class E = random::default_engine_type>
        auto rand(const S& shape, T lower = 0, T upper = 1, E& engine = random::get_default_random_engine());

        template <class T, class S, class E = random::default_engine_type>
        auto randint(
            const S& shape,
            T lower = 0,
            T upper = (std::numeric_limits<T>::max)(),
            E& engine = random::get_default_random_engine()
        );

        template <class T, class S, class E = random::default_engine_type>
        auto randn(const S& shape, T mean = 0, T std_dev = 1, E& engine = random::get_default_random_engine());

        template <class T, class S, class D = double, class E = random::default_engine_type>
        auto
        binomial(const S& shape, T trials = 1, D prob = 0.5, E& engine = random::get_default_random_engine());

        template <class T, class S, class D = double, class E = random::default_engine_type>
        auto geometric(const S& shape, D prob = 0.5, E& engine = random::get_default_random_engine());

        template <class T, class S, class D = double, class E = random::default_engine_type>
        auto
        negative_binomial(const S& shape, T k = 1, D prob = 0.5, E& engine = random::get_default_random_engine());

        template <class T, class S, class D = double, class E = random::default_engine_type>
        auto poisson(const S& shape, D rate = 1.0, E& engine = random::get_default_random_engine());

        template <class T, class S, class E = random::default_engine_type>
        auto exponential(const S& shape, T rate = 1.0, E& engine = random::get_default_random_engine());

        template <class T, class S, class E = random::default_engine_type>
        auto
        gamma(const S& shape, T alpha = 1.0, T beta = 1.0, E& engine = random::get_default_random_engine());

        template <class T, class S, class E = random::default_engine_type>
        auto weibull(const S& shape, T a = 1.0, T b = 1.0, E& engine = random::get_default_random_engine());

        template <class T, class S, class E = random::default_engine_type>
        auto
        extreme_value(const S& shape, T a = 0.0, T b = 1.0, E& engine = random::get_default_random_engine());

        template <class T, class S, class E = random::default_engine_type>
        auto
        lognormal(const S& shape, T mean = 0, T std_dev = 1, E& engine = random::get_default_random_engine());

        template <class T, class S, class E = random::default_engine_type>
        auto chi_squared(const S& shape, T deg = 1.0, E& engine = random::get_default_random_engine());

        template <class T, class S, class E = random::default_engine_type>
        auto cauchy(const S& shape, T a = 0.0, T b = 1.0, E& engine = random::get_default_random_engine());

        template <class T, class S, class E = random::default_engine_type>
        auto fisher_f(const S& shape, T m = 1.0, T n = 1.0, E& engine = random::get_default_random_engine());

        template <class T, class S, class E = random::default_engine_type>
        auto student_t(const S& shape, T n = 1.0, E& engine = random::get_default_random_engine());

        template <class T, class I, std::size_t L, class E = random::default_engine_type>
        auto
        rand(const I (&shape)[L], T lower = 0, T upper = 1, E& engine = random::get_default_random_engine());

        template <class T, class I, std::size_t L, class E = random::default_engine_type>
        auto randint(
            const I (&shape)[L],
            T lower = 0,
            T upper = (std::numeric_limits<T>::max)(),
            E& engine = random::get_default_random_engine()
        );

        template <class T, class I, std::size_t L, class E = random::default_engine_type>
        auto
        randn(const I (&shape)[L], T mean = 0, T std_dev = 1, E& engine = random::get_default_random_engine());

        template <class T, class I, std::size_t L, class D = double, class E = random::default_engine_type>
        auto
        binomial(const I (&shape)[L], T trials = 1, D prob = 0.5, E& engine = random::get_default_random_engine());

        template <class T, class I, std::size_t L, class D = double, class E = random::default_engine_type>
        auto geometric(const I (&shape)[L], D prob = 0.5, E& engine = random::get_default_random_engine());

        template <class T, class I, std::size_t L, class D = double, class E = random::default_engine_type>
        auto negative_binomial(
            const I (&shape)[L],
            T k = 1,
            D prob = 0.5,
            E& engine = random::get_default_random_engine()
        );

        template <class T, class I, std::size_t L, class D = double, class E = random::default_engine_type>
        auto poisson(const I (&shape)[L], D rate = 1.0, E& engine = random::get_default_random_engine());

        template <class T, class I, std::size_t L, class E = random::default_engine_type>
        auto exponential(const I (&shape)[L], T rate = 1.0, E& engine = random::get_default_random_engine());

        template <class T, class I, std::size_t L, class E = random::default_engine_type>
        auto
        gamma(const I (&shape)[L], T alpha = 1.0, T beta = 1.0, E& engine = random::get_default_random_engine());

        template <class T, class I, std::size_t L, class E = random::default_engine_type>
        auto
        weibull(const I (&shape)[L], T a = 1.0, T b = 1.0, E& engine = random::get_default_random_engine());

        template <class T, class I, std::size_t L, class E = random::default_engine_type>
        auto
        extreme_value(const I (&shape)[L], T a = 0.0, T b = 1.0, E& engine = random::get_default_random_engine());

        template <class T, class I, std::size_t L, class E = random::default_engine_type>
        auto lognormal(
            const I (&shape)[L],
            T mean = 0.0,
            T std_dev = 1.0,
            E& engine = random::get_default_random_engine()
        );

        template <class T, class I, std::size_t L, class E = random::default_engine_type>
        auto chi_squared(const I (&shape)[L], T deg = 1.0, E& engine = random::get_default_random_engine());

        template <class T, class I, std::size_t L, class E = random::default_engine_type>
        auto cauchy(const I (&shape)[L], T a = 0.0, T b = 1.0, E& engine = random::get_default_random_engine());

        template <class T, class I, std::size_t L, class E = random::default_engine_type>
        auto
        fisher_f(const I (&shape)[L], T m = 1.0, T n = 1.0, E& engine = random::get_default_random_engine());

        template <class T, class I, std::size_t L, class E = random::default_engine_type>
        auto student_t(const I (&shape)[L], T n = 1.0, E& engine = random::get_default_random_engine());

        template <class T, class E = random::default_engine_type>
        void shuffle(xexpression<T>& e, E& engine = random::get_default_random_engine());

        template <xtl::integral_concept T, class E = random::default_engine_type>
        xtensor<T, 1> permutation(T e, E& engine = random::get_default_random_engine());

        template <xexpression_concept T, class E = random::default_engine_type>
        std::decay_t<T> permutation(T&& e, E& engine = random::get_default_random_engine());

        template <class T, class E = random::default_engine_type>
        xtensor<typename T::value_type, 1> choice(
            const xexpression<T>& e,
            std::size_t n,
            bool replace = true,
            E& engine = random::get_default_random_engine()
        );

        template <class T, class W, class E = random::default_engine_type>
        xtensor<typename T::value_type, 1> choice(
            const xexpression<T>& e,
            std::size_t n,
            const xexpression<W>& weights,
            bool replace = true,
            E& engine = random::get_default_random_engine()
        );
    }

    namespace detail
    {
        template <class T, class E, class D>
        struct random_impl
        {
            using value_type = T;

            random_impl(E& engine, D&& dist)
                : m_engine(engine)
                , m_dist(std::move(dist))
            {
            }

            template <class... Args>
            inline value_type operator()(Args...) const
            {
                return m_dist(m_engine);
            }

            template <class It>
            inline value_type element(It, It) const
            {
                return m_dist(m_engine);
            }

            template <class EX>
            inline void assign_to(xexpression<EX>& e) const noexcept
            {
                // Note: we're not going row/col major here
                auto& ed = e.derived_cast();
                for (auto&& el : ed.storage())
                {
                    el = m_dist(m_engine);
                }
            }

        private:

            E& m_engine;
            mutable D m_dist;
        };
    }

    namespace random
    {
        /**
         * Returns a reference to the default random number engine
         */
        inline default_engine_type& get_default_random_engine()
        {
            static default_engine_type mt;
            return mt;
        }

        /**
         * Seeds the default random number generator with @p seed
         * @param seed The seed
         */
        inline void seed(seed_type seed)
        {
            get_default_random_engine().seed(seed);
        }

        /**
         * xexpression with specified @p shape containing uniformly distributed random numbers
         * in the interval from @p lower to @p upper, excluding upper.
         *
         * Numbers are drawn from @c std::uniform_real_distribution.
         *
         * @param shape shape of resulting xexpression
         * @param lower lower bound
         * @param upper upper bound
         * @param engine random number engine
         * @tparam T number type to use
         */
        template <class T, class S, class E>
        inline auto rand(const S& shape, T lower, T upper, E& engine)
        {
            std::uniform_real_distribution<T> dist(lower, upper);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        /**
         * xexpression with specified @p shape containing uniformly distributed
         * random integers in the interval from @p lower to @p upper, excluding upper.
         *
         * Numbers are drawn from @c std::uniform_int_distribution.
         *
         * @param shape shape of resulting xexpression
         * @param lower lower bound
         * @param upper upper bound
         * @param engine random number engine
         * @tparam T number type to use
         */
        template <class T, class S, class E>
        inline auto randint(const S& shape, T lower, T upper, E& engine)
        {
            std::uniform_int_distribution<T> dist(lower, T(upper - 1));
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        /**
         * xexpression with specified @p shape containing numbers sampled from
         * the Normal (Gaussian) random number distribution with mean @p mean and
         * standard deviation @p std_dev.
         *
         * Numbers are drawn from @c std::normal_distribution.
         *
         * @param shape shape of resulting xexpression
         * @param mean mean of normal distribution
         * @param std_dev standard deviation of normal distribution
         * @param engine random number engine
         * @tparam T number type to use
         */
        template <class T, class S, class E>
        inline auto randn(const S& shape, T mean, T std_dev, E& engine)
        {
            std::normal_distribution<T> dist(mean, std_dev);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        /**
         * xexpression with specified @p shape containing numbers sampled from
         * the binomial random number distribution for @p trials trials with
         * probability of success equal to @p prob.
         *
         * Numbers are drawn from @c std::binomial_distribution.
         *
         * @param shape shape of resulting xexpression
         * @param trials number of Bernoulli trials
         * @param prob probability of success of each trial
         * @param engine random number engine
         * @tparam T number type to use
         */
        template <class T, class S, class D, class E>
        inline auto binomial(const S& shape, T trials, D prob, E& engine)
        {
            std::binomial_distribution<T> dist(trials, prob);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        /**
         * xexpression with specified @p shape containing numbers sampled from
         * a gemoetric random number distribution with
         * probability of success equal to @p prob for each of the Bernoulli trials.
         *
         * Numbers are drawn from @c std::geometric_distribution.
         *
         * @param shape shape of resulting xexpression
         * @param prob probability of success of each trial
         * @param engine random number engine
         * @tparam T number type to use
         */
        template <class T, class S, class D, class E>
        inline auto geometric(const S& shape, D prob, E& engine)
        {
            std::geometric_distribution<T> dist(prob);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        /**
         * xexpression with specified @p shape containing numbers sampled from
         * a negative binomial random number distribution (also known as Pascal distribution)
         * that returns the number of successes before @p k trials with probability of success
         * equal to @p prob for each of the Bernoulli trials.
         *
         * Numbers are drawn from @c std::negative_binomial_distribution.
         *
         * @param shape shape of resulting xexpression
         * @param k number of unsuccessful trials
         * @param prob probability of success of each trial
         * @param engine random number engine
         * @tparam T number type to use
         */
        template <class T, class S, class D, class E>
        inline auto negative_binomial(const S& shape, T k, D prob, E& engine)
        {
            std::negative_binomial_distribution<T> dist(k, prob);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        /**
         * xexpression with specified @p shape containing numbers sampled from
         * a Poisson random number distribution with rate @p rate
         *
         * Numbers are drawn from @c std::poisson_distribution.
         *
         * @param shape shape of resulting xexpression
         * @param rate rate of Poisson distribution
         * @param engine random number engine
         * @tparam T number type to use
         */
        template <class T, class S, class D, class E>
        inline auto poisson(const S& shape, D rate, E& engine)
        {
            std::poisson_distribution<T> dist(rate);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        /**
         * xexpression with specified @p shape containing numbers sampled from
         * a exponential random number distribution with rate @p rate
         *
         * Numbers are drawn from @c std::exponential_distribution.
         *
         * @param shape shape of resulting xexpression
         * @param rate rate of exponential distribution
         * @param engine random number engine
         * @tparam T number type to use
         */
        template <class T, class S, class E>
        inline auto exponential(const S& shape, T rate, E& engine)
        {
            std::exponential_distribution<T> dist(rate);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        /**
         * xexpression with specified @p shape containing numbers sampled from
         * a gamma random number distribution with shape @p alpha and scale @p beta
         *
         * Numbers are drawn from @c std::gamma_distribution.
         *
         * @param shape shape of resulting xexpression
         * @param alpha shape of the gamma distribution
         * @param beta scale of the gamma distribution
         * @param engine random number engine
         * @tparam T number type to use
         */
        template <class T, class S, class E>
        inline auto gamma(const S& shape, T alpha, T beta, E& engine)
        {
            std::gamma_distribution<T> dist(alpha, beta);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        /**
         * xexpression with specified @p shape containing numbers sampled from
         * a Weibull random number distribution with shape @p a and scale @p b
         *
         * Numbers are drawn from @c std::weibull_distribution.
         *
         * @param shape shape of resulting xexpression
         * @param a shape of the weibull distribution
         * @param b scale of the weibull distribution
         * @param engine random number engine
         * @tparam T number type to use
         */
        template <class T, class S, class E>
        inline auto weibull(const S& shape, T a, T b, E& engine)
        {
            std::weibull_distribution<T> dist(a, b);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        /**
         * xexpression with specified @p shape containing numbers sampled from
         * a extreme value random number distribution with shape @p a and scale @p b
         *
         * Numbers are drawn from @c std::extreme_value_distribution.
         *
         * @param shape shape of resulting xexpression
         * @param a shape of the extreme value distribution
         * @param b scale of the extreme value distribution
         * @param engine random number engine
         * @tparam T number type to use
         */
        template <class T, class S, class E>
        inline auto extreme_value(const S& shape, T a, T b, E& engine)
        {
            std::extreme_value_distribution<T> dist(a, b);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        /**
         * xexpression with specified @p shape containing numbers sampled from
         * the Log-Normal random number distribution with mean @p mean and
         * standard deviation @p std_dev.
         *
         * Numbers are drawn from @c std::lognormal_distribution.
         *
         * @param shape shape of resulting xexpression
         * @param mean mean of normal distribution
         * @param std_dev standard deviation of normal distribution
         * @param engine random number engine
         * @tparam T number type to use
         */
        template <class T, class S, class E>
        inline auto lognormal(const S& shape, T mean, T std_dev, E& engine)
        {
            std::lognormal_distribution<T> dist(mean, std_dev);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        /**
         * xexpression with specified @p shape containing numbers sampled from
         * the chi-squared random number distribution with @p deg degrees of freedom.
         *
         * Numbers are drawn from @c std::chi_squared_distribution.
         *
         * @param shape shape of resulting xexpression
         * @param deg degrees of freedom
         * @param engine random number engine
         * @tparam T number type to use
         */
        template <class T, class S, class E>
        inline auto chi_squared(const S& shape, T deg, E& engine)
        {
            std::chi_squared_distribution<T> dist(deg);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        /**
         * xexpression with specified @p shape containing numbers sampled from
         * a Cauchy random number distribution with peak @p a and scale @p b
         *
         * Numbers are drawn from @c std::cauchy_distribution.
         *
         * @param shape shape of resulting xexpression
         * @param a peak of the Cauchy distribution
         * @param b scale of the Cauchy distribution
         * @param engine random number engine
         * @tparam T number type to use
         */
        template <class T, class S, class E>
        inline auto cauchy(const S& shape, T a, T b, E& engine)
        {
            std::cauchy_distribution<T> dist(a, b);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        /**
         * xexpression with specified @p shape containing numbers sampled from
         * a Fisher-f random number distribution with numerator degrees of
         * freedom equal to @p m and denominator degrees of freedom equal to @p n
         *
         * Numbers are drawn from @c std::fisher_f_distribution.
         *
         * @param shape shape of resulting xexpression
         * @param m numerator degrees of freedom
         * @param n denominator degrees of freedom
         * @param engine random number engine
         * @tparam T number type to use
         */
        template <class T, class S, class E>
        inline auto fisher_f(const S& shape, T m, T n, E& engine)
        {
            std::fisher_f_distribution<T> dist(m, n);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        /**
         * xexpression with specified @p shape containing numbers sampled from
         * a Student-t random number distribution with degrees of
         * freedom equal to @p n
         *
         * Numbers are drawn from @c std::student_t_distribution.
         *
         * @param shape shape of resulting xexpression
         * @param n degrees of freedom
         * @param engine random number engine
         * @tparam T number type to use
         */
        template <class T, class S, class E>
        inline auto student_t(const S& shape, T n, E& engine)
        {
            std::student_t_distribution<T> dist(n);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        template <class T, class I, std::size_t L, class E>
        inline auto rand(const I (&shape)[L], T lower, T upper, E& engine)
        {
            std::uniform_real_distribution<T> dist(lower, upper);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        template <class T, class I, std::size_t L, class E>
        inline auto randint(const I (&shape)[L], T lower, T upper, E& engine)
        {
            std::uniform_int_distribution<T> dist(lower, T(upper - 1));
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        template <class T, class I, std::size_t L, class E>
        inline auto randn(const I (&shape)[L], T mean, T std_dev, E& engine)
        {
            std::normal_distribution<T> dist(mean, std_dev);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        template <class T, class I, std::size_t L, class D, class E>
        inline auto binomial(const I (&shape)[L], T trials, D prob, E& engine)
        {
            std::binomial_distribution<T> dist(trials, prob);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        template <class T, class I, std::size_t L, class D, class E>
        inline auto geometric(const I (&shape)[L], D prob, E& engine)
        {
            std::geometric_distribution<T> dist(prob);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        template <class T, class I, std::size_t L, class D, class E>
        inline auto negative_binomial(const I (&shape)[L], T k, D prob, E& engine)
        {
            std::negative_binomial_distribution<T> dist(k, prob);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        template <class T, class I, std::size_t L, class D, class E>
        inline auto poisson(const I (&shape)[L], D rate, E& engine)
        {
            std::poisson_distribution<T> dist(rate);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        template <class T, class I, std::size_t L, class E>
        inline auto exponential(const I (&shape)[L], T rate, E& engine)
        {
            std::exponential_distribution<T> dist(rate);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        template <class T, class I, std::size_t L, class E>
        inline auto gamma(const I (&shape)[L], T alpha, T beta, E& engine)
        {
            std::gamma_distribution<T> dist(alpha, beta);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        template <class T, class I, std::size_t L, class E>
        inline auto weibull(const I (&shape)[L], T a, T b, E& engine)
        {
            std::weibull_distribution<T> dist(a, b);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        template <class T, class I, std::size_t L, class E>
        inline auto extreme_value(const I (&shape)[L], T a, T b, E& engine)
        {
            std::extreme_value_distribution<T> dist(a, b);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        template <class T, class I, std::size_t L, class E>
        inline auto lognormal(const I (&shape)[L], T mean, T std_dev, E& engine)
        {
            std::lognormal_distribution<T> dist(mean, std_dev);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        template <class T, class I, std::size_t L, class E>
        inline auto chi_squared(const I (&shape)[L], T deg, E& engine)
        {
            std::chi_squared_distribution<T> dist(deg);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        template <class T, class I, std::size_t L, class E>
        inline auto cauchy(const I (&shape)[L], T a, T b, E& engine)
        {
            std::cauchy_distribution<T> dist(a, b);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        template <class T, class I, std::size_t L, class E>
        inline auto fisher_f(const I (&shape)[L], T m, T n, E& engine)
        {
            std::fisher_f_distribution<T> dist(m, n);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        template <class T, class I, std::size_t L, class E>
        inline auto student_t(const I (&shape)[L], T n, E& engine)
        {
            std::student_t_distribution<T> dist(n);
            return detail::make_xgenerator(
                detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)),
                shape
            );
        }

        /**
         * Randomly shuffle elements inplace in xcontainer along first axis.
         * The order of sub-arrays is changed but their contents remain the same.
         *
         * @param e xcontainer to shuffle inplace
         * @param engine random number engine
         */
        template <class T, class E>
        void shuffle(xexpression<T>& e, E& engine)
        {
            T& de = e.derived_cast();

            if (de.dimension() == 1)
            {
                using size_type = typename T::size_type;
                auto first = de.begin();
                auto last = de.end();

                for (size_type i = std::size_t((last - first) - 1); i > 0; --i)
                {
                    std::uniform_int_distribution<size_type> dist(0, i);
                    auto j = dist(engine);
                    using std::swap;
                    swap(first[i], first[j]);
                }
            }
            else
            {
                using size_type = typename T::size_type;
                decltype(auto) buf = empty_like(view(de, 0));

                for (size_type i = de.shape()[0] - 1; i > 0; --i)
                {
                    std::uniform_int_distribution<size_type> dist(0, i);
                    size_type j = dist(engine);

                    buf = view(de, j);
                    view(de, j) = view(de, i);
                    view(de, i) = buf;
                }
            }
        }

        /**
         * Randomly permute a sequence, or return a permuted range.
         *
         * If the first parameter is an integer, this function creates a new
         * ``arange(e)`` and returns it randomly permuted. Otherwise, this
         * function creates a copy of the input, passes it to @sa shuffle and
         * returns the result.
         *
         * @param e input xexpression or integer
         * @param engine random number engine to use (optional)
         *
         * @return randomly permuted copy of container or arange.
         */
        template <xtl::integral_concept T, class E>
        xtensor<T, 1> permutation(T e, E& engine)
        {
            xt::xtensor<T, 1> res = xt::arange<T>(e);
            shuffle(res, engine);
            return res;
        }

        /// @cond DOXYGEN_INCLUDE_SFINAE
        template <xexpression_concept T, class E>
        std::decay_t<T> permutation(T&& e, E& engine)
        {
            using copy_type = std::decay_t<T>;
            copy_type res = e;
            shuffle(res, engine);
            return res;
        }

        /// @endcond

        /**
         * Randomly select n unique elements from xexpression e.
         * Note: this function makes a copy of your data, and only 1D data is accepted.
         *
         * @param e expression to sample from
         * @param n number of elements to sample
         * @param replace whether to sample with or without replacement
         * @param engine random number engine
         *
         * @return xtensor containing 1D container of sampled elements
         */
        template <class T, class E>
        xtensor<typename T::value_type, 1>
        choice(const xexpression<T>& e, std::size_t n, bool replace, E& engine)
        {
            const auto& de = e.derived_cast();
            XTENSOR_ASSERT((de.dimension() == 1));
            XTENSOR_ASSERT((replace || n <= de.size()));
            using result_type = xtensor<typename T::value_type, 1>;
            using size_type = typename result_type::size_type;
            result_type result;
            result.resize({n});

            if (replace)
            {
                auto dist = std::uniform_int_distribution<size_type>(0, de.size() - 1);
                for (size_type i = 0; i < n; ++i)
                {
                    result[i] = de.storage()[dist(engine)];
                }
            }
            else
            {
                // Naive resevoir sampling without weighting:
                std::copy(de.storage().begin(), de.storage().begin() + n, result.begin());
                size_type i = n;
                for (auto it = de.storage().begin() + n; it != de.storage().end(); ++it, ++i)
                {
                    auto idx = std::uniform_int_distribution<size_type>(0, i)(engine);
                    if (idx < n)
                    {
                        result.storage()[idx] = *it;
                    }
                }
            }
            return result;
        }

        /**
         * Weighted random sampling.
         *
         * Randomly sample n unique elements from xexpression ``e`` using the discrete distribution
         * parametrized by the weights ``w``. When sampling with replacement, this means that the probability
         * to sample element ``e[i]`` is defined as
         * ``w[i] / sum(w)``.
         * Without replacement, this only describes the probability of the first sample element.
         * In successive samples, the weight of items already sampled is assumed to be zero.
         *
         * For weighted random sampling with replacement, binary search with cumulative weights alogrithm is
         * used. For weighted random sampling without replacement, the algorithm used is the exponential sort
         * from [Efraimidis and Spirakis](https://doi.org/10.1016/j.ipl.2005.11.003) (2006) with the ``weight
         * / randexp(1)`` [trick](https://web.archive.org/web/20201021162211/https://krlmlr.github.io/wrswoR/)
         * from Kirill Müller.
         *
         * Note: this function makes a copy of your data, and only 1D data is accepted.
         *
         * @param e expression to sample from
         * @param n number of elements to sample
         * @param w expression for the weight distribution.
         *          Weights must be positive and real-valued but need not sum to 1.
         * @param replace set true to sample with replacement
         * @param engine random number engine
         *
         * @return xtensor containing 1D container of sampled elements
         */
        template <class T, class W, class E>
        xtensor<typename T::value_type, 1>
        choice(const xexpression<T>& e, std::size_t n, const xexpression<W>& weights, bool replace, E& engine)
        {
            const auto& de = e.derived_cast();
            const auto& dweights = weights.derived_cast();
            XTENSOR_ASSERT((de.dimension() == 1));
            XTENSOR_ASSERT((replace || n <= de.size()));
            XTENSOR_ASSERT((de.size() == dweights.size()));
            XTENSOR_ASSERT((de.dimension() == dweights.dimension()));
            XTENSOR_ASSERT(xt::all(dweights >= 0));
            static_assert(
                std::is_floating_point<typename W::value_type>::value,
                "Weight expression must be of floating point type"
            );
            using result_type = xtensor<typename T::value_type, 1>;
            using size_type = typename result_type::size_type;
            using weight_type = typename W::value_type;
            result_type result;
            result.resize({n});

            if (replace)
            {
                // Sample u uniformly in the range [0, sum(weights)[
                // The index idx of the sampled element is such that weight_cumul[idx - 1] <= u <
                // weight_cumul[idx]. Where weight_cumul[-1] is implicitly 0, as the empty sum.
                const auto wc = eval(cumsum(dweights));
                std::uniform_real_distribution<weight_type> weight_dist{0, wc[wc.size() - 1]};
                for (auto& x : result)
                {
                    const auto u = weight_dist(engine);
                    const auto idx = static_cast<size_type>(
                        std::upper_bound(wc.cbegin(), wc.cend(), u) - wc.cbegin()
                    );
                    x = de[idx];
                }
            }
            else
            {
                // Compute (modified) keys as weight/randexp(1).
                xtensor<weight_type, 1> keys;
                keys.resize({dweights.size()});
                std::exponential_distribution<weight_type> randexp{weight_type(1)};
                std::transform(
                    dweights.cbegin(),
                    dweights.cend(),
                    keys.begin(),
                    [&randexp, &engine](auto w)
                    {
                        return w / randexp(engine);
                    }
                );

                // Find indexes for the n biggest key
                xtensor<size_type, 1> indices = arange<size_type>(0, dweights.size());
                std::partial_sort(
                    indices.begin(),
                    indices.begin() + n,
                    indices.end(),
                    [&keys](auto i, auto j)
                    {
                        return keys[i] > keys[j];
                    }
                );

                // Return samples with the n biggest keys
                result = index_view(de, xtl::span<size_type>{indices.data(), n});
            }
            return result;
        }

    }
}

#endif
