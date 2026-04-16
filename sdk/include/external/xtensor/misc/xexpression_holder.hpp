/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_XEXPRESSION_HOLDER_HPP
#define XTENSOR_XEXPRESSION_HOLDER_HPP

#include <memory>

#include <nlohmann/json.hpp>

#include "../containers/xarray.hpp"
#include "../core/xtensor_config.hpp"
#include "../io/xjson.hpp"

namespace xt
{

    namespace detail
    {
        class xexpression_holder_impl;

        template <class CTE>
        class xexpression_wrapper;
    }

    class xexpression_holder  // Value semantic
    {
    public:

        using implementation_type = detail::xexpression_holder_impl;

        xexpression_holder() = default;

        template <class E>
        xexpression_holder(E&& expr);

        xexpression_holder(implementation_type* holder);
        xexpression_holder(const xexpression_holder& holder);
        xexpression_holder(xexpression_holder&& holder);

        xexpression_holder& operator=(const xexpression_holder&);
        xexpression_holder& operator=(xexpression_holder&&);

        void swap(xexpression_holder&);

        void to_json(nlohmann::json&) const;
        void from_json(const nlohmann::json&);

    private:

        void init_pointer_from_json(const nlohmann::json&);
        void check_holder() const;

        std::unique_ptr<implementation_type> p_holder;
    };

    /*************************************
     * to_json and from_json declaration *
     *************************************/

    /// @cond DOXYGEN_INCLUDE_SFINAE
    void to_json(nlohmann::json& j, const xexpression_holder& o);
    void from_json(const nlohmann::json& j, xexpression_holder& o);

    /// @endcond

    namespace detail
    {
        class xexpression_holder_impl  // Entity semantic
        {
        public:

            xexpression_holder_impl(xexpression_holder_impl&&) = delete;

            xexpression_holder_impl& operator=(const xexpression_holder_impl&) = delete;
            xexpression_holder_impl& operator=(xexpression_holder_impl&&) = delete;

            virtual xexpression_holder_impl* clone() const = 0;
            virtual void to_json(nlohmann::json&) const = 0;
            virtual void from_json(const nlohmann::json&) = 0;
            virtual ~xexpression_holder_impl() = default;

        protected:

            xexpression_holder_impl() = default;
            xexpression_holder_impl(const xexpression_holder_impl&) = default;
        };

        template <class CTE>
        class xexpression_wrapper : public xexpression_holder_impl
        {
        public:

            template <class E>
            xexpression_wrapper(E&& expr);

            xexpression_wrapper* clone() const;

            void to_json(nlohmann::json&) const;
            void from_json(const nlohmann::json&);

            ~xexpression_wrapper() = default;

        protected:

            xexpression_wrapper(const xexpression_wrapper&);

        private:

            CTE m_expression;
        };
    }

    template <class E>
    inline xexpression_holder::xexpression_holder(E&& expr)
        : p_holder(new detail::xexpression_wrapper<E>(std::forward<E>(expr)))
    {
    }

    inline xexpression_holder::xexpression_holder(implementation_type* holder)
        : p_holder(holder)
    {
    }

    inline xexpression_holder::xexpression_holder(const xexpression_holder& holder)
        : p_holder(holder.p_holder->clone())
    {
    }

    inline xexpression_holder::xexpression_holder(xexpression_holder&& holder)
        : p_holder(std::move(holder.p_holder))
    {
    }

    inline xexpression_holder& xexpression_holder::operator=(const xexpression_holder& holder)
    {
        xexpression_holder tmp(holder);
        swap(tmp);
        return *this;
    }

    inline xexpression_holder& xexpression_holder::operator=(xexpression_holder&& holder)
    {
        swap(holder);
        return *this;
    }

    inline void xexpression_holder::swap(xexpression_holder& holder)
    {
        std::swap(p_holder, holder.p_holder);
    }

    inline void xexpression_holder::to_json(nlohmann::json& j) const
    {
        if (p_holder == nullptr)
        {
            return;
        }
        p_holder->to_json(j);
    }

    inline void xexpression_holder::from_json(const nlohmann::json& j)
    {
        if (!j.is_array())
        {
            XTENSOR_THROW(std::runtime_error, "Received a JSON that does not contain a tensor");
        }

        if (p_holder == nullptr)
        {
            init_pointer_from_json(j);
        }
        p_holder->from_json(j);
    }

    inline void xexpression_holder::init_pointer_from_json(const nlohmann::json& j)
    {
        if (j.is_array())
        {
            return init_pointer_from_json(j[0]);
        }

        if (j.is_number())
        {
            xt::xarray<double> empty_arr;
            p_holder.reset(new detail::xexpression_wrapper<xt::xarray<double>>(std::move(empty_arr)));
        }

        if (j.is_boolean())
        {
            xt::xarray<bool> empty_arr;
            p_holder.reset(new detail::xexpression_wrapper<xt::xarray<bool>>(std::move(empty_arr)));
        }

        if (j.is_string())
        {
            xt::xarray<std::string> empty_arr;
            p_holder.reset(new detail::xexpression_wrapper<xt::xarray<std::string>>(std::move(empty_arr)));
        }

        XTENSOR_THROW(std::runtime_error, "Received a JSON with a tensor that contains unsupported data type");
    }

    inline void xexpression_holder::check_holder() const
    {
        if (p_holder == nullptr)
        {
            XTENSOR_THROW(std::runtime_error, "The holder does not contain an expression");
        }
    }

    /****************************************
     * to_json and from_json implementation *
     ****************************************/

    /// @cond DOXYGEN_INCLUDE_SFINAE
    inline void to_json(nlohmann::json& j, const xexpression_holder& o)
    {
        o.to_json(j);
    }

    inline void from_json(const nlohmann::json& j, xexpression_holder& o)
    {
        o.from_json(j);
    }

    /// @endcond

    namespace detail
    {
        template <class CTE>
        template <class E>
        inline xexpression_wrapper<CTE>::xexpression_wrapper(E&& expr)
            : xexpression_holder_impl()
            , m_expression(std::forward<E>(expr))
        {
        }

        template <class CTE>
        inline xexpression_wrapper<CTE>* xexpression_wrapper<CTE>::clone() const
        {
            return new xexpression_wrapper<CTE>(*this);
        }

        template <class CTE>
        inline void xexpression_wrapper<CTE>::to_json(nlohmann::json& j) const
        {
            ::xt::to_json(j, m_expression);
        }

        template <class CTE>
        inline void xexpression_wrapper<CTE>::from_json(const nlohmann::json& j)
        {
            ::xt::from_json(j, m_expression);
        }

        template <class CTE>
        inline xexpression_wrapper<CTE>::xexpression_wrapper(const xexpression_wrapper& wrapper)
            : xexpression_holder_impl()
            , m_expression(wrapper.m_expression)
        {
        }
    }
}

#endif
