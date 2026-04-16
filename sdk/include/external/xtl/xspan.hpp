/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTL_XSPAN_HPP
#define XTL_XSPAN_HPP

#include "xspan_impl.hpp"

namespace xtl
{
	using tcb::span;
	constexpr std::ptrdiff_t dynamic_extent = tcb::dynamic_extent;
}

#endif
