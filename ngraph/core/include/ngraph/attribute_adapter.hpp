// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <type_traits>
#include <vector>

#include "ngraph/enum_names.hpp"
#include "ngraph/type.hpp"
#include "openvino/core/attribute_adapter.hpp"

///
namespace ngraph {

using ov::ValueAccessor;

using ov::DirectValueAccessor;

using ov::IndirectScalarValueAccessor;

template <typename A, typename B>
A copy_from(B& b) {
    return ov::copy_from<A>(b);
}

using ov::IndirectVectorValueAccessor;

using ov::AttributeAdapter;
using ov::EnumAttributeAdapterBase;

using ov::VisitorAdapter;

}  // namespace ngraph
