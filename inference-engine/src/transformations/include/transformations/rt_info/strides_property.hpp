// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/strides.hpp>
#include <ngraph/node_input.hpp>
#include <ngraph/variant.hpp>
#include <transformations_visibility.hpp>


namespace ov {
template <>
class TRANSFORMATIONS_API VariantWrapper<ngraph::Strides> : public VariantImpl<ngraph::Strides> {
public:
    static constexpr VariantTypeInfo type_info{"Variant::Strides", 0};
    const VariantTypeInfo& get_type_info() const override { return type_info; }
    VariantWrapper(const value_type& value)
        : VariantImpl<value_type>(value) {
    }
};

} // namespace ov

TRANSFORMATIONS_API bool has_strides_prop(const ngraph::Input<ngraph::Node>& node);
TRANSFORMATIONS_API ngraph::Strides get_strides_prop(const ngraph::Input<ngraph::Node>& node);
TRANSFORMATIONS_API void insert_strides_prop(ngraph::Input<ngraph::Node>& node, const ngraph::Strides& strides);
