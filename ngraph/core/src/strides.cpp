// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/strides.hpp"

#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

std::ostream& ngraph::operator<<(std::ostream& s, const Strides& strides) {
    s << "Strides{";
    s << ngraph::join(strides);
    s << "}";
    return s;
}

ngraph::Strides::Strides() : std::vector<size_t>() {}

ngraph::Strides::Strides(const std::initializer_list<size_t>& axis_strides) : std::vector<size_t>(axis_strides) {}

ngraph::Strides::Strides(const std::vector<size_t>& axis_strides) : std::vector<size_t>(axis_strides) {}

ngraph::Strides::Strides(const Strides& axis_strides) : std::vector<size_t>(axis_strides) {}

ngraph::Strides::Strides(size_t n, size_t initial_value) : std::vector<size_t>(n, initial_value) {}

ngraph::Strides& ngraph::Strides::operator=(const Strides& v) {
    static_cast<std::vector<size_t>*>(this)->operator=(v);
    return *this;
}

ngraph::Strides& ngraph::Strides::operator=(Strides&& v) noexcept {
    static_cast<std::vector<size_t>*>(this)->operator=(v);
    return *this;
}

constexpr DiscreteTypeInfo ov::AttributeAdapter<Strides>::type_info;
