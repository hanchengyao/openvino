// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "cldnn/primitives/scatter_elements_update.hpp"
#include "primitive_inst.h"
#include <string>

namespace cldnn {
template <>
struct typed_program_node<scatter_elements_update> : public typed_program_node_base<scatter_elements_update> {
    using parent = typed_program_node_base<scatter_elements_update>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
};

using scatter_elements_update_node = typed_program_node<scatter_elements_update>;

template <>
class typed_primitive_inst<scatter_elements_update> : public typed_primitive_inst_base<scatter_elements_update> {
    using parent = typed_primitive_inst_base<scatter_elements_update>;

public:
    static layout calc_output_layout(scatter_elements_update_node const& node);
    static std::string to_string(scatter_elements_update_node const& node);

public:
    typed_primitive_inst(network& network, scatter_elements_update_node const& desc);
};

using scatter_elements_update_inst = typed_primitive_inst<scatter_elements_update>;
}  // namespace cldnn
