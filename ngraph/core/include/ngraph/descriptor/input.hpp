// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>

#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/variant.hpp"
#include "openvino/core/descriptor/input.hpp"

namespace ngraph {
class Node;

namespace descriptor {

// Describes a tensor that is an input to an op, directly or indirectly via a tuple
using ov::descriptor::Input;
}  // namespace descriptor
}  // namespace ngraph
