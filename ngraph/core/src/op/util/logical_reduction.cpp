// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/logical_reduction.hpp"

#include "itt.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::util::LogicalReduction, "LogicalReduction", 1);

op::util::LogicalReduction::LogicalReduction() {}

op::util::LogicalReduction::LogicalReduction(const Output<Node>& arg, const AxisSet& reduction_axes)
    : ReductionBase(
          arg,
          op::Constant::create(element::i64, Shape{reduction_axes.size()}, reduction_axes.to_vector())->output(0)) {
    add_provenance_group_member(input_value(1).get_node_shared_ptr());
}

op::util::LogicalReduction::LogicalReduction(const Output<Node>& arg, const Output<Node>& reduction_axes)
    : ReductionBase(arg, reduction_axes) {}

bool op::util::LogicalReduction::reduction_axes_constant() const {
    return has_and_set_equal_bounds(input_value(1));
}

const AxisSet op::util::LogicalReduction::get_reduction_axes() const {
    AxisSet axes;
    if (auto const_op = get_constant_from_source(input_value(1))) {
        axes = const_op->get_axis_set_val();
    }
    return axes;
}

void op::util::LogicalReduction::set_reduction_axes(const AxisSet& reduction_axes) {
    this->input(1).replace_source_output(
        op::Constant::create(element::i64, Shape{reduction_axes.size()}, reduction_axes.to_vector())->output(0));
}

void op::util::LogicalReduction::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(util_LogicalReduction_validate_and_infer_types);

    const element::Type& data_et = get_input_element_type(0);
    const PartialShape& axes_shape = get_input_partial_shape(1);

    NODE_VALIDATION_CHECK(this, data_et.compatible(element::boolean), "Element type of data input must be boolean.");

    const Rank axes_rank = axes_shape.rank();
    NODE_VALIDATION_CHECK(this,
                          axes_rank.compatible(0) || axes_rank.compatible(1),
                          "Axes input must be a scalar or 1D input. Got: ",
                          axes_shape);

    PartialShape result_shape = infer_reduction_output_shape(false);
    set_input_is_relevant_to_shape(1);
    set_output_type(0, data_et, result_shape);
}
