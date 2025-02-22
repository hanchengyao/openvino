// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/sink.hpp"
#include "ngraph/op/util/variable.hpp"
#include "ngraph/op/util/variable_extension.hpp"

namespace ngraph {
namespace op {
class NGRAPH_API AssignBase : public Sink, public VariableExtension {
public:
    NGRAPH_RTTI_DECLARATION;
    AssignBase() = default;
    /// \brief Constructs an AssignBase operation.
    explicit AssignBase(const OutputVector& arguments) : Sink(arguments) {}
};

namespace v3 {
/// \brief Assign operation sets an input value to the variable with `variable_id`
class NGRAPH_API Assign : public AssignBase {
public:
    NGRAPH_RTTI_DECLARATION;
    Assign() = default;

    /// \brief Constructs an Assign operation.
    ///
    /// \param new_value   Node that produces the input tensor.
    /// \param variable_id identifier of the variable to be updated.
    Assign(const Output<Node>& new_value, const std::string& variable_id);

    void validate_and_infer_types() override;
    std::string get_variable_id() const override {
        return m_variable_id;
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool visit_attributes(AttributeVisitor& visitor) override;

private:
    std::string m_variable_id;
};
}  // namespace v3
namespace v6 {
/// \brief Assign operation sets an input value to the variable with `variable_id`
class NGRAPH_API Assign : public AssignBase {
public:
    NGRAPH_RTTI_DECLARATION;
    Assign() = default;

    /// \brief Constructs an Assign operation.
    ///
    /// \param new_value Node that produces the input tensor.
    /// \param variable Class for storing and synchronizing element types, shapes and
    /// identifiers
    /// between pairs of Assign/ReadValue nodes.
    Assign(const Output<Node>& new_value, const std::shared_ptr<Variable>& variable);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::string get_variable_id() const override {
        NGRAPH_CHECK(m_variable, "Variable is not initialized. Variable_id is unavailable");
        return m_variable->get_info().variable_id;
    }
    bool evaluate(const HostTensorVector& outputs,
                  const HostTensorVector& inputs,
                  const EvaluationContext& evaluation_context) const override;
    bool has_evaluate() const override;
    bool constant_fold(OutputVector& output_values, const OutputVector& inputs_values) override;
};
}  // namespace v6
}  // namespace op
}  // namespace ngraph
