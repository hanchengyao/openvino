// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

namespace v0 {
/// \brief Operator performing Mean Variance Normalization
///
class NGRAPH_API MVN : public Op {
public:
    NGRAPH_RTTI_DECLARATION;

    MVN() = default;
    /// \brief Constructs an MVN operation.
    ///
    /// \param data Input tensor with data
    /// \param normalize_variance flag that denotes whether to perform variance
    ///                           normalization.
    /// \param across_channels flag that denotes if mean values are shared across
    /// channels.
    /// \param eps the number to be added to the variance to avoid division by zero when
    ///            normalizing the value
    ///
    MVN(const Output<Node>& data, bool across_channels = true, bool normalize_variance = true, double eps = 1e-9);

    /// \brief Constructs an MVN operation.
    ///
    /// \param data Input tensor with data
    /// \param reduction_axes A list of axes, along which to reduce.
    /// \param normalize_variance flag that denotes whether to perform variance
    ///                           normalization.
    /// \param eps the number to be added to the variance to avoid division by zero when
    ///            normalizing the value
    ///
    MVN(const Output<Node>& data, AxisSet reduction_axes, bool normalize_variance = true, double eps = 1e-9);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    double get_eps() const {
        return m_eps;
    }
    bool get_across_channels() const {
        return m_across_channels;
    }
    bool get_normalize_variance() const {
        return m_normalize_variance;
    }
    AxisSet get_reduction_axes() const {
        return m_reduction_axes;
    }
    void set_reduction_axes(AxisSet axes) {
        m_reduction_axes = axes;
    }

private:
    double m_eps;
    bool m_across_channels;
    bool m_normalize_variance;
    AxisSet m_reduction_axes;
};
}  // namespace v0
using v0::MVN;

/// \brief Specifies how eps is applied in MVN
enum class MVNEpsMode {
    // Apply eps inside sqrt
    INSIDE_SQRT,
    // Apply eps outside sqrt
    OUTSIDE_SQRT
};

NGRAPH_API
std::ostream& operator<<(std::ostream& s, const MVNEpsMode& type);

namespace v6 {
/// \brief Operator performing Mean Variance Normalization
///
class NGRAPH_API MVN : public ngraph::op::Op {
public:
    NGRAPH_RTTI_DECLARATION;

    MVN() = default;
    /// \brief Constructs an MVN operation.
    ///
    /// \param data Input tensor with data
    /// \param reduction_axes A list of axes, along which to reduce.
    /// \param normalize_variance flag that denotes whether to perform variance
    ///                           normalization.
    /// \param eps the number to be added to the variance to avoid division by zero when
    ///            normalizing the value
    /// \param eps_mode the mode of applying epsilon
    ///
    MVN(const Output<Node>& data,
        const Output<Node>& reduction_axes,
        bool normalize_variance,
        float eps,
        MVNEpsMode eps_mode);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    float get_eps() const {
        return m_eps;
    }
    bool get_normalize_variance() const {
        return m_normalize_variance;
    }
    MVNEpsMode get_eps_mode() const {
        return m_eps_mode;
    }

private:
    bool m_normalize_variance;
    float m_eps;
    MVNEpsMode m_eps_mode;
};
}  // namespace v6
}  // namespace op
}  // namespace ngraph

namespace ov {

template <>
class NGRAPH_API AttributeAdapter<ngraph::op::MVNEpsMode> : public EnumAttributeAdapterBase<ngraph::op::MVNEpsMode> {
public:
    AttributeAdapter(ngraph::op::MVNEpsMode& value) : EnumAttributeAdapterBase<ngraph::op::MVNEpsMode>(value) {}

    static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<op::MVNEpsMode>", 0};
    const DiscreteTypeInfo& get_type_info() const override {
        return type_info;
    }
};

}  // namespace ov
