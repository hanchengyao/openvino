// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/descriptor/input.hpp"

#include "ngraph/env_util.hpp"
#include "ngraph/node.hpp"
#include "openvino/core/descriptor/output.hpp"
#include "openvino/core/type/element_type.hpp"

ov::descriptor::Input::Input(ngraph::Node* node, size_t index, Output& output)
    : m_node(node),
      m_index(index),
      m_output(&output),
      m_is_relevant_to_shape(false),
      m_is_relevant_to_value(true) {
    m_src_node = std::shared_ptr<ngraph::Node>(output.get_node());
    output.add_input(this);
}

ov::descriptor::Input::Input(ngraph::Node* node, size_t index)
    : m_node(node),
      m_index(index),
      m_output(nullptr),
      m_is_relevant_to_shape(false),
      m_is_relevant_to_value(true) {}

ov::descriptor::Input::~Input() {
    remove_output();
}

void ov::descriptor::Input::replace_output(Output& new_output) {
    if (m_output != nullptr) {
        m_output->remove_input(this);
    }
    new_output.add_input(this);
    m_output = &new_output;
    m_src_node = std::shared_ptr<ngraph::Node>(new_output.get_node());

    if (ngraph::getenv_bool("NGRAPH_ENABLE_REPLACE_CHECK")) {
        // the result of clone_with_new_inputs will be thrown away or
        // an exception will be thrown by `m_node`'s class c-tor
        // if a new input violates one of the type checks in the c-tor.
        m_node->clone_with_new_inputs(m_node->input_values());
    }
}

void ov::descriptor::Input::replace_output(const std::shared_ptr<ngraph::Node>& node, size_t i) {
    replace_output(node->m_outputs.at(i));
}

void ov::descriptor::Input::remove_output() {
    if (m_output != nullptr) {
        m_output->remove_input(this);
        m_src_node = nullptr;
        m_output = nullptr;
    }
}

std::shared_ptr<ngraph::Node> ov::descriptor::Input::get_node() const {
    return m_node->shared_from_this();
}

const ov::descriptor::Tensor& ov::descriptor::Input::get_tensor() const {
    return m_output->get_tensor();
}

ov::descriptor::Tensor& ov::descriptor::Input::get_tensor() {
    return m_output->get_tensor();
}

std::shared_ptr<const ov::descriptor::Tensor> ov::descriptor::Input::get_tensor_ptr() const {
    return m_output->get_tensor_ptr();
}

std::shared_ptr<ov::descriptor::Tensor> ov::descriptor::Input::get_tensor_ptr() {
    return m_output->get_tensor_ptr();
}

const ngraph::Shape& ov::descriptor::Input::get_shape() const {
    return m_output->get_shape();
}

const ov::PartialShape& ov::descriptor::Input::get_partial_shape() const {
    return m_output->get_partial_shape();
}

const ov::element::Type& ov::descriptor::Input::get_element_type() const {
    return m_output->get_element_type();
}
