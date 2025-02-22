// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, cos_float) {
    Shape shape{11};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Cos>(A), ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({0.f, 0.25f, -0.25f, 0.5f, -0.5f, 1.f, -1.f, 2.f, -2.f, 4.f, -4.f});
    test_case.add_expected_output<float>(shape,
                                         {1.00000000f,
                                          0.96891242f,
                                          0.96891242f,
                                          0.87758256f,
                                          0.87758256f,
                                          0.54030231f,
                                          0.54030231f,
                                          -0.41614684f,
                                          -0.41614684f,
                                          -0.65364362f,
                                          -0.65364362f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, cos_int) {
    Shape shape{5};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Cos>(A), ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int32_t>({1, 2, 3, 4, 5});
    test_case.add_expected_output<int32_t>(shape, {1, 0, -1, -1, 0});
    test_case.run();
}
