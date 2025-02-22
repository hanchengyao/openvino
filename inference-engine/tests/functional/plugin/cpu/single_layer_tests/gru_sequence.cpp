// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/gru_sequence.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

using GRUSequenceCpuSpecificParams = typename std::tuple<LayerTestsDefinitions::GRUSequenceParams, CPUSpecificParams, std::map<std::string, std::string>>;

class GRUSequenceCPUTest : public testing::WithParamInterface<GRUSequenceCpuSpecificParams>,
                           virtual public LayerTestsUtils::LayerTestsCommon,
                           public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GRUSequenceCpuSpecificParams> &obj) {
        CPUSpecificParams cpuParams;
        LayerTestsDefinitions::GRUSequenceParams basicParamsSet;
        std::map<std::string, std::string> additionalConfig;

        std::tie(basicParamsSet, cpuParams, additionalConfig) = obj.param;
        std::ostringstream result;

        result << LayerTestsDefinitions::GRUSequenceTest::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::GRUSequenceParams>(basicParamsSet, 0));
        result << CPUTestsBase::getTestCaseName(cpuParams);

        if (!additionalConfig.empty()) {
            result << "_PluginConf";
            for (auto &item : additionalConfig) {
                if (item.second == PluginConfigParams::YES)
                    result << "_" << item.first << "=" << item.second;
            }
        }
        return result.str();
    }

protected:
    void SetUp() override {
        LayerTestsDefinitions::GRUSequenceParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;

        size_t seq_lengths;
        size_t batch;
        size_t hidden_size;
        size_t input_size = 10;
        std::vector<std::string> activations;
        std::vector<float> activations_alpha;
        std::vector<float> activations_beta;
        float clip;
        bool linear_before_reset;
        ngraph::op::RecurrentSequenceDirection direction;
        InferenceEngine::Precision netPrecision;

        std::tie(basicParamsSet, cpuParams, additionalConfig) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::tie(m_mode, seq_lengths, batch, hidden_size, activations, clip, linear_before_reset, direction, netPrecision, targetDevice) = basicParamsSet;

        size_t num_directions = direction == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL ? 2 : 1;
        std::vector<std::vector<size_t>> inputShapes = {
            {{batch, seq_lengths, input_size},
             {batch, num_directions, hidden_size},
             {batch},
             {num_directions, 3 * hidden_size, input_size},
             {num_directions, 3 * hidden_size, hidden_size},
             {num_directions, (linear_before_reset ? 4 : 3) * hidden_size}},
        };

        // method MKLDNNMemoryDesc::isSame can't correct compute layout for tensor with strides = 1
        // returned output format always tnc
        if (inFmts.size() == 2 && ngraph::shape_size(inputShapes[1]) == 1) {
            inFmts[1] = tnc;
        }

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        if (additionalConfig[PluginConfigParams::KEY_ENFORCE_BF16] == PluginConfigParams::YES) {
            inPrc = outPrc = Precision::BF16;
        } else {
            inPrc = outPrc = netPrecision;
        }

        selectedType += "_";
        selectedType += outPrc.name();

        m_max_seq_len = seq_lengths;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(Precision::FP32);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShapes[0], inputShapes[1]});
        if (m_mode == ngraph::helpers::SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_PARAM
            || m_mode == ngraph::helpers::SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_PARAM) {
            auto seq_lengths = ngraph::builder::makeParams(ngraph::element::i64, {inputShapes[2]}).at(0);
            seq_lengths->set_friendly_name("seq_lengths");
            params.push_back(seq_lengths);
        }
        std::vector<ngraph::Shape> WRB = {inputShapes[3], inputShapes[4], inputShapes[5], inputShapes[2]};
        auto gru_sequence = ngraph::builder::makeGRU(ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes(params)),
                                                     WRB,
                                                     hidden_size,
                                                     activations,
                                                     {},
                                                     {},
                                                     clip,
                                                     linear_before_reset,
                                                     true,
                                                     direction,
                                                     m_mode);

        // method MKLDNNMemoryDesc::isSame can't correct compute layout for tensor with strides = 1
        // returned output format always tnc
        if (ngraph::shape_size(gru_sequence->get_output_shape(0)) == 1) {
            outFmts[0] = tnc;
        } else if (ngraph::shape_size(gru_sequence->get_output_shape(1)) == 1 ||
                gru_sequence->get_output_shape(0)[0] == 1) {
            outFmts[1] = tnc;
        }

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(gru_sequence->output(0)),
                                     std::make_shared<ngraph::opset1::Result>(gru_sequence->output(1))};

        function = makeNgraphFunction(ngPrc, params, gru_sequence, "gru_sequence");

        if (m_mode != ngraph::helpers::SequenceTestsMode::PURE_SEQ) {
            ngraph::pass::Manager manager;
            if (direction == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL)
                manager.register_pass<ngraph::pass::BidirectionalGRUSequenceDecomposition>();
            manager.register_pass<ngraph::pass::ConvertGRUSequenceToTensorIterator>();
            manager.run_passes(function);
            bool ti_found = ngraph::helpers::is_tensor_iterator_exist(function);
            EXPECT_EQ(ti_found, true);
        } else {
            bool ti_found = ngraph::helpers::is_tensor_iterator_exist(function);
            EXPECT_EQ(ti_found, false);
        }
    }

    void GenerateInputs() override {
        for (const auto &input : executableNetwork.GetInputsInfo()) {
            const auto &info = input.second;
            auto blob = GenerateInput(*info);
            if (input.first == "seq_lengths") {
                blob = FuncTestUtils::createAndFillBlob(info->getTensorDesc(), m_max_seq_len, 0);
            }
            inputs.push_back(blob);
        }
    }

private:
    ngraph::helpers::SequenceTestsMode m_mode;
    int64_t m_max_seq_len = 0;
};

TEST_P(GRUSequenceCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "RNNSeq");
}

namespace {
/* CPU PARAMS */
std::vector<std::map<std::string, std::string>> additionalConfig
    = {{{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::NO}}, {{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::YES}}};

CPUSpecificParams cpuParams{{ntc, ntc}, {ntc, ntc}, {"ref_any"}, "ref_any"};
CPUSpecificParams cpuParamsBatchSizeOne{{tnc, ntc}, {tnc, ntc}, {"ref_any"}, "ref_any"};;

std::vector<ngraph::helpers::SequenceTestsMode> mode{ngraph::helpers::SequenceTestsMode::PURE_SEQ};
// output values increase rapidly without clip, so use only seq_lengths = 2
std::vector<size_t> seq_lengths_zero_clip{2};
std::vector<size_t> batch{10};
std::vector<size_t> batch_size_one{1};
std::vector<size_t> hidden_size{1, 10};
std::vector<std::vector<std::string>> activations = {{"sigmoid", "tanh"}};
std::vector<bool> linear_before_reset = {true, false};
std::vector<float> clip{0.f};
std::vector<ngraph::op::RecurrentSequenceDirection> direction = {ngraph::op::RecurrentSequenceDirection::FORWARD};

std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32};

INSTANTIATE_TEST_SUITE_P(smoke_GRUSequenceCPU,
                        GRUSequenceCPUTest,
                        ::testing::Combine(::testing::Combine(::testing::ValuesIn(mode),
                                                              ::testing::ValuesIn(seq_lengths_zero_clip),
                                                              ::testing::ValuesIn(batch),
                                                              ::testing::ValuesIn(hidden_size),
                                                              ::testing::ValuesIn(activations),
                                                              ::testing::ValuesIn(clip),
                                                              ::testing::ValuesIn(linear_before_reset),
                                                              ::testing::ValuesIn(direction),
                                                              ::testing::ValuesIn(netPrecisions),
                                                              ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                           ::testing::Values(cpuParams),
                                           ::testing::ValuesIn(additionalConfig)),
                        GRUSequenceCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GRUSequenceCPUBatchSizeOne,
                        GRUSequenceCPUTest,
                        ::testing::Combine(::testing::Combine(::testing::ValuesIn(mode),
                                                              ::testing::ValuesIn(seq_lengths_zero_clip),
                                                              ::testing::ValuesIn(batch_size_one),
                                                              ::testing::ValuesIn(hidden_size),
                                                              ::testing::ValuesIn(activations),
                                                              ::testing::ValuesIn(clip),
                                                              ::testing::ValuesIn(linear_before_reset),
                                                              ::testing::ValuesIn(direction),
                                                              ::testing::ValuesIn(netPrecisions),
                                                              ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                           ::testing::Values(cpuParamsBatchSizeOne),
                                           ::testing::ValuesIn(additionalConfig)),
                        GRUSequenceCPUTest::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions
