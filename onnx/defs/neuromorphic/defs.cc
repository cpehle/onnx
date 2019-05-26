// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

void NMShapeInference(InferenceContext& ctx) {
    TensorShapeProto::Dimension num_directions, seq_length, batch_size,
      hidden_size;

    auto direction = getAttribute(ctx, "direction", "forward");
    if ((direction == "forward") || (direction == "reverse"))
      num_directions.set_dim_value(1);
    else if (direction == "bidirectional")
      num_directions.set_dim_value(2);
    // else leave num_directions unknown in case of incorrect attribute value

    auto hidden_size_value = getAttribute(ctx, "hidden_size", -1);
    if (hidden_size_value > 0)
      hidden_size.set_dim_value(hidden_size_value);

    if (hasInputShape(ctx, 0)) {
      auto& first_input_shape = getInputShape(ctx, 0);
      if (first_input_shape.dim_size() != 3) {
        fail_shape_inference("First input tensor must have rank 3");
      }
      seq_length = first_input_shape.dim(0);
      batch_size = first_input_shape.dim(1);
    }

    auto num_outputs = ctx.getNumOutputs();

    if (num_outputs > 0) {
      // Y
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      updateOutputShape(
          ctx, 0, {seq_length, num_directions, batch_size, hidden_size});
    }

    if (num_outputs > 1) {
      // Y_h
      propagateElemTypeFromInputToOutput(ctx, 0, 1);
      updateOutputShape(ctx, 1, {num_directions, batch_size, hidden_size});
    }

    if (num_outputs > 2) {
      // Y_c : only in the case of LSTM
      propagateElemTypeFromInputToOutput(ctx, 0, 2);
      updateOutputShape(ctx, 2, {num_directions, batch_size, hidden_size});
    }
}

std::function<void(OpSchema&)> NMDocGenerator(const char* /*name*/) {
  return [=](OpSchema& schema) {
    schema.Attr(
        "direction",
        "Specify if the RNN is forward, reverse, or bidirectional. "
        "Must be one of forward (default), reverse, or bidirectional.",
        AttributeProto::STRING,
        std::string("forward"));
    schema.Attr(
        "hidden_size",
        "Number of neurons in the hidden layer",
        AttributeProto::INT,
        OPTIONAL);
    schema.Attr(
        "activation_alpha",
        "Optional scaling values used by some activation functions. The values "
        "are consumed in the order of activation functions, for example (f, g, h) "
        "in LSTM. Default values are the same as of corresponding ONNX operators."
        "For example with LeakyRelu, the default alpha is 0.01.",
        AttributeProto::FLOATS,
        OPTIONAL);
    schema.Attr(
        "activation_beta",
        "Optional scaling values used by some activation functions. The values "
        "are consumed in the order of activation functions, for example (f, g, h) "
        "in LSTM. Default values are the same as of corresponding ONNX operators.",
        AttributeProto::FLOATS,
        OPTIONAL);
    schema.Input(
        0,
        "X",
        "The input sequences packed (and potentially padded) into one 3-D "
        "tensor with the shape of `[seq_length, batch_size, input_size]`.",
        "T");
    schema.Input(
        3,
        "sequence_lens",
        "Optional tensor specifying lengths of the sequences in a batch. "
        "If not specified - assumed all sequences in the batch to have "
        "length `seq_length`. It has shape `[batch_size]`.",
        "T1",
        OpSchema::Optional);
    schema.Input(
        4,
        "initial_h",
        "Optional initial value of the hidden. If not specified - assumed "
        "to be 0. It has shape `[num_directions, batch_size, hidden_size]`.",
        "T",
        OpSchema::Optional);
    schema.Output(
        0,
        "Y",
        "A tensor that concats all the intermediate output values of the hidden. "
        "It has shape `[seq_length, num_directions, batch_size, hidden_size]`. ",
        "T",
        OpSchema::Optional);
    schema.Output(
        1,
        "Y_h",
        "The last output value of the hidden. It has shape "
        "`[num_directions, batch_size, hidden_size]`.",
        "T",
        OpSchema::Optional);
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
    schema.TypeConstraint(
        "T1", {"tensor(int32)"}, "Constrain seq_lens to integer tensor.");
    schema.TypeAndShapeInferenceFunction(NMShapeInference);
  };
}



static const char* LIFCell_doc = R"DOC(
    Test
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    LIFCell,
    10,
    OpSchema()
        .SetDoc(LIFCell_doc + GenerateOptionalArgumentsDoc())
        .Input(
            1,
            "W",
            "The weight tensor for the gates. Concatenation of `W[iofc]` and "
            "`WB[iofc]` (if bidirectional) along dimension 0. The tensor has shape "
            "`[num_directions, 4*hidden_size, input_size]`.",
            "T")
        .Input(
            2,
            "R",
            "The recurrence weight tensor. Concatenation of `R[iofc]` and "
            "`RB[iofc]` (if bidirectional) along dimension 0. This tensor has shape "
            "`[num_directions, 4*hidden_size, hidden_size]`.",
            "T")
        .FillUsing(NMDocGenerator("LIF"))
)


static const char* LIFLayer_doc = R"DOC(
    Test
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    LIFLayer,
    10,
    OpSchema()
        .SetDoc(LIFLayer_doc + GenerateOptionalArgumentsDoc())
        .Input(
            1,
            "W",
            "The weight tensor for the gates. Concatenation of `W[iofc]` and "
            "`WB[iofc]` (if bidirectional) along dimension 0. The tensor has shape "
            "`[num_directions, 4*hidden_size, input_size]`.",
            "T")
        .Input(
            2,
            "R",
            "The recurrence weight tensor. Concatenation of `R[iofc]` and "
            "`RB[iofc]` (if bidirectional) along dimension 0. This tensor has shape "
            "`[num_directions, 4*hidden_size, hidden_size]`.",
            "T")
        .Attr(
            "v_thresh",
            "Membrane voltage threshhold",
            AttributeProto::FLOATS,
            OPTIONAL)
        .Attr(
            "v_leak",
            "Membrane voltage leak",
            AttributeProto::FLOATS,
            OPTIONAL
        )
        .Attr(
            "v_reset",
            "Membrane voltage reset",
            AttributeProto::FLOATS,
            OPTIONAL
        )
        .FillUsing(NMDocGenerator("LIF"))
)

static const char* LSNNCell_doc = R"DOC(
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    LSNNCell,
    10,
    OpSchema()
        .SetDoc(LSNNCell_doc + GenerateOptionalArgumentsDoc())
        .Input(
            1,
            "W",
            "The weight tensor for the gates. Concatenation of `W[iofc]` and "
            "`WB[iofc]` (if bidirectional) along dimension 0. The tensor has shape "
            "`[num_directions, 4*hidden_size, input_size]`.",
            "T")
        .Input(
            2,
            "R",
            "The recurrence weight tensor. Concatenation of `R[iofc]` and "
            "`RB[iofc]` (if bidirectional) along dimension 0. This tensor has shape "
            "`[num_directions, 4*hidden_size, hidden_size]`.",
            "T")
        .Attr(
            "v_thresh",
            "Membrane voltage threshhold",
            AttributeProto::FLOATS,
            OPTIONAL)
        .Attr(
            "v_leak",
            "Membrane voltage leak",
            AttributeProto::FLOATS,
            OPTIONAL
        )
        .Attr(
            "v_reset",
            "Membrane voltage reset",
            AttributeProto::FLOATS,
            OPTIONAL
        )
        .FillUsing(NMDocGenerator("LSNN"))
)


static const char* LSNNLayer_doc = R"DOC(
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    LSNNLayer,
    10,
    OpSchema()
        .SetDoc(LSNNLayer_doc + GenerateOptionalArgumentsDoc())
        .Input(
            1,
            "W",
            "The weight tensor for the gates. Concatenation of `W[iofc]` and "
            "`WB[iofc]` (if bidirectional) along dimension 0. The tensor has shape "
            "`[num_directions, 4*hidden_size, input_size]`.",
            "T")
        .Input(
            2,
            "R",
            "The recurrence weight tensor. Concatenation of `R[iofc]` and "
            "`RB[iofc]` (if bidirectional) along dimension 0. This tensor has shape "
            "`[num_directions, 4*hidden_size, hidden_size]`.",
            "T")
        .FillUsing(NMDocGenerator("LSNN"))
)

static const char* ADEXCell_doc = R"DOC(
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    ADEXCell,
    10,
    OpSchema()
        .SetDoc(ADEXCell_doc + GenerateOptionalArgumentsDoc())
        .Input(
            1,
            "W",
            "The weight tensor for the gates. Concatenation of `W[iofc]` and "
            "`WB[iofc]` (if bidirectional) along dimension 0. The tensor has shape "
            "`[num_directions, 4*hidden_size, input_size]`.",
            "T")
        .Input(
            2,
            "R",
            "The recurrence weight tensor. Concatenation of `R[iofc]` and "
            "`RB[iofc]` (if bidirectional) along dimension 0. This tensor has shape "
            "`[num_directions, 4*hidden_size, hidden_size]`.",
            "T")
        .FillUsing(NMDocGenerator("ADEX"))
)


static const char* ADEX_doc = R"DOC(
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    ADEXLayer,
    10,
    OpSchema()
        .SetDoc(ADEX_doc + GenerateOptionalArgumentsDoc())
        .Input(
            1,
            "W",
            "The weight tensor for the gates. Concatenation of `W[iofc]` and "
            "`WB[iofc]` (if bidirectional) along dimension 0. The tensor has shape "
            "`[num_directions, 4*hidden_size, input_size]`.",
            "T")
        .Input(
            2,
            "R",
            "The recurrence weight tensor. Concatenation of `R[iofc]` and "
            "`RB[iofc]` (if bidirectional) along dimension 0. This tensor has shape "
            "`[num_directions, 4*hidden_size, hidden_size]`.",
            "T")
        .FillUsing(NMDocGenerator("ADEX"))
)

static const char* OnHICANN_doc = R"DOC(
    Test
)DOC";

// ONNX_OPERATOR_SET_SCHEMA(
//     OnHICANN,
//     10,
//     OpSchema()
//         .SetDoc(OnHICANN_doc)
//         .Input(
//             0,
//             "subgraph_inputs",
//             "Inputs to the subgraph to be constrained.",
//             "V",
//             OpSchema::Variadic,
//             false)
//         .Output(
//             0,
//             "subgraph_outputs",
//             "Outputs to the subgraph to be constrained",
//             "V",
//             OpSchema::Variadic,
//             false)
//         .Attr("graph",
//             "Subgraph to be constrained.",
//             AttributeProto::GRAPH
//         )
//         .Attr("hicanns",
//             "A list of hicanns this graph is supposed to be placed on.",
//             AttributeProto::INTS
//         )
// )


} // namespace ONNX_NAMESPACE
