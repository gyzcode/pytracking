import onnx
import onnx.shape_inference

def expose_node_outputs(model_path, overwrite, run_checker=True, verbose=False):
    """
    Exposes each intermediary node as one of the ONNX model's graph outputs.
     This allows inspection of data passing through the model.
    :param model_path: (str) The path to the .onnx model, with the extension.
    :param overwrite:  (boolean) If true, will overwrite the .onnx file at model_path, else will make a copy.
    :param run_checker: (boolean) If true, will run the ONNX model validity checker
    :param verbose: (boolean) If true, will print detailed messages about execution of preprocessing script
    :return: (str) file path to new ONNX model
    """
    # 1. Get name of all external outputs to the model (ie. graph-level outputs, not internal outputs shared bw nodes)
    model = onnx.load(model_path)
    external_outputs = [output.name for output in model.graph.output]
    extended_outputs = []

    # 2. Get the list of nodes in the graph
    for i, node in enumerate(model.graph.node):
        # 3. For every node, copy its (internal) output over to graph.output to make it a graph output
        output_name = [output for output in node.output if output not in external_outputs]
        extended_outputs.extend(output_name)
        for output in output_name:
            intermediate_layer_value_info = onnx.helper.make_tensor_value_info(output, onnx.TensorProto.UNDEFINED, None,
                                                                               'Added to expose Intermediate Node data')
            model.graph.output.extend([intermediate_layer_value_info])

    if verbose:
        print('The following nodes were exposed as outputs in the {} model:\n {}'.format(model_path, extended_outputs))

    # If all outputs were already "external", no changes are required to the ONNX model, return it as-is
    if len(external_outputs) == len(model.graph.output):
        if verbose:
            print('No change required for ONNX model: All nodes already exposed as outputs')
        return model_path

    # 4. Do a shape and type inference pass on the model to ensure they're defined for graph outputs
    model = onnx.shape_inference.infer_shapes(model)
    # 4.5 Remove every output node for which the type or shape could not be inferred
    for i, tensor_valueproto in reversed(list(enumerate(model.graph.output))):
        if not tensor_has_valid_type(tensor_valueproto, verbose):
            del model.graph.output[i]

    if run_checker:
        try:
            onnx.checker.check_model(model)
        except onnx.checker.ValidationError as v:
            # Ignoring this specific error because the ONNX spec says a missing shape is the legal way to define
            # a tensor of unknown dimensions. Honestly, believe this is a bug in the checker.
            # See https://github.com/onnx/onnx/issues/2492
            if str(v).endswith("Field 'shape' of type is required but missing."):
                if verbose:
                    print("Warning: Ignoring the following error because it is probably an ONNX Checker error: ", v)
            else:
                raise v

    if not overwrite:
        # Make a copy of the .onnx model to save it as a file
        model_path_components = model_path.rsplit(".", 1)  # Split before and after extension
        model_path = model_path_components[0] + '_exposed_nodes.' + model_path_components[1]
    onnx.save(model, model_path)
    return model_path


def tensor_has_valid_type(tensor_valueproto, verbose):
    """ Ensures ValueProto tensor element type is not UNDEFINED"""
    if tensor_valueproto.type.tensor_type.elem_type == onnx.TensorProto.UNDEFINED:
        if verbose:
            print('Type could not be inferred for the following output, it will be not be exposed:\n',
                  tensor_valueproto)
        return False
    return True

def main():
    expose_node_outputs('dimp50.onnx', False)


if __name__ == '__main__':
    main()