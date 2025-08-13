import os

class TensorInfo:
    def __init__(self, name, shape, dtype, fix_names=False):
        self.name = name
        if fix_names:
            self.name = self.name.replace("/", "_").replace(";", "_")
        assert isinstance(shape, (tuple, list))
        self.shape = shape
        size_lookup = {
            "float32": 4,
            "uint8": 1,
            "int8": 1,
            "int32": 4,
            "int64": 8,
        }
        assert dtype in size_lookup, f"Unsupported type: {dtype}"
        self.dtype = dtype
        self.type_size = size_lookup[self.dtype]

    @property
    def size(self):
        ret = self.type_size
        for dim in self.shape:
            if isinstance(dim, complex):
                real = dim.real
                imag = dim.imag
                assert real == int(real)
                assert imag == int(imag)
                ret *= int(real) + int(imag)
            else:
                ret *= dim
        return ret


class TfLiteTensorInfo(TensorInfo):
    def __init__(self, t, fix_names=False):
        # Local imports to get rid of tflite dependency for non-tflite models
        from tflite.TensorType import TensorType as TType

        name = t.Name().decode()
        shape = tuple([t.Shape(si) for si in range(0, t.ShapeLength())])

        type_lookup = {
            TType.FLOAT32: "float32",
            TType.UINT8: "uint8",
            TType.INT8: "int8",
            TType.INT32: "int32",
            TType.BOOL: "int8",
            TType.INT64: "int64",
        }
        dtype = type_lookup[t.Type()]
        super().__init__(name, shape, dtype, fix_names=fix_names)


class ModelInfo:
    def __init__(self, in_tensors, out_tensors, fix_names=False):
        self.in_tensors = in_tensors
        self.out_tensors = out_tensors

    def validate(self):
        assert len(self.in_tensors) > 0, "Missing inputs"
        assert len(self.out_tensors) > 0, "Missing outputs"

    @property
    def has_ins(self):
        return len(self.in_tensors) > 0

    @property
    def has_outs(self):
        return len(self.out_tensors) > 0


class TfLiteModelInfo(ModelInfo):
    def __init__(self, model, fix_names=False):
        assert model.SubgraphsLength() == 1
        g = model.Subgraphs(0)

        in_tensors = []
        for i in range(0, g.InputsLength()):
            t = g.Tensors(g.Inputs(i))
            in_tensors.append(TfLiteTensorInfo(t, fix_names=fix_names))

        out_tensors = []
        for i in range(0, g.OutputsLength()):
            t = g.Tensors(g.Outputs(i))
            out_tensors.append(TfLiteTensorInfo(t, fix_names=fix_names))
        super().__init__(in_tensors, out_tensors)



def get_tflite_model_info(model_buf):
    # Local imports to get rid of tflite dependency for non-tflite models
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(model_buf, 0)
    model_info = TfLiteModelInfo(tflite_model)
    return model_info


def get_model_info(model):
    ext = os.path.splitext(model)[1][1:].lower()
    if ext == "tflite":
        with open(model, "rb") as handle:
            model_buf = handle.read()
            return get_tflite_model_info(model_buf)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
