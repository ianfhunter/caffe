#include "caffe/layers/python_layer.hpp"

namespace caffe {

#ifdef WITH_PYTHON_LAYER

INSTANTIATE_CLASS_3T_GUARDED(PythonLayer, (half_fp), PROTO_TYPES, PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(PythonLayer, (float), PROTO_TYPES, PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(PythonLayer, (double), PROTO_TYPES, PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(PythonLayer, (uint8_t), PROTO_TYPES, PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(PythonLayer, (uint16_t), PROTO_TYPES, PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(PythonLayer, (uint32_t), PROTO_TYPES, PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(PythonLayer, (uint64_t), PROTO_TYPES, PROTO_TYPES);

#endif  // WITH_PYTHON_LAYER

}  // namespace caffe
