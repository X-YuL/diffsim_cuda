#include <torch/extension.h>
#include <vector>

// Declared in srbd_cuda.cu
std::vector<torch::Tensor> srbd_step_cuda_launch(
    torch::Tensor p,
    torch::Tensor v,
    torch::Tensor q,
    torch::Tensor w,
    torch::Tensor f_world,
    torch::Tensor q_ref12,
    float m, float g, float Ixx, float Iyy, float Izz, float dt
);

torch::Tensor foot_positions_cuda_launch(
    torch::Tensor p_base,
    torch::Tensor q,
    torch::Tensor q_ref12
);

#define CHECK_CUDA(x)        TORCH_CHECK((x).device().is_cuda(),   #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)  TORCH_CHECK((x).is_contiguous(),      #x " must be contiguous")
#define CHECK_FLOAT(x)       TORCH_CHECK((x).dtype() == torch::kFloat32, #x " must be float32")
#define CHECK_INPUT(x)       CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

std::vector<torch::Tensor> srbd_step_forward(
    torch::Tensor p,
    torch::Tensor v,
    torch::Tensor q,
    torch::Tensor w,
    torch::Tensor f_world,
    torch::Tensor q_ref12,
    float m, float g, float Ixx, float Iyy, float Izz, float dt
) {
    CHECK_INPUT(p);
    CHECK_INPUT(v);
    CHECK_INPUT(q);
    CHECK_INPUT(w);
    CHECK_INPUT(f_world);
    CHECK_INPUT(q_ref12);
    return srbd_step_cuda_launch(p, v, q, w, f_world, q_ref12, m, g, Ixx, Iyy, Izz, dt);
}

torch::Tensor foot_positions_forward(
    torch::Tensor p_base,
    torch::Tensor q,
    torch::Tensor q_ref12
) {
    CHECK_INPUT(p_base);
    CHECK_INPUT(q);
    CHECK_INPUT(q_ref12);
    return foot_positions_cuda_launch(p_base, q, q_ref12);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("srbd_step_forward",
          &srbd_step_forward,
          "SRBD dynamics step — forward pass (CUDA). "
          "Returns [p_new(B,3), v_new(B,3), q_new(B,4), w_new(B,3)].");
    m.def("foot_positions_forward",
          &foot_positions_forward,
          "Foot world positions from SRBD state (CUDA). "
          "Returns p_foot(B,4,3).");
}
