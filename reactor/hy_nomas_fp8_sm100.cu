/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

// CUTLASS SM100 blockwise FP8 GEMM with the released PixelUMM FP32 scales.
// Configuration follows NVIDIA CUTLASS example 81; no scale requantization.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"

using namespace cute;
using PixelFP8 = cutlass::float_e4m3_t;
using PixelBF16 = cutlass::bfloat16_t;
using RowLayout = cutlass::layout::RowMajor;
using ColumnLayout = cutlass::layout::ColumnMajor;
using MmaTile = Shape<_128, _128, _128>;
using Cluster = Shape<_1, _1, _1>;
using Scales = cutlass::detail::Sm1xxBlockwiseScaleConfig<128, 128, 128, UMMA::Major::K, UMMA::Major::K>;
using SA = decltype(Scales::deduce_layoutSFA());
using SB = decltype(Scales::deduce_layoutSFB());
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTile, Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float,
    PixelBF16, RowLayout, 8, PixelBF16, RowLayout, 8, cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    PixelFP8, cute::tuple<RowLayout, SA>, 16, PixelFP8, cute::tuple<ColumnLayout, SB>, 16,
    float, MmaTile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epilogue::SharedStorage)>,
    cutlass::gemm::KernelScheduleSm100Blockwise>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

at::Tensor pixel_gemm(at::Tensor a, at::Tensor b, at::Tensor sa, at::Tensor sb) {
    TORCH_CHECK(a.is_cuda() && a.dim()==2 && b.dim()==2, "Expected CUDA matrices");
    c10::cuda::CUDAGuard guard(a.device());
    for (auto t : {a,b,sa,sb})
        TORCH_CHECK(t.device()==a.device() && t.is_contiguous(), "Expected contiguous tensors on one device");
    TORCH_CHECK(a.scalar_type()==at::kFloat8_e4m3fn && b.scalar_type()==at::kFloat8_e4m3fn,
                "Expected E4M3 FP8 operands");
    TORCH_CHECK(sa.scalar_type()==at::kFloat && sb.scalar_type()==at::kFloat, "Expected FP32 scales");
    int m=a.size(0), n=b.size(0), k=a.size(1);
    TORCH_CHECK(m>0 && n>0 && k>0 && b.size(1)==k && n%128==0 && k%128==0, "Invalid GEMM shape");
    TORCH_CHECK(sa.dim()==2 && sa.size(0)==(m+127)/128 && sa.size(1)==k/128, "Invalid activation scales");
    TORCH_CHECK(sb.dim()==2 && sb.size(0)==n/128 && sb.size(1)==k/128, "Invalid weight scales");
    auto prop=at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(prop->major==10 && prop->minor==0, "SM100 FP8 requires Blackwell SM100");
    auto out=at::empty({m,n},a.options().dtype(at::kBFloat16));
    typename Kernel::StrideA da=make_stride(int64_t(k),_1{},int64_t(m)*k);
    typename Kernel::StrideB db=make_stride(int64_t(k),_1{},int64_t(n)*k);
    typename Kernel::StrideC dc=make_stride(int64_t(n),_1{},int64_t(m)*n);
    typename Kernel::StrideD dd=dc;
    auto shape=make_shape(m,n,k,1);
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {m,n,k,1},
        {reinterpret_cast<PixelFP8 const*>(a.data_ptr()),da,
         reinterpret_cast<PixelFP8 const*>(b.data_ptr()),db,
         sa.data_ptr<float>(),Scales::tile_atom_to_shape_SFA(shape),
         sb.data_ptr<float>(),Scales::tile_atom_to_shape_SFB(shape)},
        {{},reinterpret_cast<PixelBF16 const*>(out.data_ptr()),dc,
            reinterpret_cast<PixelBF16*>(out.data_ptr()),dd}};
    args.epilogue.thread.alpha=1.f;
    args.epilogue.thread.beta=0.f;
    auto workspace=at::empty({int64_t(Gemm::get_workspace_size(args))},a.options().dtype(at::kByte));
    Gemm op;
    auto status=op.can_implement(args);
    TORCH_CHECK(status==cutlass::Status::kSuccess,"Unsupported SM100 FP8 GEMM: ",cutlassGetStatusString(status));
    auto stream=at::cuda::getCurrentCUDAStream();
    status=op.initialize(args,workspace.data_ptr(),stream);
    TORCH_CHECK(status==cutlass::Status::kSuccess,"SM100 FP8 initialization failed: ",cutlassGetStatusString(status));
    status=op.run(stream);
    TORCH_CHECK(status==cutlass::Status::kSuccess,"SM100 FP8 launch failed: ",cutlassGetStatusString(status));
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("gemm",&pixel_gemm); }
