import os
import sys
cur_dir = os.getcwd()
aries_path = cur_dir + "/../../../../"
sys.path.append(aries_path)
from frontend import *

# GEMM0: C[i0, j0] += A[i0, k0] * B[k0, j0] (64*64*128)
# GEMM1: E[i0, n0] += C[i0, j0] * D[j0, n0] (64*128*64)
I, J, K, N = 64, 128, 64, 64
TI, TJ, TK = 32, 32, 32
GRID_I0, GRID_J0, GRID_K0 = (I // TI, J // TJ, K // TK)
GRID_I1, GRID_J1, GRID_K1 = (I // TI, N // TJ, J // TK)

@task_kernel(external_path="aie1/adf/kernel_mm/aie_fp32_v0", para = [TI, TJ, TK])
def kernel_gemm(TileA: float32[TI, TK], 
                TileB: float32[TK, TJ], 
                TileC: float32[TI, TJ]):
    for i0 in range(0, TI):
        for j0 in range(0, TJ):
            TileC[i0, j0] = float32(0)
            for k0 in range(0, TK):
                TileC[i0, j0] += TileA[i0, k0] * TileB[k0, j0]



@task_tile()
def gemm(A: float32[-1, -1], B: float32[-1, -1], 
         C: float32[-1, -1], GRID_I, GRID_J, GRID_K):
    for i in range(GRID_I):
        for j in range(GRID_J):
            for k in range(GRID_K):
                # Compute tile slices for multiple dimensions
                ti = aries.arange(i*TI, (i+1)*TI)  # I tile range
                tj = aries.arange(j*TJ, (j+1)*TJ)  # J tile range
                tk = aries.arange(k*TK, (k+1)*TK)  # K tile range

                L1_A = aries.buffer((TI, TK), "float32")
                L1_B = aries.buffer((TK, TJ), "float32")
                L1_C = aries.buffer((TI, TJ), "float32")

                L1_A = aries.load(A, (ti, tk))
                L1_B = aries.load(B, (tk, tj))
                kernel_gemm(L1_A, L1_B, L1_C)
                aries.accstore(L1_C, C, (ti, tj))

@task_top()
def top(A: float32[I, K], B: float32[K, J], C: float32[I, J],
        D: float32[J, N], E: float32[I, N]):
    cast_A = aries.cast(A, (-1, -1)) # This is for lowering
    cast_B = aries.cast(B, (-1, -1))
    cast_C = aries.cast(C, (-1, -1))
    cast_D = aries.cast(D, (-1, -1))
    cast_E = aries.cast(E, (-1, -1))
    gemm_task = gemm(cast_A, cast_B, cast_C, GRID_I0, GRID_J0, GRID_K0)
    gemm_task = gemm(cast_C, cast_D, cast_E, GRID_I1, GRID_J1, GRID_K1)
    return gemm_task

all_code = sys.modules[__name__]



# Initialize the buffers
np.random.seed(0)
A = np.random.rand(I, K).astype(np.float32)
B = np.random.rand(K, J).astype(np.float32)
C = np.zeros((I, J)).astype(np.float32)
D = np.random.rand(J, N).astype(np.float32)
E = np.zeros((I, N)).astype(np.float32)

# Execute on CPU
gemm_task = top(A, B, C, D, E)

# Golden file generation
golden_C = np.matmul(A, B)
golden_E = np.matmul(golden_C, D)

# Compare the program with golden file
print("ARIES MLP output matches golden reference:", np.allclose(E, golden_E))

# # Generate files for on-board test
aries.gen_sim([A, B, golden_C, D, golden_E])



# Specify primitives to optimize hardware design
sch = Schedule(gemm_task)

############# Primitives #############
sch.parallel(gemm_task, [1, 1, 1]) # AIE Array Parallelism
sch.l2buffer(gemm_task, [2, 2, 2]) # L2 buffer data reuse
sch.bufsel(gemm_task, [1, 1, 0]) # Select the type of buffer of A, B, C, 1:BRAM; 0:URAM
######################################

sch.to("VCK190")

# Set the project dir and template dir
prj_dir= cur_dir + '/project_overlay'
temp_dir= aries_path + '/templates'
# Generate Initial MLIR and ARIES Opts
sch.build(all_code, prj_dir, temp_dir)
sch.compile(aries_path, prj_dir)