import os
import sys
cur_dir = os.getcwd()
aries_path = cur_dir + "/../../../../"
sys.path.append(aries_path)
from frontend import *
from IPython import get_ipython

# GEMM: C[i0, j0] += A[i0, k0] * B[k0, j0]
I, J, K = 256, 256, 256
TI, TJ, TK = 32, 32, 32
grid = (I // TI, J // TJ, K // TK)  # grid must be a tuple

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
def gemm(A: float32[I, K], B: float32[K, J], 
         C: float32[I, J], **kwargs):
    i, j, k = aries.tile_ranks(**kwargs)
    
    L1_A = aries.buffer((TI, TK), "float32")
    L1_B = aries.buffer((TK, TJ), "float32")
    L1_C = aries.buffer((TI, TJ), "float32")
    
    
    # Slice data using aries.arrage(start, stop)
    ti = aries.arange(i*TI, (i+1)*TI)
    tk = aries.arange(k*TK, (k+1)*TK)
    tj = aries.arange(j*TJ, (j+1)*TJ)  # J tile range
    
    
    L1_A = aries.load(A, (ti, tk))
    L1_B = aries.load(B, (tk, tj))
    kernel_gemm(L1_A, L1_B, L1_C)
    aries.accstore(L1_C, C, (ti, tj))



@task_top()
def top(A: float32[I, K], B: float32[K, J], C: float32[I, J]):
    gemm_task = gemm[grid](A, B, C)
    return gemm_task

all_code = sys.modules[__name__]

# Initialize the buffers
np.random.seed(0)
A = np.random.rand(I, K).astype(np.float32)
B = np.random.rand(K, J).astype(np.float32)
C = np.zeros((I, J)).astype(np.float32)

# Execute on CPU
gemm_task = top(A, B, C)
golden_C = np.matmul(A, B)

# Compare the program with golden file
print("ARIES gemm output matches golden reference:", np.allclose(C, golden_C))

# Generate files for on-board test
aries.gen_sim([A, B, golden_C])

# Apply schedulings
sch = Schedule(gemm_task)

sch.to("VCK190")

# Set the project dir and template dir
prj_dir= cur_dir + '/project_gemm_opt'
temp_dir= aries_path + '/templates'
# Generate Initial MLIR and ARIES Opts
sch.build(all_code, prj_dir, temp_dir)
sch.compile(aries_path, prj_dir, target = "report")

