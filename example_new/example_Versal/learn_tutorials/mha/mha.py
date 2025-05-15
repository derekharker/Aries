import os
import sys
cur_dir = os.getcwd()
aries_path = cur_dir + "/../../../../"
sys.path.append(aries_path)
from frontend import *

# MHA0:  Q_K[heads] = Q[heads][seq][head_dim] * K[heads][head_dim][seq]
# Softmax:  temp[heads][seq][seq] = softmax(Q_K, dim=3) / (head_dim) ^ (1/2)
# MHA1: out[heads][seq][head_dim] = temp[heads][seq][seq] * V[heads][seq][head_dim]

SEQ = 64
HEADS = 4
HEAD_DIM = 64
HIDDEN = HEADS * HEAD_DIM

TB, TI, TJ, TK = 1, 32, 32, 32
GRID_B0, GRID_I0, GRID_J0, GRID_K0 = HEADS//TB, SEQ // TI, HEAD_DIM // TJ, SEQ // TK
GRID_B1, GRID_I1, GRID_J1, GRID_K1 = HEADS//TB, SEQ // TI, SEQ // TJ, HEAD_DIM // TK


@task_kernel(external_path="aie1/adf/kernel_mm/aie_fp32_v0", para = [TI, TJ, TK])
def kernel_gemm(TileA: float32[TB, TI, TK], 
                TileB: float32[TB, TK, TJ], 
                TileC: float32[TB, TI, TJ]):
    for tb in range(0, TB):
      for i0 in range(0, TI):
          for j0 in range(0, TJ):
              TileC[tb, i0, j0] = float32(0)
              for k0 in range(0, TK):
                  TileC[tb, i0, j0] += TileA[tb, i0, k0] * TileB[tb, k0, j0]

@task_tile()
def bmm(A: float32[-1, -1, -1], B: float32[-1, -1, -1], 
        C: float32[-1, -1, -1], GRID_B, GRID_I, GRID_J, GRID_K):
    
    for b in range(GRID_B):
      for i in range(GRID_I):
          for j in range(GRID_J):
              for k in range(GRID_K):
                  # Compute tile slices for multiple dimensions
                  tb = aries.arange(b*TB, (b+1)*TB)  # B tile range
                  ti = aries.arange(i*TI, (i+1)*TI)  # I tile range
                  tj = aries.arange(j*TJ, (j+1)*TJ)  # J tile range
                  tk = aries.arange(k*TK, (k+1)*TK)  # K tile range

                  L1_A = aries.buffer((TB, TI, TK), "float32")
                  L1_B = aries.buffer((TB, TK, TJ), "float32")
                  L1_C = aries.buffer((TB, TI, TJ), "float32")

                  L1_A = aries.load(A, (tb, ti, tk))
                  L1_B = aries.load(B, (tb, tk, tj))
                  kernel_gemm(L1_A, L1_B, L1_C)
                  aries.accstore(L1_C, C, (tb, ti, tj))

@task_top()
def top(Q: float32[HEADS, SEQ, HEAD_DIM], K_Trans: float32[HEADS, HEAD_DIM, SEQ], 
        QK: float32[HEADS, SEQ, SEQ], SoftM: float32[HEADS, SEQ, SEQ], 
        V: float32[HEADS, SEQ, HEAD_DIM], OUT: float32[HEADS, SEQ, HEAD_DIM]):
    # Cast the Arrays to dynamic shape
    cast_Q = aries.cast(Q, (-1, -1, -1)) # This is for lowering
    cast_K_Trans = aries.cast(K_Trans, (-1, -1, -1))
    cast_QK = aries.cast(QK, (-1, -1, -1))
    cast_SoftM = aries.cast(SoftM, (-1, -1, -1))
    cast_V = aries.cast(V, (-1, -1, -1))
    cast_OUT = aries.cast(OUT, (-1, -1, -1))
    
    bmm_task = bmm(cast_Q, cast_K_Trans, cast_QK, GRID_B0, GRID_I0, GRID_J0, GRID_K0)
    softmax(cast_QK, cast_SoftM)
    bmm_task = bmm(cast_SoftM, cast_V, cast_OUT, GRID_B1, GRID_I1, GRID_J1, GRID_K1)
    return bmm_task

all_code = sys.modules[__name__]

# Initialize the buffers
np.random.seed(0)
Q = np.random.rand(HEADS, SEQ, HEAD_DIM).astype(np.float32)
K_Trans = np.random.rand(HEADS, HEAD_DIM, SEQ).astype(np.float32)
QK = np.zeros((HEADS, SEQ, SEQ)).astype(np.float32)
SoftM = np.zeros((HEADS, SEQ, SEQ)).astype(np.float32)
V = np.random.rand(HEADS, SEQ, HEAD_DIM).astype(np.float32)
OUT = np.zeros((HEADS, SEQ, HEAD_DIM)).astype(np.float32)

# Execute on CPU
gemm_task = top(Q, K_Trans, QK, SoftM, V, OUT)

# Golden file generation
golden_QK = np.matmul(Q, K_Trans)
golden_SoftM = softmax_sw(golden_QK)
golden_OUT = np.matmul(golden_SoftM, V)

# Compare the program with golden file
print("QK matches golden reference:", np.allclose(QK, golden_QK))
print("Softmax output matches golden reference:", np.allclose(SoftM, golden_SoftM))
print("Final output matches golden reference:", np.allclose(OUT, golden_OUT))

# # Generate files for on-board test
aries.gen_sim([Q, K_Trans, golden_QK, golden_SoftM, V, golden_OUT])

# Specify primitives to optimize hardware design
sch = Schedule(gemm_task)
sch.to("VCK190")
sch.axiWidth(gemm_task, 32)

# Set the project dir and template dir
prj_dir= cur_dir + '/project_mha'
temp_dir= aries_path + '/templates'
# Generate Initial MLIR and ARIES Opts
sch.build(all_code, prj_dir, temp_dir)
sch.compile(aries_path, prj_dir)