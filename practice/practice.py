import os
import sys
cur_dir = os.getcwd()
aries_path = cur_dir + "/.."
sys.path.append(aries_path)
from frontend import *

# Vector Add: C[i0] += A[i0] * B[i0]
I = 256
TI = 32
grid = (I // TI, ) # grid must be a tuple


@task_kernel()
def kernel_add(TileA: float32[TI], TileB: float32[TI], TileC: float32[TI]):
    for i0 in range(0, TI):
        TileC[i0] = TileA[i0] + TileB[i0]

@task_tile()
def vadd(A: float32[I], B: float32[I], C: float32[I], **kwargs):
    i = aries.tile_ranks(**kwargs)
    
    L1_A = aries.buffer((TI, ), "float32")
    L1_B = aries.buffer((TI, ), "float32")
    L1_C = aries.buffer((TI, ), "float32")

    # Compute tile slices for multiple dimensions
    ti = aries.arange(i*TI, (i+1)*TI)  # I tile range

    # Move data between L3 and L1
    L1_A = aries.load(A, (ti, ))
    L1_B = aries.load(B, (ti, ))
    kernel_add(L1_A, L1_B, L1_C)
    aries.store(L1_C, C, (ti, ))

@task_top()
def top(A: float32[I], B: float32[I], C: float32[I]):
    gemm_vadd = vadd[grid](A, B, C)
    return gemm_vadd

all_code = sys.modules[__name__]

# Initialize the buffers
np.random.seed(0)
A = np.random.rand(I).astype(np.float32)
B = np.random.rand(I).astype(np.float32)
C = np.zeros((I)).astype(np.float32)

# Execute on CPU
vadd_task = top(A, B, C)
golden_C = np.add(A, B)
print("ARIES vadd output matches golden reference:", np.allclose(C, golden_C))

# Generate files for on-board test
aries.gen_sim([A, B, golden_C])

# Apply schedulings
sch = Schedule(vadd_task)
sch.to("VCK190")
sch.ioWidth(vadd_task, 128)
sch.axiWidth(vadd_task, 512)
sch.parallel(vadd_task, [4, ])

# Set the project dir and template dir
prj_dir= cur_dir + '/project_vadd'
temp_dir= aries_path + '/templates'
# Generate Initial MLIR and ARIES Opts
sch.build(all_code, prj_dir, temp_dir)

print("ARIES Initial IR:\n")

sch.print_mlir(all_code)

# ## By setting the versal image path, users can run compilation in Jupyter cell
# versal_image_path = '/tools/Xilinx/Vitis/2023.2/base_platforms/xilinx_vck190_base_202320_1/hw_emu'
# sch.compile(aries_path, prj_dir, versal_image_path, "hw_emu")

