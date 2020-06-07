import heterocl as hcl
from itertools import permutations
import os, sys
import numpy as np
import heterocl.report as report

tcl = None

def test_vivado_hls():
    if os.system("which vivado_hls >> /dev/null") != 0:
        return 

    def test_hls(target_mode):
        hcl.init()
        A = hcl.placeholder((10, 32), "A")
        def kernel(A):
            B = hcl.compute(A.shape, lambda *args : A[args] + 1, "B")
            C = hcl.compute(A.shape, lambda *args : B[args] + 1, "C")
            D = hcl.compute(A.shape, lambda *args : C[args] * 2, "D")
            return D
        
        target = hcl.platform.zc706
        # target = hcl.platform.llvm
        s = hcl.create_schedule([A], kernel)
        s.to(kernel.B, target.xcel)
        s.to(kernel.C, target.host)
        if target_mode == "custom":
            tcl = open("run-test.tcl","r").read()
            target.config(compile="vivado_hls", script=tcl)
        else:
            tcl = None
            target.config(compile="vivado_hls", mode=target_mode)
        f = hcl.build(s, target)

        if target_mode == "debug":
            print(f)
            return

        np_A = np.random.randint(10, size=(10,32))
        np_B = np.zeros((10,32))

        hcl_A = hcl.asarray(np_A)
        hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))
        f(hcl_A, hcl_B)
        ret_B = hcl_B.asnumpy()

        if "csyn" in target_mode and tcl == None:
            report = f.report()
            assert "ReportVersion" in report
        elif "csim" in target_mode:
            np.testing.assert_array_equal(ret_B, (np_A + 2) * 2)

    test_hls("csim")
    # test_hls("csyn")
    # test_hls("csim|csyn")
    # test_hls("cosim")
    # test_hls("impl")
    # test_hls("custom")
    # test_hls("debug")

def test_debug_mode():

    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    def kernel(A):
        B = hcl.compute(A.shape, lambda *args : A[args] + 1, "B")
        C = hcl.compute(A.shape, lambda *args : B[args] + 1, "C")
        D = hcl.compute(A.shape, lambda *args : C[args] * 2, "D")
        return D

    def test_sdaccel_debug():
        target = hcl.platform.aws_f1
        s = hcl.create_schedule([A], kernel)
        s.to(kernel.B, target.xcel)
        s.to(kernel.C, target.host)
        target.config(compile="sdaccel", mode="debug", backend="vhls")
        code = hcl.build(s, target)
        print(code)
        assert "cl::Kernel kernel(program, \"test\", &err)" in code

    def test_vhls_debug():
        target = hcl.platform.zc706
        s = hcl.create_schedule([A], kernel)
        s.to(kernel.B, target.xcel)
        s.to(kernel.C, target.host)
        target.config(compile="vivado_hls", mode="debug")
        code = hcl.build(s, target)
        print(code)
        assert "test(hls::stream<ap_int<32> >& B_channel, hls::stream<ap_int<32> >& C_channel)" in code

    test_sdaccel_debug()
    test_vhls_debug()

def test_csyn():
    # 1. Declare computation
    A = hcl.placeholder((10, 32), "A")
    def kernel(A):
        B = hcl.compute(A.shape, lambda *args : A[args] + 1, "B")
        C = hcl.compute(B.shape, lambda *args : B[args] + 1, "C")
        return C

    # 2. Create schedule
    s = hcl.create_schedule([A], kernel)

    # 3. Specify the target platform and mode
    target = hcl.platform.zc706
    target.config(compile="vivado_hls", mode="csyn")

    # 4. Data movement
    s.to(kernel.B, target.xcel)
    s.to(kernel.C, target.host)

    # 5. Build the kernel
    #    (A misleading interface here, no kernel code is generated.
    #     Only the template Tcl file is copied to the current folder.)
    f = hcl.build(s, target)

    # 6. Create required arrays
    np_A = np.random.randint(10, size=(10,32))
    np_B = np.zeros((10,32))
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    # 7. Generate kernel code and do synthesis
    f(hcl_A, hcl_B)

if __name__ == '__main__':
    test_vivado_hls()
    # report.parse_xml("project",True)
    # test_debug_mode()
    # test_csyn()