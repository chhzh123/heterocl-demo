import heterocl as hcl
import numpy as np

def test1():
    def pack(A):
        return hcl.pack(A, axis=1, dtype=hcl.UInt(4))

    hcl.init()
    a = hcl.placeholder((2,4), dtype=hcl.UInt(1), name="A")
    s = hcl.create_schedule([a], pack)
    f = hcl.build(s)

    np_array = np.array([[0,1,1,1],[1,0,0,1]])
    in_array = hcl.asarray(np_array, dtype=hcl.UInt(1))
    out_array = hcl.asarray(np.zeros((2,1)), dtype=hcl.UInt(4))
    f(in_array, out_array)
    print("Input: {}".format(in_array.asnumpy()))
    print("Output: {}".format(out_array.asnumpy()))
    print("Numpy out: {}".format(np.packbits(np_array,axis=1,bitorder="little")))

def test2():
    def pack(A):
        return hcl.pack(A, axis=1, dtype=hcl.UInt(2))

    hcl.init()
    a = hcl.placeholder((2,4), dtype=hcl.UInt(1), name="A")
    s = hcl.create_schedule([a], pack)
    f = hcl.build(s)

    in_array = hcl.asarray(np.array([[0,1,1,1],[1,0,0,1]]), dtype=hcl.UInt(1))
    out_array = hcl.asarray(np.zeros((2,2)), dtype=hcl.UInt(2))
    f(in_array, out_array)
    print("Input: {}".format(in_array.asnumpy()))
    print("Output: {}".format(out_array.asnumpy()))

def test3():
    def pack(A):
        pack = hcl.pack(A, axis=1, factor=8, dtype=hcl.UInt(8))
        return pack

    hcl.init()
    a = hcl.placeholder((2,32), dtype=hcl.UInt(1), name="A")
    s = hcl.create_schedule([a], pack)
    f = hcl.build(s)

    np_array = np.random.randint(0,2,(2,32)).astype(np.bool)
    np_out = np.packbits(np_array,axis=1,bitorder="little")
    in_array = hcl.asarray(np_array, dtype=hcl.UInt(1))
    out_array = hcl.asarray(np.zeros((2,4)), dtype=hcl.UInt(8))
    f(in_array, out_array)
    print("Input: {}".format(in_array.asnumpy()))
    print("Output: {}".format(out_array.asnumpy()))
    print("Numpy out: {}".format(np_out))

def test4():
    def pack(A):
        pack = hcl.pack(A, axis=1, factor=32, dtype=hcl.UInt(32))
        return pack

    hcl.init()
    a = hcl.placeholder((2,32), dtype=hcl.UInt(1), name="A")
    s = hcl.create_schedule([a], pack)
    f = hcl.build(s)

    np_array = np.random.randint(0,2,(2,32)).astype(np.bool)
    np_out = np.packbits(np_array,axis=1,bitorder="little").view(np.uint32)
    in_array = hcl.asarray(np_array, dtype=hcl.UInt(1))
    out_array = hcl.asarray(np.zeros((2,1)), dtype=hcl.UInt(32))
    f(in_array, out_array)
    print("Input: {}".format(in_array.asnumpy()))
    print("Output: {}".format(out_array.asnumpy()))
    print("Numpy out: {}".format(np_out))

def test5():
    def pack(A):
        pack = hcl.pack(A, axis=1, factor=32, dtype=hcl.UInt(32))
        return pack

    hcl.init()
    a = hcl.placeholder((10,256), dtype=hcl.UInt(1), name="A")
    s = hcl.create_schedule([a], pack)
    f = hcl.build(s)

    np_array = np.random.randint(0,2,(10,256)).astype(np.bool)
    np_out = np.packbits(np_array,axis=1,bitorder="little").view(np.uint32)
    in_array = hcl.asarray(np_array, dtype=hcl.UInt(1))
    out_array = hcl.asarray(np.zeros((10,8)), dtype=hcl.UInt(32))
    f(in_array, out_array)
    print("Input: {}".format(in_array.asnumpy()))
    print("Output: {}".format(out_array.asnumpy()))
    print("Numpy out: {}".format(np_out))

def test6(): # endian
    def pack(A):
        return hcl.pack(A, axis=1, factor=8, dtype=hcl.UInt(8), bitorder="big")

    hcl.init()
    a = hcl.placeholder((2,8), dtype=hcl.UInt(1), name="A")
    s = hcl.create_schedule([a], pack)
    f = hcl.build(s)

    np_array = np.array([[0,1,1,1,0,0,0,0],[1,0,0,1,1,0,0,1]])
    in_array = hcl.asarray(np_array, dtype=hcl.UInt(1))
    out_array = hcl.asarray(np.zeros((2,1)), dtype=hcl.UInt(8))
    f(in_array, out_array)
    print("Input: {}".format(in_array.asnumpy()))
    print("Output: {}".format(out_array.asnumpy()))
    print("Numpy out: {}".format(np.packbits(np_array,axis=1,bitorder="big")))

def test7(): # endian
    def pack(A):
        return hcl.pack(A, axis=1, factor=16, dtype=hcl.UInt(8), bitorder="big")

    hcl.init()
    a = hcl.placeholder((2,16), dtype=hcl.UInt(1), name="A")
    s = hcl.create_schedule([a], pack)
    f = hcl.build(s)

    np_array = np.array([[0,1,1,1,0,0,0,0,1,0,0,0,0,0,0,1],[1,0,0,1,1,0,0,1,1,0,1,0,0,1,1,0]])
    np_out = np.packbits(np_array,axis=1,bitorder="big")[:,::-1]
    in_array = hcl.asarray(np_array, dtype=hcl.UInt(1))
    out_array = hcl.asarray(np.zeros((2,1)), dtype=hcl.UInt(16))
    f(in_array, out_array)
    print("Input: {}".format(in_array.asnumpy()))
    print("Output: {}".format(out_array.asnumpy()))
    print("Numpy out: {}".format(np_out.copy().view(np.uint16)))

def test8(): # endian
    def pack(A):
        pack = hcl.pack(A, axis=1, factor=32, dtype=hcl.UInt(32), bitorder="big")
        return pack

    hcl.init()
    a = hcl.placeholder((10,256), dtype=hcl.UInt(1), name="A")
    s = hcl.create_schedule([a], pack)
    f = hcl.build(s)

    np_array = np.random.randint(0,2,(10,256)).astype(np.bool)
    np_out = np.packbits(np_array,axis=1,bitorder="big").view(np.uint32)
    in_array = hcl.asarray(np_array, dtype=hcl.UInt(1))
    out_array = hcl.asarray(np.zeros((10,8)), dtype=hcl.UInt(32))
    f(in_array, out_array)
    print("Input: {}".format(in_array.asnumpy()))
    print("Output: {}".format(out_array.asnumpy()))
    print("Numpy out: {}".format(np_out))

def test9():
    def pack(A):
        pack = hcl.pack(A, axis=0, factor=8, dtype=hcl.UInt(8))
        return pack

    hcl.init()
    a = hcl.placeholder((32,4,4), dtype=hcl.UInt(1), name="A")
    s = hcl.create_schedule([a], pack)
    f = hcl.build(s)

    np_array = np.random.randint(0,2,(32,4,4)).astype(np.bool)
    np_out = np.packbits(np_array,axis=0,bitorder="little")
    in_array = hcl.asarray(np_array, dtype=hcl.UInt(1))
    out_array = hcl.asarray(np.zeros((4,4,4)), dtype=hcl.UInt(8))
    f(in_array, out_array)
    # print("Input: {}".format(in_array.asnumpy()))
    print("Output: {}".format(out_array.asnumpy()))
    print("Numpy out: {}".format(np_out))

if __name__ == '__main__':
    # test1()
    # test2()
    # test3()
    # test4()
    # test5()
    # test6()
    # test7()
    # test8()
    test9()