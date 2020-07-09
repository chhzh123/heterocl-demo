import heterocl as hcl
import numpy as np

def test_reorder():
    hcl.init()
    a = hcl.placeholder((10, 20, 30, 40), name="a")
    b = hcl.placeholder((10, 20, 30, 40), name="b")
    c = hcl.compute(a.shape, lambda i, j, k, l: a[i, j, k, l] + b[i, j, k, l], name="c")

    # axes are consecutive
    def test_case_1():
        s = hcl.create_schedule([a, b, c])
        s[c].reorder(c.axis[2], c.axis[1])
        ir = hcl.lower(s)
        assert "(i, 0, 10)" in str(ir.body.body)
        assert "(k, 0, 30)" in str(ir.body.body.body)
        assert "(j, 0, 20)" in str(ir.body.body.body.body)
        assert "(l, 0, 40)" in str(ir.body.body.body.body.body)

    # axes are not consecutive
    def test_case_2():
        s = hcl.create_schedule([a, b, c])
        s[c].reorder(c.axis[3], c.axis[0])
        ir = hcl.lower(s)
        assert "(l, 0, 40)" in str(ir.body.body)
        assert "(j, 0, 20)" in str(ir.body.body.body)
        assert "(k, 0, 30)" in str(ir.body.body.body.body)
        assert "(i, 0, 10)" in str(ir.body.body.body.body.body)

    test_case_1()
    test_case_2()

def test_reorder_num_axis():
    hcl.init()
    a = hcl.placeholder((10, 20, 30, 40), name="a")
    b = hcl.placeholder((10, 20, 30, 40), name="b")
    c = hcl.compute(a.shape, lambda i, j, k, l: a[i, j, k, l] + b[i, j, k, l], name="c")

    s = hcl.create_schedule([a, b, c])
    s[c].reorder(2, 1)
    ir = hcl.lower(s)
    assert "(i, 0, 10)" in str(ir.body.body)
    assert "(k, 0, 30)" in str(ir.body.body.body)
    assert "(j, 0, 20)" in str(ir.body.body.body.body)
    assert "(l, 0, 40)" in str(ir.body.body.body.body.body)

def test_split():
    hcl.init()
    a = hcl.placeholder((10, 20), name="a")
    b = hcl.placeholder((10, 20), name="b")
    c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j], name="c")

    # without if condition
    def test_transform_mode_1():
        s = hcl.create_schedule([a, b, c])
        s[c].split(c.axis[1], factor=4, mode="transform")
        ir = hcl.lower(s)
        assert "(i, 0, 10)" in str(ir.body.body)
        assert "(j.outer, 0, 5)" in str(ir.body.body.body)
        assert "(j.inner, 0, 4)" in str(ir.body.body.body.body)
        assert str(ir.body.body.body.body.body).startswith("c[")

    # with if condition
    def test_transform_mode_2():
        s = hcl.create_schedule([a, b, c])
        s[c].split(c.axis[1], factor=3, mode="transform")
        ir = hcl.lower(s)
        assert "(i, 0, 10)" in str(ir.body.body)
        assert "(j.outer, 0, 7)" in str(ir.body.body.body)
        assert "(j.inner, 0, 3)" in str(ir.body.body.body.body)
        assert str(ir.body.body.body.body.body).startswith(
            "if ((j.inner < (20 - (j.outer*3))))")

    def test_annotate_mode():
        split_factor = 3
        s = hcl.create_schedule([a, b, c])
        s[c].split(c.axis[1], factor=split_factor, mode="annotate")
        split_hint_str = "\"split_factor\"="+str(split_factor)
        ir = hcl.lower(s)
        assert split_hint_str in str(ir)

    test_transform_mode_1()
    test_transform_mode_2()
    test_annotate_mode()

def test_split_num_axis():
    hcl.init()
    a = hcl.placeholder((10, 20), name="a")
    b = hcl.placeholder((10, 20), name="b")
    c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j], name="c")

    s = hcl.create_schedule([a, b, c])
    s[c].split(1, factor=4, mode="transform")
    ir = hcl.lower(s)
    assert "(i, 0, 10)" in str(ir.body.body)
    assert "(j.outer, 0, 5)" in str(ir.body.body.body)
    assert "(j.inner, 0, 4)" in str(ir.body.body.body.body)
    assert str(ir.body.body.body.body.body).startswith("c[")

def test_split_reorder():
    hcl.init()
    a = hcl.placeholder((10, 20), name="a")
    b = hcl.placeholder((10, 20), name="b")
    c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j], name="c")

    def test_case_1():
        s = hcl.create_schedule([a, b, c])
        xo, xi = s[c].split(c.axis[0], factor=2, mode="transform")
        yo, yi = s[c].split(c.axis[1], factor=5, mode="transform")
        s[c].reorder(yo, xo, yi, xi)
        ir = hcl.lower(s)
        assert "(j.outer, 0, 4)" in str(ir.body.body)
        assert "(i.outer, 0, 5)" in str(ir.body.body.body)
        assert "(j.inner, 0, 5)" in str(ir.body.body.body.body)
        assert "(i.inner, 0, 2)" in str(ir.body.body.body.body.body)

    def test_case_2():
        s = hcl.create_schedule([a, b, c])
        xo, xi = s[c].split(c.axis[0], factor=3, mode="transform")
        yo, yi = s[c].split(c.axis[1], factor=3, mode="transform")
        s[c].reorder(yi, xi, yo, xo)
        ir = hcl.lower(s)
        assert "(j.inner, 0, 3)" in str(ir.body.body)
        assert "(i.inner, 0, 3)" in str(ir.body.body.body)
        assert "(j.outer, 0, 7)" in str(ir.body.body.body.body)
        assert "(i.outer, 0, 4)" in str(ir.body.body.body.body.body)
        assert str(ir.body.body.body.body.body.body).startswith(
            "if ((j.inner < (20 - (j.outer*3))))")
        assert str(ir.body.body.body.body.body.body.then_case).startswith(
            "if ((i.inner < (10 - (i.outer*3)))")

    test_case_1()
    test_case_2()

def test_split_reorder_num_axis():
    # note that this is not the recommanded way
    hcl.init()
    a = hcl.placeholder((10, 20), name="a")
    b = hcl.placeholder((10, 20), name="b")
    c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j], name="c")

    s = hcl.create_schedule([a, b, c])
    xo, xi = s[c].split(0, factor=2, mode="transform")
    yo, yi = s[c].split(2, factor=5, mode="transform")
    s[c].reorder(2, 0, 3, 1)
    ir = hcl.lower(s)
    assert "(j.outer, 0, 4)" in str(ir.body.body)
    assert "(i.outer, 0, 5)" in str(ir.body.body.body)
    assert "(j.inner, 0, 5)" in str(ir.body.body.body.body)
    assert "(i.inner, 0, 2)" in str(ir.body.body.body.body.body)

if __name__ == "__main__":
    test_reorder()
    test_reorder_num_axis()
    test_split()
    test_split_num_axis()
    test_split_reorder()
    test_split_reorder_num_axis()