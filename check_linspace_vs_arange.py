import torch

# to_int = False
to_int = True

for isize, osize in [
    (500, 256), (4567, 123), (123, 3456), (456, 234),
]:
    for offset in [0.0, 0.5]:
        scale = isize / osize
        print("-", isize, osize, offset, scale)

        output_indices = torch.arange(osize)
        expected_input_indices = ((output_indices + offset) * scale)
        input_indices = torch.linspace(offset * scale, (osize - 1 + offset) * scale, steps=osize)

        if to_int:
            expected_input_indices = expected_input_indices.to(torch.int64)
            input_indices = input_indices.to(torch.int64)

        try:
            torch.testing.assert_close(expected_input_indices, input_indices)
        except AssertionError:
            print("expected: ", expected_input_indices)
            print("output  : ", input_indices)
            m = (expected_input_indices - input_indices).abs()
            print("expected - output  : ", m)
            print("expected[m] : ", expected_input_indices[m > 0])
            print("output[m]   : ", input_indices[m > 0])
