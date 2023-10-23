import torch

# We need to set cache size very large to avoid benchmarking eager mode as compiled
torch._dynamo.config.cache_size_limit = 100000


def transform(img, osize):
    img = torch.nn.functional.interpolate(img, size=osize, mode="bilinear", antialias=True)

    return img


def main():

    for osize in [(271, 272), (567, 678)]:
        for bs in [1, 4]:
            for dtype in [torch.uint8, torch.float32]:

                for memory_format in [torch.contiguous_format, torch.channels_last]:

                    torch.manual_seed(12)
                    for num_threads in [1,]:
                        torch.set_num_threads(num_threads)

                        for device in ["cpu", "cuda"]:

                            print(f"- {osize} {bs} {device} {memory_format} {dtype}")

                            if device == "cuda" and dtype == torch.uint8:
                                continue

                            x = torch.randint(0, 256, size=(bs, 3, 345, 456), dtype=dtype, device=device)
                            x = x.contiguous(memory_format=memory_format)

                            c_transform = torch.compile(transform)
                            output = c_transform(x, osize)
                            expected = transform(x, osize)

                            assert output.stride() == expected.stride(), (output.stride(), expected.stride())

                            if x.is_floating_point():
                                torch.testing.assert_close(output, expected, atol=5e-3, rtol=0.0)
                            else:
                                torch.testing.assert_close(output.float(), expected.float(), atol=1.0, rtol=0.0)

if __name__ == "__main__":

    print("")
    print(f"Torch version: {torch.__version__}")
    print(f"Torch config: {torch.__config__.show()}")
    print("")

    main()