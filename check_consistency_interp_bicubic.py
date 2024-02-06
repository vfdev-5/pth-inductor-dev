import torch

# We need to set cache size very large to avoid benchmarking eager mode as compiled
torch._dynamo.config.cache_size_limit = 100000


def transform(img, osize, align_corners):
    img = torch.nn.functional.interpolate(img, size=osize, mode="bicubic", antialias=False, align_corners=align_corners)
    return img


compile_kwargs = {
    # "backend": "eager",

    "fullgraph": True,
    # "dynamic": True,
    "dynamic": False,
}


def main():

    for osize in [(224, 224), (800, 700)]:
        for bs in [1, 4]:
            for align_corners in [False, True]:
                # for dtype in [torch.uint8, torch.float32]:
                for dtype in [torch.float32, ]:

                    for memory_format in [torch.contiguous_format, torch.channels_last]:

                        for num_threads in [1,]:
                            torch.set_num_threads(num_threads)

                            for device in ["cpu", "cuda"]:

                                print(f"- {osize} {bs} {device} {memory_format} {dtype} {align_corners}")

                                if device == "cuda" and dtype == torch.uint8:
                                    continue

                                torch.manual_seed(12)
                                x = torch.randint(0, 256, size=(bs, 3, 500, 400), dtype=dtype, device=device)
                                x = x.contiguous(memory_format=memory_format)

                                c_transform = torch.compile(transform, **compile_kwargs)
                                output = c_transform(x, osize, align_corners=align_corners)
                                expected = transform(x, osize, align_corners=align_corners)

                                if device != "cuda":
                                    assert output.stride() == expected.stride(), (output.stride(), expected.stride())

                                if x.is_floating_point():
                                    if device == "cpu":
                                        torch.testing.assert_close(output, expected, atol=5e-3, rtol=0.0)
                                    else:
                                        torch.testing.assert_close(output, expected, atol=5e-2, rtol=0.0)
                                else:
                                    torch.testing.assert_close(output.float(), expected.float(), atol=1.0, rtol=0.0)


if __name__ == "__main__":

    print("")
    print(f"Torch version: {torch.__version__}")
    print(f"Torch config: {torch.__config__.show()}")
    print("")

    main()