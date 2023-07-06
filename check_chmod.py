import os
import tempfile


def _get_umask():
    # os.umask sets-and-gets, so we need to restore the value
    mask = os.umask(0)
    os.umask(mask)
    return mask


def write_atomic(path: str, source_code: str, mode=0o666):
    # use a temp file for thread safety
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path))
    with os.fdopen(fd, "w") as f:
        f.write(source_code)

    # set file permissions
    masked_mode = mode & ~_get_umask()
    # masked_mode = mode
    os.chmod(tmp_path, masked_mode)

    # rename atomically
    os.rename(tmp_path, path)


write_atomic("/tmp/pth/cpu_inductor/to-remove", "abc")
