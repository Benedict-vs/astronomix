"""No-op autocvd for jobs inside gpuq.sh (CUDA_VISIBLE_DEVICES is already set) and CPU-only runs."""


def autocvd(*args, **kwargs):
    """Do nothing: the GPU (or the CPU) is already chosen by the caller."""
    return None
