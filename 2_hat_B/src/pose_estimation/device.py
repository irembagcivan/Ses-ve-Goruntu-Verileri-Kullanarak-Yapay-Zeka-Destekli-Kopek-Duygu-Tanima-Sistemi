"""Lazy device selection. Explicit CPU fallback is always available."""
def select_device(requested="auto", torch_module=None):
    if requested not in {"auto", "cpu", "mps"}:
        raise ValueError("Device must be auto, cpu or mps")
    if requested == "cpu":
        return "cpu"
    if torch_module is None:
        import torch as torch_module
    available = torch_module.backends.mps.is_available()
    if requested == "mps" and not available:
        raise RuntimeError("MPS unavailable; rerun with --device cpu")
    return "mps" if available else "cpu"


def inference_error(exc, device):
    hint = "; rerun with --device cpu (MPS operation support varies)" if device == "mps" else ""
    return RuntimeError(f"Inference failed on {device}: {exc}{hint}")
