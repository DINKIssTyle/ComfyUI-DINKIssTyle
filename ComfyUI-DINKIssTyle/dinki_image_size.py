from comfy_execution.graph_utils import ExecutionBlocker


class DINKI_GetImageSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "batch_size")
    FUNCTION = "get_size"
    CATEGORY = "DINKIssTyle/Image"
    DESCRIPTION = "Returns image dimensions. Without an image, blocks downstream execution."

    def get_size(self, image=None):
        if image is None or isinstance(image, ExecutionBlocker):
            return tuple(ExecutionBlocker(None) for _ in self.RETURN_TYPES)
        batch_size, height, width = image.shape[:3]
        if batch_size == 0 or height == 0 or width == 0:
            return tuple(ExecutionBlocker(None) for _ in self.RETURN_TYPES)
        return (int(width), int(height), int(batch_size))
