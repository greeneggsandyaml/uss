from PIL import Image
import numpy as np
import torch
import torchvision

    
def apply_connected_components_(m: np.ndarray, threshold: float):
    """Return masks with small connected components removed"""

    # Get connected components
    component, num = measure_label(m, return_num=True, background=0)
    areas = np.zeros([num + 1])
    for comp in range(1, num + 1, 1):
        areas[comp] = np.sum(component == comp)

    # Get area of biggest connected component
    max_component = np.argmax(areas)
    max_component_area = areas[max_component]

    # Create new mask (in-place) with filtered connected components
    m *= 0
    for comp in range(1, num + 1, 1):
        area = areas[comp]
        if float(area) / max_component_area > threshold:
            m[component == comp] = True
    return m


def apply_connected_components_filter(mask: torch.Tensor, threshold: float):
    """Iterates over mask and applies connected components filter"""
    processed_mask = mask.numpy()
    for m in processed_mask:
        apply_connected_components_(m, threshold)
    processed_mask = torch.from_numpy(processed_mask).to(mask.device)
    return processed_mask


def to_image(t: torch.Tensor, is_mask: bool = False):
    t = t.cpu().detach()
    if len(t.shape) == 4:
        t = t[0]
    if is_mask:
        t = t.squeeze(dim=0)  # convert to 2-dimensional mask
        t = t.to(torch.uint8).numpy()
        return Image.fromarray(t)
    else:
        t = torch.clamp(t * 0.5 + 0.5, min=0, max=1)  # de-normalize
        return torchvision.transforms.ToPILImage()(t)  # convert to image
