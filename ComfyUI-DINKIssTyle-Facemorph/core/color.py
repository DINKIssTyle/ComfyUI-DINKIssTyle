"""Color statistics shared by preview and full-resolution rendering."""
import numpy as np


def validate(settings):
    if not isinstance(settings, dict) or set(settings) - {'color_match', 'edge_feather'}:
        raise ValueError('Invalid blend settings.')
    for key in ('color_match', 'edge_feather'):
        value = settings.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 100:
            raise ValueError('Blend strengths must be between 0 and 100.')


def corrections(images, assets, base, masks=None):
    import cv2
    from .topology import OVAL

    def statistics(image, index):
        h, w = image.shape[:2]
        face = np.asarray(assets[index]['points'])[:468] * [w-1, h-1]
        lo = np.maximum(np.floor(face.min(0)).astype(int), 0)
        hi = np.minimum(np.ceil(face.max(0)).astype(int), [w-1, h-1])
        if (hi <= lo).any():
            return None
        xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 128), np.linspace(lo[1], hi[1], 128))
        pixels = cv2.remap(image, xx.astype(np.float32), yy.astype(np.float32), cv2.INTER_LINEAR).astype(np.float32)/255
        # Ignore background, eyes, eyebrows and mouth, even without hair protection.
        oval = np.zeros((128, 128), np.uint8)
        polygon = np.round((face[OVAL]-lo)/(hi-lo)*127).astype(np.int32)
        cv2.fillPoly(oval, [polygon], 1)
        from .topology import CURVES
        for curve in CURVES[1:7]:
            polygon = np.round((face[curve]-lo)/(hi-lo)*127).astype(np.int32)
            cv2.fillPoly(oval, [polygon], 0)
        valid = oval > 0
        if masks is not None:
            mh, mw = masks[index].shape[:2]
            skin = cv2.remap(masks[index][:,:,0], (xx*(mw-1)/max(w-1,1)).astype(np.float32),
                             (yy*(mh-1)/max(h-1,1)).astype(np.float32), cv2.INTER_LINEAR)
            valid &= skin > .8
        samples = pixels[valid]
        if len(samples) < 32:
            return None
        # Trim exposure outliers before measuring channel statistics.
        low, high = np.percentile(samples, [5, 95], axis=0)
        samples = np.clip(samples, low, high)
        return samples.mean(0), samples.std(0)

    stats = [statistics(image, i) for i, image in enumerate(images)]
    result = []
    for i, source in enumerate(stats):
        gain, offset = np.ones(3), np.zeros(3)
        if i != base and source is not None and stats[base] is not None:
            mean, std = source
            ref_mean, ref_std = stats[base]
            gain = np.clip(ref_std/np.maximum(std, .02), .5, 2)
            offset = ref_mean-mean*gain
        result.append({'gain':gain.tolist(), 'offset':offset.tolist()})
    return result
