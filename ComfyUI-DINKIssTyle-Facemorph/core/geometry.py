"""Pure geometry. Coordinates use normalized pixel centers, x/(width-1)."""
import numpy as np
import re
from .topology import COUNT, GROUPS, ALPHAS, REGIONS, OVAL, PAIRS, PARENTS


def upgrade_state(state):
    """Keep previously saved workflows usable after retiring the old vertex layout."""
    if not isinstance(state, dict) or not isinstance(state.get('assets'), list):
        return state
    for asset in state['assets']:
        if not isinstance(asset, dict):
            continue
        points = asset.get('points')
        if isinstance(points, list) and len(points) == 544:
            asset['points'] = points[:468] + points[484:520] + points[536:544]
        asset.pop('ears', None)
        regions = asset.get('regions', {})
        if isinstance(regions, dict):
            regions.pop('ear_l', None)
            regions.pop('ear_r', None)
        if isinstance(asset.get('warnings'), list):
            asset['warnings'] = [w for w in asset['warnings'] if not re.search(r'\bears?\b', str(w), re.IGNORECASE)]
    return state


def region_value(asset, region, key):
    overrides = asset.get('regions', {})
    parent = overrides.get(PARENTS.get(region), {})
    return overrides.get(region, {}).get(key, parent.get(key, asset.get(key, 1)))


def validate_state(state):
    if not isinstance(state, dict) or state.get('version') != 1:
        raise ValueError('Unsupported project. Open Face Morph Studio and add photos.')
    from .liquify import validate
    validate(state.get('liquify', {}))
    from .hair import validate as validate_hair
    validate_hair(state.get('hair', {}))
    from .color import validate as validate_blend
    validate_blend(state.get('blend', {}))
    upgrade_state(state)
    assets = state.get('assets', [])
    if not isinstance(assets,list) or not 1 <= len(assets) <= 10:
        raise ValueError('Add between 1 and 10 photos.')
    base = state.get('base', 0)
    if isinstance(base, bool) or not isinstance(base, int) or not 0 <= base < len(assets):
        raise ValueError('Invalid base photo.')
    for asset in assets:
        if not isinstance(asset,dict):
            raise ValueError('Invalid photo entry.')
        validate_hair({},asset.get('mask_edit',{}))
        from .crop import validate as validate_crop
        validate_crop(asset.get('crop',{}))
        pts = np.asarray(asset.get('points'), dtype=np.float64)
        if pts.shape != (COUNT, 2) or not np.isfinite(pts).all() or (abs(pts) > 4).any():
            raise ValueError('Invalid correspondence points.')
        for key in ('shape', 'feature'):
            value = asset.get(key, 1)
            if not isinstance(value, (int,float)) or not np.isfinite(value) or not 0 <= value <= 100:
                raise ValueError('Weights must be between 0 and 100.')
        overrides = asset.get('regions', {})
        if not isinstance(overrides, dict) or set(overrides) - (set(REGIONS) | set(PAIRS)):
            raise ValueError('Unknown region.')
        for values in overrides.values():
            if not isinstance(values, dict) or set(values) - {'shape', 'feature'}:
                raise ValueError('Invalid region weights.')
            for value in values.values():
                if not isinstance(value, (int,float)) or not np.isfinite(value) or not 0 <= value <= 100:
                    raise ValueError('Invalid region weight.')
    return assets, base


def similarity(source, reference):
    # Eye corners define roll and scale; fit in base pixel aspect to avoid distortion.
    ids = [33,133,362,263]
    a, b = source[ids], reference[ids]
    ac, bc = a.mean(0), b.mean(0)
    x, y = a-ac, b-bc
    u, s, vt = np.linalg.svd(x.T @ y)
    fix = np.eye(2)
    fix[-1,-1] = np.linalg.det(u @ vt)
    r = u @ fix @ vt
    scale = np.sum(s * np.diag(fix)) / max(np.sum(x*x), 1e-12)
    return (source-ac) @ r * scale + bc


def aligned_points(assets, base, sizes):
    bw, bh = sizes[base]
    aspect = np.array([bw-1, bh-1], dtype=float)
    reference = np.array(assets[base]['points']) * aspect
    result = []
    for i, a in enumerate(assets):
        sw, sh = sizes[i]
        src = np.array(a['points']) * [sw-1, sh-1]
        pts = reference.copy() if i == base else similarity(src, reference)
        pts /= aspect
        # Frame anchors stay on the output canvas. They only protect the background.
        pts[-8:] = np.array(assets[base]['points'])[-8:]
        result.append(pts)
    return np.asarray(result)


def weights(assets, base, key, visibility=None):
    raw = np.array([[region_value(a, REGIONS[g], key) for g in GROUPS]
                    for a in assets], dtype=float)
    if visibility is not None and key == 'shape':
        raw *= np.asarray(visibility)
    totals = raw.sum(0)
    raw[base,totals == 0] = 1
    return raw / np.maximum(raw.sum(0), 1e-12)


def plan(state, sizes, visibility=None):
    from scipy.spatial import QhullError
    from .triangulation import triangulate
    assets, base = validate_state(state)
    aligned = aligned_points(assets, base, sizes)
    sw, fw = weights(assets,base,'shape',visibility), weights(assets,base,'feature')
    target = np.sum(aligned * sw[:,:,None], axis=0)
    warnings = []
    neutral = weights([{**a,'shape':1,'regions':{}} for a in assets],base,'shape')
    reference = np.sum(aligned*neutral[:,:,None],axis=0)
    try:
        triangles = triangulate(reference, warnings)
    except QhullError as exc:
        raise ValueError('Correspondence mesh is degenerate. Reset edited points.') from exc
    # Never silently render folded or collapsed triangles.
    p, q = reference[triangles], target[triangles]
    def area(t):
        a,b = t[:,1]-t[:,0], t[:,2]-t[:,0]
        return a[:,0]*b[:,1]-a[:,1]*b[:,0]
    ref_area, dst_area = area(p), area(q)
    def folded(dst):
        return np.any(ref_area*area(dst[triangles]) < -1e-12) or np.any(abs(area(dst[triangles])) < 1e-12)
    if folded(target) and visibility is not None:
        # Occlusion is a confidence hint, never a reason to break an otherwise valid mesh.
        plain=weights(assets,base,'shape')
        original=np.sum(aligned*plain[:,:,None],axis=0)
        if not folded(original):
            desired=sw.copy()
            for amount in (.5,.25,.125,.0625,.03125,.015625,0):
                sw=plain*(1-amount)+desired*amount
                target=np.sum(aligned*sw[:,:,None],axis=0)
                if not folded(target):
                    warnings.append('Hair occlusion shape weighting was softened to keep the mesh valid.')
                    break
    if folded(target):
        raise ValueError('Mesh folds over. Reduce shape weights or correct correspondence points.')
    alpha = np.array(ALPHAS)
    return {'warnings':warnings, 'reference':reference.tolist(), 'aligned':aligned.tolist(), 'target':target.tolist(), 'triangles':triangles.tolist(),
            'shape_weights':sw.tolist(), 'feature_weights':fw.tolist(), 'alpha':alpha.tolist(),
            'groups':GROUPS, 'regions':REGIONS}


def add_support(points):
    pts = np.asarray(points, dtype=float).tolist()[:468]
    oval = np.asarray(pts)[OVAL]
    center = oval.mean(0)
    pts.extend((center + (oval-center)*1.12).tolist())
    pts.extend([[0,0],[.5,0],[1,0],[1,.5],[1,1],[.5,1],[0,1],[0,.5]])
    return pts


def protect_target(mesh, base_points, protection):
    """Constrain the shared mesh, so face and backing stay in correspondence.

    Pinning only the backing's inverse map exposes its original face underneath
    a differently deformed foreground. Use the cached base-hair distance field
    to limit the forward displacement of both layers together instead.
    """
    if protection is None:
        return mesh
    base = np.asarray(base_points, dtype=float)
    target = np.asarray(mesh['target'], dtype=float)
    h, w = protection.shape
    distance = np.full(len(base), max(h,w)*.25)
    for t in (0, .25, .5, .75, 1):
        uv = base+(target-base)*t
        xy = np.clip(uv,0,1)*[w-1,h-1]
        lo = np.floor(xy).astype(int)
        hi = np.minimum(lo+1,[w-1,h-1])
        f = xy-lo
        a = protection[lo[:,1],lo[:,0]]*(1-f[:,0])+protection[lo[:,1],hi[:,0]]*f[:,0]
        b = protection[hi[:,1],lo[:,0]]*(1-f[:,0])+protection[hi[:,1],hi[:,0]]*f[:,0]
        distance = np.minimum(distance,(a*(1-f[:,1])+b*f[:,1])*max(h,w)*.25)
    travel = np.linalg.norm((target-base)*[w-1,h-1],axis=-1)
    x = np.clip(distance/np.maximum(3,3*travel),0,1)
    desired = base+(target-base)*(x*x*(3-2*x))[:,None]
    triangles = np.asarray(mesh['triangles'])
    def areas(points):
        p=points[triangles];a,b=p[:,1]-p[:,0],p[:,2]-p[:,0]
        return a[:,0]*b[:,1]-a[:,1]*b[:,0]
    reference = areas(np.asarray(mesh['reference']))
    for amount in (1,.5,.25,.125,.0625,.03125,.015625,0):
        adjusted = target+(desired-target)*amount
        area = areas(adjusted)
        if np.all(reference*area >= -1e-12) and np.all(abs(area) >= 1e-12):
            return {**mesh,'target':adjusted.tolist(),'hair_pin_amount':amount}
    raise ValueError('The mesh folds over. Reduce shape weights or correct correspondence points.')
