"""Recover semantic contour edges by flipping a sparse Delaunay triangulation.

No new vertices are added: all photos retain identical correspondence indices.
Intersecting manual contours are rejected instead of creating ambiguous mappings.
"""
import numpy as np
from scipy.spatial import Delaunay
from .topology import CURVES


def cross(a,b,c):
    x,y=b-a,c-a
    return x[0]*y[1]-x[1]*y[0]


def intersects(a,b,c,d):
    return cross(a,b,c)*cross(a,b,d)<-1e-18 and cross(c,d,a)*cross(c,d,b)<-1e-18


def triangulate(points, warnings=None):
    curves=CURVES[:7]
    indices=set(sum(CURVES,[])) | set(range(468,512))
    indices=np.array(sorted(indices))
    triangles=indices[Delaunay(points[indices]).simplices].tolist()
    segments=[]
    for curve in curves:
        pairs=[(curve[i],curve[(i+1)%len(curve)]) for i in range(len(curve))]
        crossed=any(len({a,b,c,d})==4 and intersects(points[a],points[b],points[c],points[d])
                    for a,b in pairs for c,d in pairs)
        if crossed:
            if curve not in (CURVES[1],CURVES[2],CURVES[6]):
                raise ValueError('A facial contour crosses itself. Correct its correspondence points.')
            if warnings is not None:
                warnings.append('A closed eye/mouth contour overlaps itself; its edge constraints were relaxed. Inspect or correct its points.')
            continue
        segments.extend([tuple(sorted((curve[i],curve[(i+1)%len(curve)]))) for i in range(len(curve))])
    protected=set()
    for start,end in segments:
        for _ in range(len(triangles)*2):
            edges={}
            for i,tri in enumerate(triangles):
                for j in range(3):
                    edge=tuple(sorted((tri[j],tri[(j+1)%3])))
                    edges.setdefault(edge,[]).append(i)
            if (start,end) in edges:
                protected.add((start,end))
                break
            progress=False
            for (u,v),neighbors in edges.items():
                if len(neighbors)!=2 or start in (u,v) or end in (u,v):
                    continue
                if not intersects(points[start],points[end],points[u],points[v]):
                    continue
                if (u,v) in protected:
                    raise ValueError('Facial contours intersect. Correct their correspondence points.')
                i,j=neighbors
                a=next(x for x in triangles[i] if x not in (u,v))
                b=next(x for x in triangles[j] if x not in (u,v))
                if not intersects(points[u],points[v],points[a],points[b]):
                    continue
                if intersects(points[start],points[end],points[a],points[b]):
                    continue
                if tuple(sorted((a,b))) in edges:
                    continue
                triangles[i],triangles[j]=[a,b,u],[b,a,v]
                progress=True
                break
            if not progress:
                raise ValueError('Cannot preserve facial contours. Correct overlapping correspondence points.')
        else:
            raise ValueError('Could not build contour mesh. Reset the edited points.')
    return np.asarray(triangles,dtype=int)
