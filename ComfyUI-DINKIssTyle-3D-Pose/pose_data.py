"""Single-person COCO_18 + 21-point hands. Internal coordinates are normalized."""
import copy
import math

POSE_FIELDS = {"pose_keypoints_2d": 18, "hand_left_keypoints_2d": 21, "hand_right_keypoints_2d": 21, "face_keypoints_2d": 0}


def validate_pose(document, width=None, height=None):
    required={"version","canvas_width","canvas_height","people"}
    if not isinstance(document, dict) or not required<=set(document) or set(document)-required-{"floor_grid"}:
        raise ValueError("Invalid OpenPose document fields")
    if type(document["version"]) not in (float,int) or document["version"] != 1.3:
        raise ValueError("Invalid OpenPose document version")
    for name, expected in (("canvas_width",width),("canvas_height",height)):
        n=document[name]
        if type(n) is not int or not 64<=n<=4096 or (expected is not None and n!=expected):
            raise ValueError("OpenPose canvas does not match the scene")
    people=document["people"]
    if not isinstance(people,list) or len(people)!=1 or not isinstance(people[0],dict) or set(people[0])!=set(POSE_FIELDS):
        raise ValueError("OpenPose requires exactly one mannequin")
    for field,count in POSE_FIELDS.items():
        points=people[0][field]
        if not isinstance(points,list) or len(points)!=count*3:
            raise ValueError(f"Invalid OpenPose point count: {field}")
        for i in range(0,len(points),3):
            x,y,c=points[i:i+3]
            if not all(type(n) in (float,int) and math.isfinite(n) and 0<=n<=1 for n in (x,y,c)):
                raise ValueError("OpenPose points must be finite normalized coordinates")
            if c==0 and (x!=0 or y!=0):
                raise ValueError("Missing OpenPose points must use [0, 0, 0]")
    if "floor_grid" in document:
        lines=document["floor_grid"]
        if not isinstance(lines,list) or len(lines)>256:
            raise ValueError("Invalid projected floor grid")
        for line in lines:
            if not isinstance(line,list) or len(line)!=5 or not all(type(n) in (float,int) and math.isfinite(n) and 0<=n<=1 for n in line) or line[4] not in (0,1):
                raise ValueError("Invalid projected floor line")
    return document


def select_pose(document, include_body=True, include_hands=True, include_head=True):
    selected=copy.deepcopy(validate_pose(document))
    selected.pop("floor_grid",None)  # Keep standard exported keypoints clean.
    person=selected["people"][0]
    if not include_body:
        person["pose_keypoints_2d"]=[0]*54
    elif not include_head:
        for index in (0,14,15,16,17):
            person["pose_keypoints_2d"][index*3:index*3+3]=[0,0,0]
    if not include_hands:
        person["hand_left_keypoints_2d"]=[0]*63
        person["hand_right_keypoints_2d"]=[0]*63
    return selected


def pixel_document(document):
    result=copy.deepcopy(validate_pose(document))
    result.pop("floor_grid",None)
    w,h=result["canvas_width"],result["canvas_height"]
    for points in result["people"][0].values():
        for i in range(0,len(points),3):
            points[i]=round(points[i]*w,4)
            points[i+1]=round(points[i+1]*h,4)
    return result
