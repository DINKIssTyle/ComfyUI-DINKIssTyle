"""Render the standard ControlNet COCO_18 color layout without pose detection."""
import colorsys
import json
import math
import numpy as np
import torch
from PIL import Image, ImageDraw
from .pose_data import validate_pose, select_pose, pixel_document
from .storage import load_pose

# Conventional OpenPose/ControlNet body palette and 17 visible COCO_18 bones.
COLORS=((255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),(85,255,0),(0,255,0),(0,255,85),
        (0,255,170),(0,255,255),(0,170,255),(0,85,255),(0,0,255),(85,0,255),(170,0,255),(255,0,255),(255,0,170),(255,0,85))
BODY_EDGES=((1,2),(1,5),(2,3),(3,4),(5,6),(6,7),(1,8),(8,9),(9,10),(1,11),(11,12),(12,13),(1,0),(0,14),(14,16),(0,15),(15,17))
HAND_EDGES=tuple((0 if i==0 else base+i-1,base+i) for base in (1,5,9,13,17) for i in range(4))


def draw_openpose(document, line_width=4, joint_radius=4, floor_grid=None):
    validate_pose(document)
    w,h=document["canvas_width"],document["canvas_height"]
    for value in (line_width,joint_radius):
        if type(value) is not int or not 1<=value<=32:
            raise ValueError("OpenPose line width and joint radius must be 1–32 pixels")
    scale=2 if w*h<=4_194_304 else 1
    image=Image.new("RGB",(w*scale,h*scale),(0,0,0));draw=ImageDraw.Draw(image)
    if floor_grid is not None:
        validate_pose({**document,"floor_grid":floor_grid})
        for x1,y1,x2,y2,major in floor_grid:
            color=(90,104,119) if major else (43,52,63)
            draw.line((x1*w*scale,y1*h*scale,x2*w*scale,y2*h*scale),fill=color,width=scale*(2 if major else 1))
    def points(field):
        values=document["people"][0][field]
        return [(values[i]*w*scale,values[i+1]*h*scale,values[i+2]) for i in range(0,len(values),3)]
    body=points("pose_keypoints_2d")
    for index,(a,b) in enumerate(BODY_EDGES):
        if body[a][2]<=0 or body[b][2]<=0:continue
        x1,y1,_=body[a];x2,y2,_=body[b]
        cx,cy=(x1+x2)/2,(y1+y2)/2;dx,dy=(x2-x1)/2,(y2-y1)/2
        length=math.hypot(dx,dy)
        if length<.001:continue
        px,py=-dy/length*line_width*scale,dx/length*line_width*scale
        polygon=[(cx+dx*math.cos(t)+px*math.sin(t),cy+dy*math.cos(t)+py*math.sin(t)) for t in np.linspace(0,math.tau,48,endpoint=False)]
        draw.polygon(polygon,fill=tuple(round(v*.6) for v in COLORS[index]))
    def dots(keypoints,colors,radius):
        for i,(x,y,confidence) in enumerate(keypoints):
            if confidence>0:draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=colors[i%len(colors)])
    dots(body,COLORS,joint_radius*scale)
    for field in ("hand_left_keypoints_2d","hand_right_keypoints_2d"):
        hand=points(field)
        for index,(a,b) in enumerate(HAND_EDGES):
            if hand[a][2]>0 and hand[b][2]>0:
                color=tuple(round(c*255) for c in colorsys.hsv_to_rgb(index/20,1,1))
                draw.line((hand[a][0],hand[a][1],hand[b][0],hand[b][1]),fill=color,width=max(1,round(line_width*scale/2)))
        dots(hand,[(0,0,255)],max(1,joint_radius*scale/2))
    if scale!=1:
        smaller=image.resize((w,h),Image.Resampling.LANCZOS);image.close();image=smaller
    return image


class OpenPoseImage:
    CATEGORY="DINKIssTyle/3D Pose"
    RETURN_TYPES=("IMAGE","POSE_KEYPOINT","STRING")
    RETURN_NAMES=("image","keypoints","openpose_json")
    FUNCTION="render"
    DESCRIPTION="Project the DKST 3D Pose camera into a COCO-18 OpenPose map, with 21 landmarks per hand."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "pose":("DKST_POSE",),
            "include_body":("BOOLEAN",{"default":True}),
            "include_hands":("BOOLEAN",{"default":True,"tooltip":"Include finger landmarks when the 3D scene uses detailed hands. Body wrist points remain when disabled."}),
            "include_head":("BOOLEAN",{"default":True,"tooltip":"Include synthetic nose, eye and ear landmarks. No 70-point facial-expression model."}),
            "line_width":("INT",{"default":4,"min":1,"max":32}),
            "joint_radius":("INT",{"default":4,"min":1,"max":32}),
            "include_floor":("BOOLEAN",{"default":True,"tooltip":"Show the floor grid when enabled in the 3D scene. Disable for a plain OpenPose conditioning map."}),
        }}

    def render(self,pose,include_body=True,include_hands=True,include_head=True,line_width=4,joint_radius=4,include_floor=True):
        if not isinstance(pose,dict) or set(pose)!={"scene_json","render_id"}:
            raise ValueError("Connect the pose output from DKST 3D Pose")
        stored=load_pose(pose["scene_json"],pose["render_id"])
        document=select_pose(stored,include_body,include_hands,include_head)
        image=draw_openpose(document,line_width,joint_radius,stored.get("floor_grid") if include_floor else None)
        try:pixels=np.asarray(image,dtype=np.float32)/255.0
        finally:image.close()
        return (torch.from_numpy(pixels.copy()).unsqueeze(0),[document],json.dumps(pixel_document(document),separators=(",",":")))
