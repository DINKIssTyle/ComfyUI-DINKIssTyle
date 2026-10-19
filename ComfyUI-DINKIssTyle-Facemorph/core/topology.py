"""Stable semantic vertex order shared by automatic and manually edited faces."""
OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
EYE_L = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
EYE_R = [263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466]
BROW_L = [70,63,105,66,107,55,65,52,53,46]
BROW_R = [300,293,334,296,336,285,295,282,283,276]
LIPS = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185]
MOUTH = [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191]
NOSE = [1,2,4,5,6,19,94,97,98,115,168,195,197,220,275,278,294,326,327,344,440]
REGIONS = ['face', 'eye_l', 'eye_r', 'nose', 'mouth', 'lips', 'brow_l', 'brow_r']
GROUPS = [0] * 468
for group, indices in [(1,EYE_L),(2,EYE_R),(3,NOSE),(4,MOUTH),(5,LIPS),(6,BROW_L),(7,BROW_R)]:
    for index in indices:
        GROUPS[index] = group
PAIRS = {'eyes': ('eye_l', 'eye_r'), 'brows': ('brow_l', 'brow_r')}
PARENTS = {child: parent for parent, children in PAIRS.items() for child in children}
GROUPS += [0]*(len(OVAL)+8)
ALPHAS = [1.0]*468 + [0.0]*(len(OVAL)+8)
COUNT = len(GROUPS)
CURVES = [OVAL, EYE_L, EYE_R, BROW_L, BROW_R, LIPS, MOUTH, NOSE]
