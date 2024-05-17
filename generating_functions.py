import random

NOISE = 1
### Standard sythethic generating fuctions ###
V3_GEN = lambda v1, v2: v1 + v2 + random.uniform(-NOISE, NOISE)
V4_GEN = lambda v2, v3: v2 * v3 + random.uniform(-NOISE, NOISE)
V5_GEN = lambda v2, v3: 3*v2 + v3 + random.uniform(-NOISE, NOISE)
Y_GEN = lambda v1, v3, v4, v6: v1 * 0.4 + v3*v6*0.3 + v4*0.1 + v6 + random.uniform(-NOISE, NOISE)
V7_GEN = lambda y: y*y*0.5 + random.uniform(-NOISE, NOISE)
V8_GEN = lambda v2, y: v2 + y*0.2 + random.uniform(-NOISE, NOISE)
V9_GEN = lambda v7, v8: v7*0.4 + v8*v7*0.1 + random.uniform(-NOISE, NOISE)

standard = {
    "name": "data/standard.csv",
    "starting_names": ["V1", "V2", "V6"],
    "starting_generating_boundaries": [(0, 1), (0, 1), (0, 1)],
    "downstream_names": ["V3", "V4", "V5", "Y", "V7", "V8", "V9"],
    "downstream_generating_functions": [V3_GEN, V4_GEN, V5_GEN, Y_GEN, V7_GEN, V8_GEN, V9_GEN],
    "downstream_parents": [("V1", "V2"), ("V2", "V3"), ("V2", "V3"), ("V1", "V3", "V4", "V6"), ("Y",), ("V2", "Y"), ("V7", "V8")]
} 

demo = {
    "name": "data/demo.csv",
    "starting_names": ["V1", "V2", "V6"],
    "starting_generating_boundaries": [(0, 1), (0, 1), (0, 1)],
    "downstream_names": ["V3", "V4", "V5", "Y", "V7", "V8", "V9"],
    "downstream_generating_functions": [V3_GEN, V4_GEN, V5_GEN, Y_GEN, V7_GEN, V8_GEN, V9_GEN],
    "downstream_parents": [("V1", "V2"), ("V2", "V3"), ("V2", "V3"), ("V1", "V3", "V4", "V6"), ("Y",), ("V2", "Y"), ("V7", "V8")]
} 
### END standard sythethic generating fuctions ###

### Downstream Distribution shift sythethic generating fuctions ###
V8_GEN_SHIFT = lambda v2, y: v2 + 0.2 + random.uniform(-NOISE, NOISE)

downstream_shift = {
    "name": "data/downstream_shift.csv",
    "starting_names": ["V1", "V2", "V6"],
    "starting_generating_boundaries": [(0, 1), (0, 1), (0, 1)],
    "downstream_names": ["V3", "V4", "V5", "Y", "V7", "V8", "V9"],
    "downstream_generating_functions": [V3_GEN, V4_GEN, V5_GEN, Y_GEN, V7_GEN, V8_GEN_SHIFT, V9_GEN],
    "downstream_parents": [("V1", "V2"), ("V2", "V3"), ("V2", "V3"), ("V1", "V3", "V4", "V6"), ("Y"), ("V2", "Y"), ("V7", "V8")]
} 
### END Downstream Distribution shift sythethic generating fuctions ###

### Upstream Distribution shift sythethic generating fuctions ###

upstream_shift = {
    "name": "data/upstream_shift.csv",
    "starting_names": ["V1", "V2", "V6"],
    "starting_generating_boundaries": [(0, 1), (5, 6), (0, 1)],
    "downstream_names": ["V3", "V4", "V5", "Y", "V7", "V8", "V9"],
    "downstream_generating_functions": [V3_GEN, V4_GEN, V5_GEN, Y_GEN, V7_GEN, V8_GEN, V9_GEN],
    "downstream_parents": [("V1", "V2"), ("V2", "V3"), ("V2", "V3"), ("V1", "V3", "V4", "V6"), ("Y"), ("V2", "Y"), ("V7", "V8")]
} 
### END Upstream Distribution shift sythethic generating fuctions ###


### Unmeasured Confounding sythethic generating fuctions ###

# note: starting vars = U_3_Y, U_6_Y, V1, V2
V3_GEN = lambda v1, v2, u3y: v1 + v2 + 0.7*u3y + random.uniform(-NOISE, NOISE)
V4_GEN = lambda v2: v2 * 0.5 + random.uniform(-NOISE, NOISE)
V5_GEN = lambda v2, v3: 3*v2 + v3 + random.uniform(-NOISE, NOISE)
Y_GEN = lambda v3, v4, v7, u3y, u6y: u6y + u3y * 0.4 + v3*v7*0.3 + v4*0.1 + v7 + random.uniform(-NOISE, NOISE)
V6_GEN = lambda u6y: u6y + random.uniform(-NOISE, NOISE)
V7_GEN = lambda v6: v6 + random.uniform(-NOISE, NOISE)
V8_GEN = lambda v2, y: v2 + y*0.2 + random.uniform(-NOISE, NOISE)


mixed_standard = {
    "name": "data/temp.csv",
    "starting_names": ["V1", "V2", "U_3_Y", "U_6_Y"],
    "starting_generating_boundaries": [(0, 1), (0, 1), (0, 1), (0, 1)],
    "downstream_names": ["V3", "V4", "V5", "V6", "V7", "Y", "V8"],
    "downstream_generating_functions": [V3_GEN, V4_GEN, V5_GEN, V6_GEN, V7_GEN, Y_GEN, V8_GEN],
    "downstream_parents": [("V1", "V2", "U_3_Y"), ("V2",), ("V2", "V3"), ("U_6_Y",), ("V6",), ( "V3", "V4", "V7", "U_3_Y", "U_6_Y"), ("V2", "Y")]
}

### END Unmeasured Confounding sythethic generating fuctions ###




