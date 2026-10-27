#!/usr/bin/env python3
"""GAD graph node definition for snap pin joint."""

import math

joint_node = {
    "type": "snap_pin",
    "dof": 1,
    "axis": [
        0,
        0,
        1
    ],
    "limits_deg": [
        -90,
        90
    ],
    "limits_rad": [
        -1.570796,
        1.570796
    ],
    "pin_diameter_mm": 7.0,
    "pin_height_mm": 5.0,
    "ridge_diameter_mm": 8.6,
    "ridge_height_mm": 1.0,
    "hole_diameter_mm": 7.3,
    "cavity_diameter_mm": 8.7,
    "cavity_height_mm": 1.5,
    "clearance_radial_mm": 0.1499999999999999,
    "interference_ridge_mm": -0.04999999999999982,
    "link_dimensions_mm": [
        60.0,
        12.0,
        4.0
    ],
    "material": "PLA",
    "mass_kg": 0.01,
    "joint_position_mm": [
        54.0,
        0.0,
        4.0
    ],
    "stl_files": [
        "link_A.stl",
        "link_B.stl"
    ],
    "urdf_file": "snap_joint.urdf",
    "printable": True,
    "assembly_method": "press_fit_snap",
    "min_wall_thickness_mm": 1.7000000000000002
}

if __name__ == "__main__":
    print("=== GAD Snap Pin Joint Node ===")
    for k, v in joint_node.items():
        print(f"  {k:.<30s} {v}")
