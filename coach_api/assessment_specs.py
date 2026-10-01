"""CoachOS assessment display/reference specifications.

Reference limits are aligned with report_generator.py. Each metric carries a
machine-readable rule so the API can mark out-of-spec values deterministically.
"""

ASSESSMENT_SPECS = {
    "front": {
        "title": "Front 正面體態",
        "section_path": "biomechanics.front",
        "image": "/resources/front.png",
        "metrics": [
            {"key": "f1_rotation_amp_pct", "label": "F-1. 旋轉幅度", "reference": "< 3%", "unit": "%", "decimals": 1, "rule": {"type": "lt", "value": 3}},
            {"key": "f2_shldr_diff_cm", "label": "F-2. 肩高度差 (cm)", "reference": "< 1", "unit": "cm", "decimals": 1, "rule": {"type": "lt", "value": 1}},
            {"key": "f3_hip_diff_cm", "label": "F-3. 髖高度差 (cm)", "reference": "< 1", "unit": "cm", "decimals": 1, "rule": {"type": "lt", "value": 1}},
            {"key": "f4_knee_diff_cm", "label": "F-4. 膝高度差 (cm)", "reference": "< 1", "unit": "cm", "decimals": 1, "rule": {"type": "lt", "value": 1}},
            {"key": "f5_l_knee_out_angle", "label": "F-5. 左膝外側角度", "reference": "175° - 185°", "unit": "°", "decimals": 0, "rule": {"type": "range", "low": 175, "high": 185}},
            {"key": "f6_r_knee_out_angle", "label": "F-6. 右膝外側角度", "reference": "175° - 185°", "unit": "°", "decimals": 0, "rule": {"type": "range", "low": 175, "high": 185}},
        ],
    },
    "side": {
        "title": "Side 側面體態",
        "section_path": "biomechanics.side",
        "image": "/resources/side.png",
        "metrics": [
            {"key": "s1_neck_dev_angle", "label": "S-1. 頸偏移角度", "reference": "< 10°", "unit": "°", "decimals": 0, "rule": {"type": "lt", "value": 10}},
            {"key": "s2_shldr_dev_angle", "label": "S-2. 肩偏移角度", "reference": "< 3°", "unit": "°", "decimals": 0, "rule": {"type": "lt", "value": 3}},
            {"key": "s3_hip_angle", "label": "S-3. 髖關節角度", "reference": "170° - 180°", "unit": "°", "decimals": 0, "rule": {"type": "range", "low": 170, "high": 180}},
            {"key": "s4_knee_angle", "label": "S-4. 膝關節角度", "reference": "170° - 175°", "unit": "°", "decimals": 0, "rule": {"type": "range", "low": 170, "high": 175}},
            {"key": "s5_ankle_dev_angle", "label": "S-5. 踝偏移角度", "reference": "< 3°", "unit": "°", "decimals": 0, "rule": {"type": "lt", "value": 3}},
        ],
    },
    "bridge": {
        "title": "Shoulder Bridge 肩橋式",
        "section_path": "biomechanics.shoulder_bridge",
        "image": "/resources/bridge.png",
        "metrics": [
            {"key": "b1_neck_angle", "label": "B-1. 頸部角度", "reference": "127° - 130°", "unit": "°", "decimals": 0, "rule": {"type": "range", "low": 127, "high": 130}},
            {"key": "b2_hip_angle", "label": "B-2. 髖關節角度", "reference": "175° - 185°", "unit": "°", "decimals": 0, "rule": {"type": "range", "low": 175, "high": 185}},
            {"key": "b3_knee_angle", "label": "B-3. 膝關節角度", "reference": "69° - 72°", "unit": "°", "decimals": 0, "rule": {"type": "range", "low": 69, "high": 72}},
            {"key": "b4_back_lift_angle", "label": "B-4. 背部上抬角度", "reference": "24° - 26°", "unit": "°", "decimals": 0, "rule": {"type": "range", "low": 24, "high": 26}},
        ],
    },
    "ohs": {
        "title": "Overhead Squat 過頭深蹲",
        "section_path": "biomechanics.overhead_squat",
        "image": "/resources/ohs.png",
        "metrics": [
            {"key": "o1_shldr_angle", "label": "O-1. 肩關節角度", "reference": "155° - 162°", "unit": "°", "decimals": 0, "rule": {"type": "range", "low": 155, "high": 162}},
            {"key": "o2_hip_angle", "label": "O-2. 髖關節角度", "reference": "< 91°", "unit": "°", "decimals": 0, "rule": {"type": "lt", "value": 91}},
            {"key": "o3_knee_angle", "label": "O-3. 膝關節角度", "reference": "< 95°", "unit": "°", "decimals": 0, "rule": {"type": "lt", "value": 95}},
            {"key": "o4_ankle_flex_angle", "label": "O-4. 踝前屈角度", "reference": "> 24°", "unit": "°", "decimals": 0, "rule": {"type": "gt", "value": 24}},
            {"key": "o5_trunk_lean_angle", "label": "O-5. 軀幹前傾角度", "reference": "13° - 30°", "unit": "°", "decimals": 0, "rule": {"type": "range", "low": 13, "high": 30}},
        ],
    },
    "bird_dog": {
        "title": "Bird Dog 鳥狗式",
        "section_path": "biomechanics.swimming",
        "image": "/resources/birddog.png",
        "metrics": [
            {"key": "w1_neck_angle", "label": "W-1. 頸部角度", "reference": "185° - 190°", "unit": "°", "decimals": 0, "rule": {"type": "range", "low": 185, "high": 190}},
            {"key": "w2_elbow_angle", "label": "W-2. 肘關節角度", "reference": "166° - 169°", "unit": "°", "decimals": 0, "rule": {"type": "range", "low": 166, "high": 169}},
            {"key": "w3_hip_angle", "label": "W-3. 髖關節角度", "reference": "170° - 186°", "unit": "°", "decimals": 0, "rule": {"type": "range", "low": 170, "high": 186}},
            {"key": "w4_knee_angle", "label": "W-4. 膝關節角度", "reference": "> 156°", "unit": "°", "decimals": 0, "rule": {"type": "gt", "value": 156}},
            {"key": "w5_trunk_sway_cm", "label": "W-5. 軀幹搖晃值", "reference": "< 10 cm", "unit": "cm", "decimals": 1, "rule": {"type": "lt", "value": 10}},
        ],
    },
}
