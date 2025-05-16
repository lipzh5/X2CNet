# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None


LOW_LEVEL_CTRLS = {
	"Jaw Yaw", "Jaw Pitch",
	"Nose Wrinkle",
	"Lip Corner Raise Left", "Lip Corner Raise Right", "Lip Corner Stretch Left", "Lip Corner Stretch Right",
	"Lip Bottom Curl", "Lip Bottom Depress Left", "Lip Bottom Depress Right", "Lip Bottom Depress Middle",
	"Lip Top Raise Left", "Lip Top Raise Middle", "Lip Top Raise Right", "Lip Top Curl",
	"Eyelid Upper Left", "Eyelid Upper Right", "Eyelid Lower Left", "Eyelid Lower Right",
	"Brow Inner Left", "Brow Inner Right", "Brow Outer Left", "Brow Outer Right",
	'Head Roll', 'Head Yaw', 'Head Pitch', "Neck Pitch", "Neck Roll",
	"Gaze Target Phi", "Gaze Target Theta",  # 30 in total, NOTE: left & right eyes are always toward the same direction
	# 'Eye Pitch Left', 'Eye Pitch Right', 'Eye Yaw Left', 'Eye Yaw Right', "Gaze Target Distance",
}

HIGH_LEVEL_CTRLS = {"Jaw Yaw",
                    "Mouth Anger", "Mouth Content", "Mouth Disgust", "Mouth Fear", "Mouth Happy",
                    "Mouth Huh", "Mouth Joy", "Mouth Sad", "Mouth Sneer", "Mouth Surprise", "Mouth Worried",
                    "Mouth Open",  # NOTE: Mouth Open [0, 2]
                    "Nose Wrinkle",
                    "Brow Inner Left", "Brow Inner Right", "Brow Outer Left", "Brow Outer Right",  # 18 intotal
                    'Head Roll', 'Head Yaw', 'Head Pitch', "Neck Pitch", "Neck Roll",  # 23 intotal
                    "Gaze Target Phi", "Gaze Target Theta",  # 25 intotal  # # "Gaze Target Distance",
                    "Eyelid Upper Left", "Eyelid Upper Right", "Eyelid Lower Left", "Eyelid Lower Right",  # 29 intotal
                    }

ORDERED_CTRLS = sorted(list(LOW_LEVEL_CTRLS))
ORDERED_CTRLS_HIGH_LEVEL = sorted(list(HIGH_LEVEL_CTRLS))
ORDERED_CTRLS_RANGE = [("Brow Inner Left", [0, 1]), ("Brow Inner Right", [0, 1]), ("Brow Outer Left", [0, 1]),
                       ("Brow Outer Right", [0, 1]),
                       ("Eyelid Lower Left", [-1, 2]), ("Eyelid Lower Right", [-1, 2]), ("Eyelid Upper Left", [-1, 2]),
                       ("Eyelid Upper Right", [-1, 2]),
                       ("Gaze Target Phi", [-130, 130]), ("Gaze Target Theta", [-60, 60]), ("Head Pitch", [-30, 20]),
                       ("Head Roll", [-20, 20]),
                       ("Head Yaw", [-30, 30]), ("Jaw Pitch", [0, 1]), ("Jaw Yaw", [0, 1]), ("Lip Bottom Curl", [0, 1]),
                       ("Lip Bottom Depress Left", [0, 1]),
                       ("Lip Bottom Depress Middle", [0, 1]), ("Lip Bottom Depress Right", [0, 1]),
                       ("Lip Corner Raise Left", [0, 1]), ("Lip Corner Raise Right", [0, 1]),
                       ("Lip Corner Stretch Left", [0, 1]), ("Lip Corner Stretch Right", [0, 1]),
                       ("Lip Top Curl", [0, 1]), ("Lip Top Raise Left", [0, 1]),
                       ("Lip Top Raise Middle", [0, 1]), ("Lip Top Raise Right", [0, 1]),
                       ("Neck Pitch", [-20, 30]), ("Neck Roll", [-24, 24]), ("Nose Wrinkle", [0, 1]),
                       ]

NUM_CTRLS = len(ORDERED_CTRLS)
NUM_CTRLS_HIGH_LEVEL = len(ORDERED_CTRLS_HIGH_LEVEL)

HIGH_LEVEL_CTRL_IDX = {ctrl: idx for idx, ctrl in enumerate(ORDERED_CTRLS_HIGH_LEVEL)}
LOW_LEVEL_CTRL_IDX = {ctrl: idx for idx, ctrl in enumerate(ORDERED_CTRLS)}

TOTAL_CTRLS = {
	'Jaw Pitch': ('Jaw Pitch', None),
	'Jaw Yaw': ('Jaw Yaw', 'Mesmer Mouth 2'), 'Mouth Open': ('Mouth Open', 'Mesmer Mouth 2'),
	'Mouth Anger': ('Mouth Anger', 'Mesmer Mouth 2'), 'Mouth Content': ('Mouth Content', 'Mesmer Mouth 2'),
	'Mouth Disgust': ('Mouth Disgust', 'Mesmer Mouth 2'), 'Mouth Fear': ('Mouth Fear', 'Mesmer Mouth 2'),
	'Mouth Happy': ('Mouth Happy', 'Mesmer Mouth 2'), 'Mouth Huh': ('Mouth Huh', 'Mesmer Mouth 2'),
	'Mouth Joy': ('Mouth Joy', 'Mesmer Mouth 2'), 'Mouth Sad': ('Mouth Sad', 'Mesmer Mouth 2'),
	'Mouth Sneer': ('Mouth Sneer', 'Mesmer Mouth 2'), 'Mouth Surprise': ('Mouth Surprise', 'Mesmer Mouth 2'),
	'Mouth Worried': ('Mouth Worried', 'Mesmer Mouth 2'), 'Nose Wrinkle': ('Nose Wrinkle', 'Mesmer Nose 1'),
	'Eyelid Upper Left': ('Eyelid Upper Left', 'Mesmer Eyelids 1'),
	'Eyelid Upper Right': ('Eyelid Upper Right', 'Mesmer Eyelids 1'),
	'Eyelid Lower Left': ('Eyelid Lower Left', 'Mesmer Eyelids 1'),
	'Eyelid Lower Right': ('Eyelid Lower Right', 'Mesmer Eyelids 1'),
	'Brow Inner Left': ('Brow Inner Left', 'Mesmer Brows 1'),
	'Brow Inner Right': ('Brow Inner Right', 'Mesmer Brows 1'),
	'Brow Outer Left': ('Brow Outer Left', 'Mesmer Brows 1'),
	'Brow Outer Right': ('Brow Outer Right', 'Mesmer Brows 1'),
	# head/heck
	"Head Pitch": ('Head Pitch', "Mesmer Neck 1"), "Head Roll": ('Head Roll', 'Mesmer Neck 1'),
	"Head Yaw": ('Head Yaw', 'Mesmer Neck 1'), "Neck Pitch": ('Neck Pitch', 'Mesmer Neck 1'),
	"Neck Roll": ('Neck Roll', 'Mesmer Neck 1'),
	# low level control
	"Nose": ('Nose', None), "Lip Bottom Curl": ('Lip Bottom Curl', None),
	"Lip Bottom Depress Left": ('Lip Bottom Depress Left', None),
	"Lip Bottom Depress Middle": ('Lip Bottom Depress Middle', None),
	"Lip Bottom Depress Right": ('Lip Bottom Depress Right', None),
	"Lip Corner Raise Left": ('Lip Corner Raise Left', None),
	"Lip Corner Raise Right": ('Lip Corner Raise Right', None),
	"Lip Corner Stretch Left": ('Lip Corner Stretch Left', None),
	"Lip Corner Stretch Right": ('Lip Corner Stretch Right', None),
	"Lip Top Curl": ('Lip Top Curl', None), "Lip Top Raise Left": ('Lip Top Raise Left', None),
	"Lip Top Raise Middle": ('Lip Top Raise Middle', None),
	"Lip Top Raise Right": ('Lip Top Raise Right', None),
	"Gaze Target Theta": ('Gaze Target Theta', 'Mesmer Gaze 1'),
	"Gaze Target Phi": ('Gaze Target Phi', 'Mesmer Gaze 1'),
}

MOUTH_LOW_LEVEL = {"Jaw Yaw", "Jaw Pitch",
                   "Lip Corner Raise Left", "Lip Corner Raise Right", "Lip Corner Stretch Left",
                   "Lip Corner Stretch Right",
                   "Lip Bottom Curl", "Lip Bottom Depress Left", "Lip Bottom Depress Right",
                   "Lip Bottom Depress Middle",
                   "Lip Top Raise Left", "Lip Top Raise Middle", "Lip Top Raise Right", "Lip Top Curl", }

ORDERED_MOUTH_LOW_LEVEL = sorted(list(MOUTH_LOW_LEVEL))

MOUTH_NEUTRAL = {
	"Jaw Yaw": 0.5, "Jaw Pitch": 1,
	"Lip Corner Raise Left": 0.47, "Lip Corner Raise Right": 0.62, "Lip Corner Stretch Left": 0.64,
	"Lip Corner Stretch Right": 0.31,
	"Lip Bottom Curl": 0.46, "Lip Bottom Depress Left": 0.56, "Lip Bottom Depress Right": 0.54,
	"Lip Bottom Depress Middle": 0.43,
	"Lip Top Raise Left": 0.48, "Lip Top Raise Middle": 0.3, "Lip Top Raise Right": 0.45, "Lip Top Curl": 0.41
}

FACE_NEUTRAL = {
	"Jaw Yaw": 0.5, "Jaw Pitch": 1,
	"Lip Corner Raise Left": 0.47, "Lip Corner Raise Right": 0.62, "Lip Corner Stretch Left": 0.64,
	"Lip Corner Stretch Right": 0.31,
	"Lip Bottom Curl": 0.46, "Lip Bottom Depress Left": 0.56, "Lip Bottom Depress Right": 0.54,
	"Lip Bottom Depress Middle": 0.43,
	"Lip Top Raise Left": 0.48, "Lip Top Raise Middle": 0.3, "Lip Top Raise Right": 0.45, "Lip Top Curl": 0.41,
	'Nose': 0.85, 'Brow Inner Left': 0.6, 'Brow Inner Right': 0.68, 'Brow Outer Left': 0.65, 'Brow Outer Right': 0.53,
	'Eyelid Upper Left': 1, 'Eyelid Upper Right': 1, 'Eyelid Lower Left': 1, 'Eyelid Lower Right': 1,
	'Gaze Target Theta': 0, 'Gaze Target Phi': 0, 'Head Pitch': 0, 'Head Roll': 0, 'Head Yaw': 0, 'Neck Pitch': 0,
	'Neck Roll': 0,

}

FACE_NEUTRAL_HLV = {
	"Jaw Yaw": 0.5,
	"Mouth Anger": 0, "Mouth Content": 0, "Mouth Disgust": 0, "Mouth Fear": 0, "Mouth Happy": 0,
	"Mouth Huh": 0, "Mouth Joy": 0, "Mouth Sad": 0, "Mouth Sneer": 0, "Mouth Surprise": 0, "Mouth Worried": 0,
	"Mouth Open": 0,  # NOTE: Mouth Open [0, 2]
	"Nose Wrinkle": 0,
	'Brow Inner Left': 0.6, 'Brow Inner Right': 0.68, 'Brow Outer Left': 0.65, 'Brow Outer Right': 0.53,
	'Eyelid Upper Left': 1, 'Eyelid Upper Right': 1, 'Eyelid Lower Left': 1, 'Eyelid Lower Right': 1,
	'Gaze Target Theta': 0, 'Gaze Target Phi': 0, 'Head Pitch': 0, 'Head Roll': 0, 'Head Yaw': 0, 'Neck Pitch': 0,
	'Neck Roll': 0,
}

