"""Procedural asset generators. Each returns an entity template: a World State ``Entity`` whose id
is ``w:template`` and whose pose is the identity; :func:`steward.assets.library.instantiate`
assigns the scene id and pose.

Sizes are full extents in metres. Every template origin sits on the floor (z = 0) at the
footprint center, so a placement pose only needs x, y and yaw.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from steward.math3d import quat_from_axis_angle
from steward.scene.schema import Entity, Est, Joint, Link, RigidTree, exact, pose, primitive

TEMPLATE_ID = "w:template"

WOOD = [0.62, 0.45, 0.30, 1.0]
LIGHT_WOOD = [0.80, 0.67, 0.50, 1.0]
WALL = [0.92, 0.91, 0.88, 1.0]
FLOOR = [0.55, 0.52, 0.48, 1.0]
STEEL = [0.70, 0.72, 0.75, 1.0]
CERAMIC = [0.95, 0.95, 0.97, 1.0]
GLASS = [0.75, 0.88, 0.95, 0.6]
ROBOT_BODY = [0.20, 0.32, 0.55, 1.0]
ROBOT_ARM = [0.85, 0.55, 0.15, 1.0]


def _fixed(jid: str, parent: str, child: str, at: dict) -> Joint:
    return Joint(id=jid, parent=parent, child=child, type=exact("fixed"), origin=exact(at), q=exact([]), qd=exact([]))


def _entity(category: str, role: str, root: str, links: list[Link], joints: list[Joint] | None = None,
            state_vars: dict[str, Est] | None = None, root_actuator: dict | None = None) -> Entity:
    return Entity(
        id=TEMPLATE_ID,
        role=role,
        category=exact(category),
        pose=exact(pose()),
        body=RigidTree(root=exact(root), links=links, joints=joints or [], root_actuator=root_actuator),
        state_vars=state_vars or {},
    )


def building(width: float = 6.0, depth: float = 5.0, height: float = 2.6, wall: float = 0.1) -> Entity:
    """Floor plus four walls; every wall is a link welded to the floor by a fixed joint."""
    floor = Link(id="floor", name="floor",
                 geometry={"collision": [primitive("box", [width, depth, 0.1], pose([0, 0, -0.05]), FLOOR)]},
                 material=exact({"friction": 0.9}))
    walls = {
        "wall_north": ([0, depth / 2 + wall / 2, height / 2], [width + 2 * wall, wall, height]),
        "wall_south": ([0, -depth / 2 - wall / 2, height / 2], [width + 2 * wall, wall, height]),
        "wall_east": ([width / 2 + wall / 2, 0, height / 2], [wall, depth, height]),
        "wall_west": ([-width / 2 - wall / 2, 0, height / 2], [wall, depth, height]),
    }
    links = [floor]
    joints = []
    for name, (center, size) in walls.items():
        links.append(Link(id=name, name=name, geometry={"collision": [primitive("box", size, None, WALL)]}))
        joints.append(_fixed(f"{name}_mount", "floor", name, pose(center)))
    return _entity("building", "environment", "fixed", links, joints)


def counter(width: float = 1.8, depth: float = 0.6, height: float = 0.9) -> Entity:
    """Fixed kitchen counter without openable parts."""
    top = 0.04
    carcass = primitive("box", [width, depth, height - top], pose([0, 0, (height - top) / 2]), WOOD)
    slab = primitive("box", [width + 0.02, depth + 0.02, top], pose([0, 0, height - top / 2]), STEEL)
    link = Link(id="body", name="counter", geometry={"collision": [carcass, slab]},
                affordances=[exact("support-surface")])
    return _entity("counter", "object", "fixed", [link])


def dining_table(width: float = 1.2, depth: float = 0.8, height: float = 0.75, mass: float = 25.0) -> Entity:
    """Free-standing table: heavy but movable."""
    top_t, leg = 0.04, 0.05
    geoms = [primitive("box", [width, depth, top_t], pose([0, 0, height - top_t / 2]), LIGHT_WOOD)]
    leg_h = height - top_t
    for sx in (-1, 1):
        for sy in (-1, 1):
            at = pose([sx * (width / 2 - leg), sy * (depth / 2 - leg), leg_h / 2])
            geoms.append(primitive("box", [leg, leg, leg_h], at, WOOD))
    link = Link(id="body", name="table", geometry={"collision": geoms}, inertial=exact({"mass": mass}),
                material=exact({"friction": 0.8}), affordances=[exact("support-surface")])
    return _entity("table", "object", "free", [link])


def cabinet_with_drawer(width: float = 0.6, depth: float = 0.5, height: float = 0.8,
                        drawer_height: float = 0.2) -> Entity:
    """Fixed cabinet whose top section holds one drawer on a prismatic joint along +x (front)."""
    wall = 0.02
    lower_h = height - drawer_height - wall
    carcass = [
        primitive("box", [depth, width, lower_h], pose([0, 0, lower_h / 2]), WOOD),
        primitive("box", [depth, width, wall], pose([0, 0, height - wall / 2]), WOOD),
        primitive("box", [depth, wall, drawer_height], pose([0, width / 2 - wall / 2, lower_h + drawer_height / 2]), WOOD),
        primitive("box", [depth, wall, drawer_height], pose([0, -width / 2 + wall / 2, lower_h + drawer_height / 2]), WOOD),
        primitive("box", [wall, width, drawer_height], pose([-depth / 2 + wall / 2, 0, lower_h + drawer_height / 2]), WOOD),
    ]
    base = Link(id="carcass", name="carcass", geometry={"collision": carcass}, affordances=[exact("support-surface")])
    inner_w = width - 2 * wall - 0.01
    inner_d = depth - wall - 0.02
    dh = drawer_height - 0.02
    drawer_geoms = [
        primitive("box", [inner_d, inner_w, 0.01], pose([0, 0, 0.005]), LIGHT_WOOD),
        primitive("box", [0.02, inner_w + 0.008, dh], pose([inner_d / 2, 0, dh / 2]), LIGHT_WOOD),
        primitive("box", [0.012, inner_w, dh * 0.7], pose([-inner_d / 2, 0, dh * 0.35]), LIGHT_WOOD),
        primitive("box", [inner_d, 0.012, dh * 0.7], pose([0, inner_w / 2 - 0.006, dh * 0.35]), LIGHT_WOOD),
        primitive("box", [inner_d, 0.012, dh * 0.7], pose([0, -inner_w / 2 + 0.006, dh * 0.35]), LIGHT_WOOD),
        primitive("box", [0.03, 0.12, 0.02], pose([inner_d / 2 + 0.025, 0, dh / 2]), STEEL),
    ]
    drawer = Link(id="drawer", name="drawer", geometry={"collision": drawer_geoms}, inertial=exact({"mass": 1.5}),
                  affordances=[exact("pullable"), exact("graspable-handle"), exact("container")])
    slide = Joint(
        id="drawer_slide", parent="carcass", child="drawer", type=exact("prismatic"),
        origin=exact(pose([wall / 2 + 0.005, 0, lower_h + 0.006])), axis=exact([1.0, 0.0, 0.0]),
        limits=exact({"lo": 0.0, "hi": inner_d * 0.85}), q=exact([0.0]), qd=exact([0.0]),
        dynamics=exact({"damping": 8.0, "friction": 0.5}),
    )
    return _entity("cabinet", "object", "fixed", [base, drawer], [slide])


def cup(radius: float = 0.04, height: float = 0.10, mass: float = 0.25, fragile: bool = True) -> Entity:
    link = Link(id="body", name="cup",
                geometry={"collision": [primitive("cylinder", [radius, height], pose([0, 0, height / 2]), GLASS if fragile else CERAMIC)]},
                inertial=exact({"mass": mass}), material=exact({"friction": 1.0, "fragility": 0.8 if fragile else 0.2}),
                affordances=[exact("graspable"), exact("container")])
    return _entity("cup", "object", "free", [link])


def plate(radius: float = 0.12, height: float = 0.02, mass: float = 0.4) -> Entity:
    link = Link(id="body", name="plate",
                geometry={"collision": [primitive("cylinder", [radius, height], pose([0, 0, height / 2]), CERAMIC)]},
                inertial=exact({"mass": mass}), material=exact({"friction": 0.9, "fragility": 0.6}),
                affordances=[exact("graspable"), exact("support-surface")])
    return _entity("plate", "object", "free", [link])


def bowl(radius: float = 0.08, height: float = 0.06, mass: float = 0.3) -> Entity:
    link = Link(id="body", name="bowl",
                geometry={"collision": [primitive("cylinder", [radius, height], pose([0, 0, height / 2]), CERAMIC)]},
                inertial=exact({"mass": mass}), material=exact({"friction": 0.9, "fragility": 0.6}),
                affordances=[exact("graspable"), exact("container")])
    return _entity("bowl", "object", "free", [link])


def steward_bot(camera_pitch_deg: float = 30.0, camera_fovy_deg: float = 60.0) -> Entity:
    """Phase 0 mobile manipulator: planar base, vertical lift, horizontal reach, ideal gripper.

    The base collides with the world. The lift carriage, arm and gripper are visual-only links,
    so an approaching gripper never pushes the target: Phase 0 grasping is an ideal attach
    performed by WSim when the gripper closes within capture range.
    """
    base = Link(
        id="base", name="base",
        geometry={
            "collision": [primitive("box", [0.5, 0.44, 0.28], pose([0, 0, 0.16]), ROBOT_BODY)],
            "visual": [
                primitive("box", [0.5, 0.44, 0.28], pose([0, 0, 0.16]), ROBOT_BODY),
                primitive("box", [0.05, 0.05, 1.15], pose([0.18, 0, 0.87]), STEEL),
                primitive("box", [0.05, 0.05, 1.10], pose([-0.12, 0, 0.85]), STEEL),
            ],
        },
        inertial=exact({"mass": 60.0}),
    )
    pitch = math.radians(camera_pitch_deg)
    head = Link(
        id="head", name="head",
        geometry={"visual": [primitive("box", [0.08, 0.16, 0.08], None, [0.15, 0.15, 0.15, 1.0])]},
        inertial=exact({"mass": 0.5}),
        sensors=[{
            "id": "head_camera", "kind": "rgbd", "mount": pose([0.05, 0, 0], quat_from_axis_angle([0, 1, 0], pitch)),
            "intrinsics": {"width": 320, "height": 240, "fovy": camera_fovy_deg}, "rateHz": 10,
        }],
    )
    carriage = Link(id="carriage", name="carriage",
                    geometry={"visual": [primitive("box", [0.08, 0.16, 0.05], None, STEEL)]},
                    inertial=exact({"mass": 2.0}))
    arm = Link(id="arm", name="arm",
               geometry={"visual": [primitive("box", [0.5, 0.04, 0.03], pose([-0.25, 0, 0]), ROBOT_ARM)]},
               inertial=exact({"mass": 1.5}))
    gripper = Link(id="gripper", name="gripper",
                   geometry={"visual": [
                       primitive("box", [0.06, 0.012, 0.05], pose([0.0, 0.045, 0]), [0.1, 0.1, 0.1, 1.0]),
                       primitive("box", [0.06, 0.012, 0.05], pose([0.0, -0.045, 0]), [0.1, 0.1, 0.1, 1.0]),
                       primitive("box", [0.02, 0.10, 0.03], pose([-0.03, 0, 0]), [0.1, 0.1, 0.1, 1.0]),
                   ]},
                   inertial=exact({"mass": 0.5}), affordances=[exact("grasp-point")])
    joints = [
        _fixed("head_mount", "base", "head", pose([0.16, 0, 1.50])),
        Joint(id="lift", parent="base", child="carriage", type=exact("prismatic"), origin=exact(pose([0.22, 0, 0])),
              axis=exact([0.0, 0.0, 1.0]), limits=exact({"lo": 0.30, "hi": 1.30}), q=exact([0.60]), qd=exact([0.0]),
              dynamics=exact({"damping": 250.0}),
              actuator={"kind": "position", "limits": {"effort": 400.0, "velocity": 0.3}, "gains": {"kp": 6000.0}}),
        Joint(id="reach", parent="carriage", child="arm", type=exact("prismatic"), origin=exact(pose([0.0, 0, 0])),
              axis=exact([1.0, 0.0, 0.0]), limits=exact({"lo": 0.05, "hi": 0.70}), q=exact([0.10]), qd=exact([0.0]),
              dynamics=exact({"damping": 150.0}),
              actuator={"kind": "position", "limits": {"effort": 200.0, "velocity": 0.3}, "gains": {"kp": 2500.0}}),
        _fixed("gripper_mount", "arm", "gripper", pose([0.04, 0, 0])),
    ]
    root_actuator = {
        "kind": "position",
        "limits": {"effort": [900.0, 900.0, 250.0], "velocity": [0.5, 0.5, 1.0]},
        "gains": {"kp": [8000.0, 8000.0, 900.0], "damping": [1500.0, 1500.0, 130.0]},
    }
    return _entity("steward-bot", "robot", "planar", [base, head, carriage, arm, gripper], joints,
                   state_vars={"gripper": exact("open")}, root_actuator=root_actuator)


GENERATORS: dict[str, Callable[..., Entity]] = {
    "building": building,
    "counter": counter,
    "dining_table": dining_table,
    "cabinet_with_drawer": cabinet_with_drawer,
    "cup": cup,
    "plate": plate,
    "bowl": bowl,
    "steward_bot": steward_bot,
}
