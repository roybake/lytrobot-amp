"""MuJoCo GLFW viewer with press/release-aware arrow-key walking control."""

from __future__ import annotations

import glfw
import mujoco
import numpy as np

from .walking_commands import SPEED_ACCELERATION, YAW_ACCELERATION, heading_velocity, world_yaw_rate


class KeyboardWalkingViewer:
    """Render on the main thread so held keys are read without OS repeat delays."""

    def __init__(self, model, data, env, initial_speed: float):
        self.model, self.data, self.env = model, data, env
        self.speed = float(np.clip(initial_speed, *env.cfg.velocity_range))
        self.yaw_rate = 0.0
        self.tap_turn = 0
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed; keyboard playback requires a desktop display.")
        self.window = glfw.create_window(1280, 720, "LYT command walking", None, None)
        if self.window is None:
            glfw.terminate()
            raise RuntimeError("Could not create the MuJoCo keyboard window.")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        self.env.walking_commands.set_manual(self.speed, 0.0)

    def _on_key(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key in (glfw.KEY_UP, glfw.KEY_DOWN):
            self.speed += 0.05 if key == glfw.KEY_UP else -0.05
            self.speed = float(np.clip(self.speed, *self.env.cfg.velocity_range))
        elif key in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
            self.tap_turn = 1 if key == glfw.KEY_LEFT else -1

    def _on_scroll(self, window, dx, dy):
        self.cam.distance = max(0.2, self.cam.distance * np.exp(-0.1 * dy))

    def update_commands(self):
        glfw.poll_events()
        focused = glfw.get_window_attrib(self.window, glfw.FOCUSED)
        pressed = lambda key: focused and glfw.get_key(self.window, key) == glfw.PRESS
        speed_direction = int(pressed(glfw.KEY_UP)) - int(pressed(glfw.KEY_DOWN))
        turn_direction = int(pressed(glfw.KEY_LEFT)) - int(pressed(glfw.KEY_RIGHT))
        self.speed = float(np.clip(
            self.speed + speed_direction * SPEED_ACCELERATION * self.env.dt,
            *self.env.cfg.velocity_range))
        if turn_direction:
            self.yaw_rate += turn_direction * YAW_ACCELERATION * self.env.dt
            if self.tap_turn:
                self.yaw_rate = self.tap_turn * 0.15
        else:
            # A short tap still supplies a turn command for one policy step.
            self.yaw_rate = self.tap_turn * 0.15 if focused else 0.0
        self.tap_turn = 0
        self.yaw_rate = float(np.clip(self.yaw_rate, *self.env.cfg.yaw_rate_range))
        self.env.walking_commands.set_manual(self.speed, self.yaw_rate)

    def is_running(self):
        return not glfw.window_should_close(self.window)

    def sync(self):
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        if width <= 0 or height <= 0:
            glfw.poll_events()
            return
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                              mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.context)
        velocity = heading_velocity(self.data.qpos[3:7], self.data.qvel[:3])
        yaw = float(world_yaw_rate(self.data.qpos[3:7], self.data.qvel[3:6]))
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            "UP / DOWN: speed   LEFT / RIGHT: turn\nHold: accelerate   Release: keep speed, stop turning\n"
            f"Command: {self.speed:.2f} m/s   {self.yaw_rate:+.2f} rad/s\n"
            f"Measured: {velocity[0]:.2f} m/s   {yaw:+.2f} rad/s",
            "", self.context)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self):
        glfw.make_context_current(self.window)
        self.context.free()
        del self.scene
        glfw.destroy_window(self.window)
        glfw.terminate()
