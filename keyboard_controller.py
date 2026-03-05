# keyboard_controller.py
import os
# Force pygame/SDL2 to use X11 (not Wayland) to avoid GLFW/Wayland conflict
# when MuJoCo viewer is also running. SDL_VIDEODRIVER must be set BEFORE pygame.init().
os.environ.setdefault('SDL_VIDEODRIVER', 'x11')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
import pygame
import numpy as np

class KeyboardController:
    def __init__(self,
                 vx_scale=0.5,
                 vy_scale=0.5,
                 yaw_scale=2.5,
                 smooth=0.3):

        pygame.init()

        self.vx_scale = vx_scale
        self.vy_scale = vy_scale
        self.yaw_scale = yaw_scale
        self.smooth = smooth

        self.vx = 0.0
        self.vy = 0.0
        self.yaw = 0.0

        # Larger window for control display
        self.screen = pygame.display.set_mode((400, 200))
        pygame.display.set_caption("Keyboard Control Instructions")

        self.font = pygame.font.SysFont("Arial", 18)

    def smooth_update(self, old, new):
        return old * (1 - self.smooth) + new * self.smooth

    def draw_instructions(self):
        """Draw text UI (control instructions) on the pygame window."""
        try:
            self.screen.fill((20, 20, 20))  # dark background

            lines = [
                "Keyboard Control Rules:",
                "-----------------------------",
                "W / S : Move Forward / Backward (vx)",
                "A / D : Move Left / Right     (vy)",
                "Q / E : Rotate Left / Right   (yaw)",
            ]

            y = 20
            for line in lines:
                text_surface = self.font.render(line, True, (255, 255, 255))
                self.screen.blit(text_surface, (20, y))
                y += 30

            pygame.display.flip()
        except pygame.error:
            # Can't update display from non-main thread, skip drawing
            pass

    def read(self):
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        # Draw UI every frame
        self.draw_instructions()

        target_vx = 0.0
        target_vy = 0.0
        target_yaw = 0.0

        # Forward / backward
        if keys[pygame.K_w]:
            target_vx = 1.0
        elif keys[pygame.K_s]:
            target_vx = -1.0

        # Left / right
        if keys[pygame.K_a]:
            target_vy = 1.0
        elif keys[pygame.K_d]:
            target_vy = -1.0

        # Yaw rotation
        if keys[pygame.K_q]:
            target_yaw = self.yaw_scale
        elif keys[pygame.K_e]:
            target_yaw = -self.yaw_scale

        # Smooth transition
        self.vx = self.smooth_update(self.vx, target_vx)
        self.vy = self.smooth_update(self.vy, target_vy)
        self.yaw = self.smooth_update(self.yaw, target_yaw)

        return np.array([self.vx * self.vx_scale, self.vy * self.vy_scale, self.yaw], dtype=np.float32)
