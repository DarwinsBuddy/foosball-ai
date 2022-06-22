import sys
import glfw
import OpenGL.GL as gl


class OpenGLDisplay:

    def __init__(self, calibration_mode):
        # Initialize the library
        if not glfw.init():
            sys.exit()
        self.calibration_mode = calibration_mode
        self.all_windows = {}
        self.closed = False
        self.tracker = None

    def set_tracker(self, tracker):
        self.tracker = tracker

    def _create_window(self, name, width, height):
        # Create a windowed mode window and its OpenGL context
        window = glfw.create_window(width, height, name, None, None)
        if not window:
            glfw.terminate()
            sys.exit()
        else:
            self.all_windows[name] = window
        # Install a key handler
        glfw.set_key_callback(window, self._on_key)
        return window

    def show(self, name, frame, pos='tl'):
        # TODO: use position
        [height, width, channels] = frame.shape
        if name not in self.all_windows:
            window = self._create_window(name, width, height)
        else:
            window = self.all_windows[name]
        # Make the window's context current
        glfw.make_context_current(window)
        ratio = width / float(height)
        gl.glViewport(0, 0, width, height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(-ratio, ratio, -1, 1, 1, -1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        gl.glBegin(gl.GL_POINTS)
        gl.glLoadIdentity()
        gl.glDrawPixels(width, height, gl.GL_RGB, gl.GL_INT, frame)

        gl.glEnd()

        # Swap front and back buffers
        glfw.swap_buffers(window)

    def _on_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_Q:
                glfw.set_window_should_close(window, 1)
                self.closed = True
            elif key == glfw.KEY_R and self.tracker is not None and self.calibration_mode:
                self.tracker.reset_bounds()

    def poll_key(self):
        glfw.poll_events()
        return self.closed

    @staticmethod
    def stop():
        glfw.terminate()
