import sys
import termios
import tty
import threading
import time

class KeyListener:
    def __init__(self):
        self.last_key_pressed = None
        self.should_exit = False
        self.listener_thread = None

    def getch(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    def listen(self):
        while not self.should_exit:
            char = self.getch()
            self.last_key_pressed = char
            if char == 'q':
                self.should_exit = True

    def start(self):
        self.listener_thread = threading.Thread(target=self.listen)
        self.listener_thread.daemon = True
        self.listener_thread.start()

