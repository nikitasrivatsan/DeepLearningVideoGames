import os
import Tkinter

child_env = dict(os.environ)
child_env['SDL_WINDOWID'] = the_window_id
child_env['SDL_VIDEO_WINDOW_POS'] = '{},{}'.format(left, top)
p = subprocess.Popen([path, width, height], env=child_env)
