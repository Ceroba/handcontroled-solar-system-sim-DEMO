import glfw
from OpenGL.GL import *
import glm
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import random as rnd
# #include <bits/stdc++.h>
# using namespace std;
 
# int main()
# {
#     ios::sync_with_stdio(0);
#     cin.tie(nullptr);
#     int t;
#     cin >> t;
#     while (t--)
#     {
#         int n;
#         cin >> n;
#         int i = n;
#         while (i >= 1)
#         {
#             cout << i << " ";
#             i--;
#         }
#     }
#     return 0;
# }
# #include <iostream>
# #include <vector>
 
# using namespace std;
 
# void solve() {
#     int n;
#     cin >> n;

#     int blocked_chairs = 0;
#     for (int i = 1; i <= n; ++i) {
#         int p_i;
#         cin >> p_i;

#         if (p_i > i) {
#             blocked_chairs++;
#         }
#     }

#     cout << n - blocked_chairs << "\n";
# }
 
# int main() {
#     ios_base::sync_with_stdio(false);
#     cin.tie(NULL);

#     int t;
#     cin >> t;
#     while (t--) {
#         solve();
#     }
#     return 0;
# }
vs_code = """
#version 330 core
layout(location=0)in vec3 pos;
layout(location=1)in vec3 col;
uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;
out float dist;
out vec3 _col;
void main(){
    dist = sqrt(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);
    gl_Position = projection * view *transform*vec4(pos, 1.0);
    gl_PointSize = 4.0f;
    _col = col;
}
"""
fs_code = """
#version 330 core
out vec4 color;
in float dist;
in vec3 _col;
void main(){
    color = vec4(_col, 1.0);
    //color = dist * vec4(0, 0, 1, 1);
}
"""
def look_at(pos : glm.vec3, tar: glm.vec3):
    forward = glm.normalize(tar - pos) 
    w_up = glm.vec3(0, 1, 0)
    right = glm.normalize(glm.cross(forward, w_up))
    up = glm.normalize(glm.cross(right, forward))
    return glm.lookAt(pos, pos + forward, up)


G = 6.6743e-11  
H_ANGLE = np.pi / 180 * 72
V_ANGLE = glm.atan(1.0 / 2.0)
radius = 2
h_angle1 = -np.pi / 2 - H_ANGLE / 2
h_angle2 = -np.pi / 2


def sun_color_func():
    color : glm.vec3 = glm.vec3(1.0, 183.0 / 255.0 * rnd.random(), 0.25)
    return color
def earth_color_func():
    color : glm.vec3 = glm.vec3(.1, 183.0 / 255.0 * rnd.random(),1)
    return color
def murc_color_func():
    color : glm.vec3 = glm.vec3(1, 1,1 )* rnd.random()
    return color
def generateUV_sphere(radius, sector_count, stack_count, color_func):
    positons = []
    colors= []
    sec_step = (2 * np.pi) / sector_count
    stk_step = np.pi / stack_count
    for i in range(stack_count + 1):
        stack_angle = np.pi / 2 - i * stk_step
        xz = radius * glm.cos(stack_angle)
        y = radius * glm.sin(stack_angle)
        for j in range(sector_count + 1):
            sector_angle = j * sec_step + rnd.random() * np.pi
            x = xz * glm.cos(sector_angle)
            z = xz * glm.sin(sector_angle)
            positons.append(x)
            positons.append(y)
            positons.append(z)
            color = color_func()
            colors.append(color.x)
            colors.append(color.y)
            colors.append(color.z)
    positons = np.array(positons, dtype=np.float32)
    colors = np.array(colors, dtype=np.float32)
    return positons, colors
class Obj:
    def __init__(self,_pos, _vel, _mass, _den, _rad, _color_func):
        self.pos = _pos
        self.vel = _vel
        self.mass = _mass
        self.den = _den
        self.rad = _rad
        self.color_func = _color_func
        pass
    def render_tovert(self, pos, col):
        _pos,_col = generateUV_sphere(self.rad, 32, 32, self.color_func)
        for i in _pos:
            pos.append(i)
        for i in _col:
            col.append(i)




glfw.init()
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
glfw.window_hint(glfw.SAMPLES, 4)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

window = glfw.create_window(800, 800, "hello", None, None)
glfw.make_context_current(window)
shader_prog = compileProgram(compileShader(vs_code, GL_VERTEX_SHADER), compileShader(fs_code, GL_FRAGMENT_SHADER))
glUseProgram(shader_prog)

glEnable(GL_PROGRAM_POINT_SIZE)
positions = []
colors = []
# sun = Obj(glm.vec3(0, 0, 0), glm.vec3(3, 3, 3), 70, 1, 2, sun_color_func)
# earth = Obj(glm.vec3(0, 0, 0), glm.vec3(3, 3, 3), 70, 1, 3, earth_color_func)

_positions = []
_colors = []
objs = [
    Obj(glm.vec3(0, 0.0001, 0), glm.vec3(0, 0, 0), 5.97219 *pow(10, 9), 1, 2.5, sun_color_func),
    Obj(glm.vec3(0, 0.003, 9), glm.vec3(-2, 0, 1), 5.97219*pow(10, 5), 1, 1, earth_color_func),
    Obj(glm.vec3(0, -0.005, 7), glm.vec3(-2, 0, 1), 5.97219*pow(10, 6), 1, 0.5, murc_color_func)
]
for i in objs:
    i.render_tovert(_positions, _colors)
# _positions = np.array(_positions, dtype=np.float32)
# _colors = np.array(_colors, dtype=np.float32)
# for i in _positions:
#     positions.append(i)
# for i in _colors:
#     colors.append(i)
# _positions,_colors = generateUV_sphere(1.0, 86, 86, earth_color_func)
# for i in _positions:
#     positions.append(i)
# for i in _colors:
#     colors.append(i)
positions = np.array(_positions, dtype=np.float32)
colors = np.array(_colors, dtype=np.float32)
indecies = []
vao = glGenVertexArrays(1)
glBindVertexArray(vao)
pos_vbo = glGenBuffers(1)
glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
glBindBuffer(GL_ARRAY_BUFFER, pos_vbo)
glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_STATIC_DRAW)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
glEnableVertexAttribArray(0)
col_vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, col_vbo)
glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
glEnableVertexAttribArray(1)
# ebo = glGenBuffers(1)
# glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
# glBufferData(GL_ELEMENT_ARRAY_BUFFER, indecies.nbytes, indecies, GL_STATIC_DRAW)
fps = 60.0
last_t = 0.0
current_t = 60.0
dt = 0.16666
projection = glm.perspective(glm.radians(70.0),800/800,0.1, 100.0)
right = glm.vec3(1, 0, 0)
theta = np.pi / 2
peta = -np.pi / 2
cam_dist = 15
xz = cam_dist * glm.cos(peta)
y = cam_dist * glm.sin(peta)
x = xz * glm.cos(theta)
z = xz * glm.sin(theta)
view = look_at(glm.vec3(x, y, z), glm.vec3(0, 0, 0))
transform = glm.mat4(1)
mvp = projection * view * transform
glUniformMatrix4fv(glGetUniformLocation(shader_prog, "transform"), 1, GL_FALSE, glm.value_ptr(transform))
glUniformMatrix4fv(glGetUniformLocation(shader_prog, "view"), 1, GL_FALSE, glm.value_ptr(view))
glUniformMatrix4fv(glGetUniformLocation(shader_prog, "projection"), 1, GL_FALSE, glm.value_ptr(projection))
glEnable(GL_DEPTH_TEST)
transform_loc = glGetUniformLocation(shader_prog, "transform")
while not glfw.window_should_close(window):
    current_t = glfw.get_time()
    if (current_t - last_t) >= (1 /fps):
        dt = (current_t - last_t) * 40
        peta += 0.01
        peta = glm.clamp(peta, -np.pi / 2, np.pi / 2)
        xz = cam_dist * glm.cos(peta)
        y = cam_dist * glm.sin(peta)
        x = xz * glm.cos(theta)
        z = xz * glm.sin(theta)
        view = look_at(glm.vec3(x, y, z), glm.vec3(0, 0, 0))
        glUniformMatrix4fv(glGetUniformLocation(shader_prog, "view"), 1, GL_FALSE, glm.value_ptr(view))
        last_t = current_t
        glfw.poll_events()
        # print(dt)
        for i in range(len(objs)):
            for j in range(i +1, len(objs)):
                _dir = objs[j].pos - objs[i].pos
                dist = np.sqrt(_dir.x * _dir.x + _dir.y * _dir.y + _dir.z * _dir.z)
                f = G * ((objs[i].mass * objs[j].mass) / (dist * dist))
                _dir = glm.normalize(_dir)
                a_1 = (_dir * f) / objs[i].mass
                a_2 = (-_dir * f) / objs[j].mass

                objs[i].vel += a_1 * dt
                objs[j].vel += a_2 * dt
        # for j in range(len(objs)):
        #     objs[j].pos += objs[j].vel * dt / 96
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        j = 0    
        for i in objs:
            transform = glm.translate(glm.mat4(1.0), i.pos)
            glUniformMatrix4fv(transform_loc, 1, GL_FALSE, glm.value_ptr(transform))
            glDrawArrays(GL_POINTS, j * 33 * 33 ,33*33)
            j += 1    
        glfw.swap_buffers(window=window)