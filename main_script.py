import glfw
from OpenGL.GL import *
import glm
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import random as rnd
import keyboard
import cv2
import mediapipe as mp
import math as m
import time
import pyautogui
import numpy as np
# from sklearn.svm import SVC
import joblib
vs_code = """
#version 330 core
layout(location=0)in vec3 pos;
layout(location=1)in vec3 col;
uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;
uniform int is_stars;
out float dist;
out vec3 _col;
void main(){
    dist = sqrt(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);
    gl_Position = projection * view *transform*vec4(pos, 1.0);
    if (is_stars != 0){
    gl_PointSize = 1.0f;
    _col = vec3(0, 1, 1) * (dist - 50 + 0.1);
    }
    else{
    gl_PointSize = 2.0f;
    _col = col;
    }
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

G = 6.6743e-11  
H_ANGLE = np.pi / 180 * 72
V_ANGLE = glm.atan(1.0 / 2.0)
h_angle1 = -np.pi / 2 - H_ANGLE / 2
h_angle2 = -np.pi / 2

def bruh(landmarks):
    return [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]
def extract_features(landmark):
    middle = landmark[9]
    wrist  = landmark[0]

    size = m.hypot(
        middle.x - wrist.x,
        middle.y - wrist.y
    )
    f = []
    for lm in landmark:
        x = (lm.x - wrist.x) / size
        y = (lm.y - wrist.y) / size
        z = (lm.z - wrist.z) / size
        f.extend([x, y, z])
    return f
# def finger_up(hand_landmark, tip_id, pip_id):
#     tip = hand_landmarks.hand_landmarks[tip_id].y
#     pip = hand_landmarks.hand_landmarks[pip_id].y
#     return tip > pip
def look_at(pos : glm.vec3, tar: glm.vec3):
    forward = glm.normalize(tar - pos) 
    w_up = glm.vec3(0, 1, 0)
    right = glm.normalize(glm.cross(forward, w_up))
    up = glm.normalize(glm.cross(right, forward))
    return glm.lookAt(pos, pos + forward, up)
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
            sector_angle = j * sec_step + rnd.random() * np.pi / 4
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
def gen_stars(n):
    radius = 50
    positons = []
    sec_step = (2 * np.pi) / n
    stk_step = np.pi / n
    for i in range(n + 1):
        for j in range(n + 1):
            stack_angle = np.pi / 2 - i * stk_step + rnd.random() * np.pi
            rnd_rad = radius + rnd.random()
            xz = rnd_rad * glm.cos(stack_angle)
            y = rnd_rad * glm.sin(stack_angle)
            sector_angle = j * sec_step + rnd.random() * np.pi
            x = xz * glm.cos(sector_angle)
            z = xz * glm.sin(sector_angle)
            positons.append(x)
            positons.append(y)
            positons.append(z)
    positons = np.array(positons, dtype=np.float32)
    return positons
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


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.8
)
def smooth_step(a, b, t):
    tmp = glm.clamp((t - a) / (b - a), 0, 1)
    return tmp * tmp * (3 - 2 * tmp) 
cap = cv2.VideoCapture(0)
w = cap.get(3)
h = cap.get(4)
data = []
lable = []
can_press = True
clf = joblib.load("clf.pkl")
hand_pos = []
glfw.init()
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
glfw.window_hint(glfw.SAMPLES, 4)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

window = glfw.create_window(900, 900, "hello", None, None)
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

positions = gen_stars(128)
stars_vao = glGenVertexArrays(1)
glBindVertexArray(stars_vao)
star_pos_vbo = glGenBuffers(1)
glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
glBindBuffer(GL_ARRAY_BUFFER, star_pos_vbo)
glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_STATIC_DRAW)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
glEnableVertexAttribArray(0)

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
cam_dist = 20
xz = cam_dist * glm.cos(peta)
y = cam_dist * glm.sin(peta)
x = xz * glm.cos(theta)
z = xz * glm.sin(theta)
t_peta = 0.0
t_theta = 0.0
view = look_at(glm.vec3(x, y, z), glm.vec3(0, 0, 0))
transform = glm.mat4(1)
mvp = projection * view * transform
glUniformMatrix4fv(glGetUniformLocation(shader_prog, "transform"), 1, GL_FALSE, glm.value_ptr(transform))
glUniformMatrix4fv(glGetUniformLocation(shader_prog, "view"), 1, GL_FALSE, glm.value_ptr(view))
glUniformMatrix4fv(glGetUniformLocation(shader_prog, "projection"), 1, GL_FALSE, glm.value_ptr(projection))
glUniform1i(glGetUniformLocation(shader_prog, "is_stars"), 0)
glEnable(GL_DEPTH_TEST)
transform_loc = glGetUniformLocation(shader_prog, "transform")
dt_mul = 40
while not glfw.window_should_close(window):
    current_t = glfw.get_time()
    if (current_t - last_t) >= (1 /fps):
        ret, frame = cap.read()
        if not ret:
            break
        # peta += 0.01
        t_peta = glm.clamp(t_peta, -np.pi / 2, np.pi / 2)
        if peta != t_peta:
            peta += (t_peta - peta) / 10 
        if theta != t_theta:
            theta += (t_theta - theta) / 10 
        peta = glm.clamp(peta, -np.pi / 2, np.pi / 2)
        xz = cam_dist * glm.cos(peta)
        y = cam_dist * glm.sin(peta)
        x = xz * glm.cos(theta)
        z = xz * glm.sin(theta)
        view = look_at(glm.vec3(x, y, z), glm.vec3(0, 0, 0))
        glUniformMatrix4fv(glGetUniformLocation(shader_prog, "view"), 1, GL_FALSE, glm.value_ptr(view))
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            
        
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
                fingertips = {
                    "nukle": hand_landmarks.landmark[5],
                    "thumb": hand_landmarks.landmark[4],
                    "index": hand_landmarks.landmark[8]
                }
                
                # x0,y0 = fingertips["thumb"].x, fingertips["thumb"].y 
                # x1,y1 = fingertips["index"].x, fingertips["index"].y
                # dist = m.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0))

                # if dist < 0.05 and can_press:
                #     if can_press:
                #         print("pause")
                #         pyautogui.press("space")
                #         can_press = False
                # elif dist >= 0.05:
                #     can_press = True
                x_pos = int(fingertips["index"].x * w)
                y_pos = int(fingertips["index"].y * h - 10)
                pred = str(clf.predict([extract_features(hand_landmarks.landmark)])[0])
                cv2.putText( frame,pred, (x_pos, y_pos),cv2.FONT_HERSHEY_COMPLEX,1,(255, 255, 255), 2)
                hand_pos.append(fingertips["nukle"])
                if len(hand_pos) > 2:
                    hand_pos.pop(0)
                if len(hand_pos) > 1:
                   
                    dx = hand_pos[1].x - hand_pos[0].x 
                    dy = hand_pos[1].y - hand_pos[0].y
                    thre = 0.02
                    if "open_palm" == pred:
                        if abs(dx) > abs(dy):
                            if dx > thre:
                                t_theta += 0.2
                            elif dx < -thre:
                                t_theta -= 0.2
                        else:
                            if dy > thre:
                                t_peta += 0.2
                            elif dy < -thre:
                                t_peta -= 0.2
                    
                    # if pred == "open_palm":

                # if pred == "pinch":
                #     if can_press:
                #         print("pause")
                #         pyautogui.press("space")
                #         can_press = False
                # else:
                #     can_press = True 
                # if pred == "fist":
                #     dt_mul = 1000
                # else:
                #     dt_mul = 40  
                if keyboard.is_pressed("q"):
                    data.append(extract_features(hand_landmarks.landmark))
                    lable.append("fist")
                if keyboard.is_pressed("a"):
                    data.append(extract_features(hand_landmarks.landmark))
                    lable.append("open_palm")
                if keyboard.is_pressed("w"):
                    data.append(extract_features(hand_landmarks.landmark))
                    lable.append("pinch")
                if keyboard.is_pressed("space"):
                    print(clf.predict([extract_features(hand_landmarks.landmark)])[0])
            
                

        cv2.imshow("Hand Tracking", frame)

        dt = (current_t - last_t) * dt_mul
        last_t = current_t
        glfw.poll_events()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glBindVertexArray(stars_vao)
        glUniform1i(glGetUniformLocation(shader_prog, "is_stars"), 1)
        transform = glm.mat4(1.0)
        glUniformMatrix4fv(glGetUniformLocation(shader_prog, "transform"), 1, GL_FALSE, glm.value_ptr(transform))
        glDrawArrays(GL_POINTS,0, 129 *129)



        glBindVertexArray(vao)
        glUniform1i(glGetUniformLocation(shader_prog, "is_stars"), 0)
        
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
        for j in range(len(objs)):
            objs[j].pos += objs[j].vel * dt / 96
        j = 0    
        for i in objs:
            transform = glm.translate(glm.mat4(1.0), i.pos)
            glUniformMatrix4fv(transform_loc, 1, GL_FALSE, glm.value_ptr(transform))
            glDrawArrays(GL_POINTS, j * 33 * 33 ,33*33)
            j += 1    
        glfw.swap_buffers(window=window)