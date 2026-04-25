from flask import Flask, render_template, Response, send_file
import cv2
import mediapipe as mp
import numpy as np
import io

app = Flask(__name__)

# ---------- Mediapipe init ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------- Video capture ----------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1920)
cap.set(4, 1080)

canvas = None
prev_x, prev_y = None, None
smooth_x, smooth_y = None, None
alpha = 0.65
lost_frames = 0
max_lost_frames = 6
draw_color = (0, 0, 255) # BGR format
draw_thickness = 6
bg_mode = 'camera'
last_frame = None
line_style = 'normal'
start_x, start_y = None, None
last_x, last_y = None, None

rainbow_hue = 0
undo_stack = []
redo_stack = []
is_drawing = False

def gen_frames():
    global canvas, prev_x, prev_y, smooth_x, smooth_y, lost_frames, draw_color, draw_thickness, bg_mode, last_frame
    global line_style, start_x, start_y, last_x, last_y, rainbow_hue
    global undo_stack, redo_stack, is_drawing


    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        if canvas is None:
            h, w, _ = frame.shape
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        draw_mode = False
        clear_mode = False

        if draw_color == 'rainbow':
            rainbow_hue = (rainbow_hue + 1) % 180
            hsv = np.uint8([[[rainbow_hue, 255, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            current_draw_color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        else:
            current_draw_color = draw_color

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            lm = hand_landmarks.landmark
            h, w, _ = frame.shape

            ix, iy = int(lm[8].x * w), int(lm[8].y * h)
            if smooth_x is None or smooth_y is None:
                smooth_x, smooth_y = ix, iy
            else:
                smooth_x = int(alpha * ix + (1 - alpha) * smooth_x)
                smooth_y = int(alpha * iy + (1 - alpha) * smooth_y)

            def dist_w(idx):
                return np.hypot(lm[idx].x - lm[0].x, lm[idx].y - lm[0].y)
            
            index_up = dist_w(8) > dist_w(6)
            middle_up = dist_w(12) > dist_w(10)
            ring_up = dist_w(16) > dist_w(14)
            pinky_up = dist_w(20) > dist_w(18)
            thumb_up = dist_w(4) > dist_w(2)
            total_fingers = sum([index_up, middle_up, ring_up, pinky_up, thumb_up])

            pause_mode = False
            
            if total_fingers == 5:
                clear_mode = True
            elif index_up and middle_up and not ring_up and not pinky_up:
                pause_mode = True
            elif index_up:
                draw_mode = True

            if draw_mode:
                if not is_drawing:
                    undo_stack.append(canvas.copy())
                    if len(undo_stack) > 20:
                        undo_stack.pop(0)
                    redo_stack.clear()
                    is_drawing = True
                    
                current_color = current_draw_color
                current_thickness = draw_thickness
                
                if line_style == 'normal':
                    if prev_x is None:
                        prev_x, prev_y = smooth_x, smooth_y
                    cv2.line(canvas, (prev_x, prev_y), (smooth_x, smooth_y), current_color, current_thickness)
                    prev_x, prev_y = smooth_x, smooth_y
                elif line_style == 'dotted':
                    if prev_x is None:
                        prev_x, prev_y = smooth_x, smooth_y
                        cv2.circle(canvas, (smooth_x, smooth_y), max(2, draw_thickness // 2), current_draw_color, -1)
                    else:
                        dist = np.hypot(smooth_x - prev_x, smooth_y - prev_y)
                        if dist > draw_thickness * 2.5:
                            cv2.circle(canvas, (smooth_x, smooth_y), max(2, draw_thickness // 2), current_draw_color, -1)
                            prev_x, prev_y = smooth_x, smooth_y
                elif line_style in ['straight', 'rectangle', 'circle']:
                    if start_x is None:
                        start_x, start_y = smooth_x, smooth_y
                    last_x, last_y = smooth_x, smooth_y
                lost_frames = 0
            elif pause_mode:
                is_drawing = False
                prev_x, prev_y = smooth_x, smooth_y
                start_x, start_y = smooth_x, smooth_y
                last_x, last_y = smooth_x, smooth_y
                lost_frames = 0
            else:
                is_drawing = False
                lost_frames += 1
                if lost_frames > max_lost_frames:
                    if start_x is not None and last_x is not None:
                        if line_style == 'straight':
                            cv2.line(canvas, (start_x, start_y), (last_x, last_y), current_draw_color, draw_thickness)
                        elif line_style == 'rectangle':
                            cv2.rectangle(canvas, (start_x, start_y), (last_x, last_y), current_draw_color, draw_thickness)
                        elif line_style == 'circle':
                            radius = int(np.hypot(last_x - start_x, last_y - start_y))
                            cv2.circle(canvas, (start_x, start_y), radius, current_draw_color, draw_thickness)
                    prev_x, prev_y = None, None
                    start_x, start_y = None, None
                    last_x, last_y = None, None
                    smooth_x, smooth_y = None, None

            if clear_mode:
                if canvas.max() > 0:
                    undo_stack.append(canvas.copy())
                    if len(undo_stack) > 20:
                        undo_stack.pop(0)
                    redo_stack.clear()
                    canvas[:] = 0
                prev_x, prev_y = None, None
                start_x, start_y = None, None
                last_x, last_y = None, None

        else:
            is_drawing = False
            lost_frames += 1
            if lost_frames > max_lost_frames:
                if start_x is not None and last_x is not None:
                    if line_style == 'straight':
                        cv2.line(canvas, (start_x, start_y), (last_x, last_y), current_draw_color, draw_thickness)
                    elif line_style == 'rectangle':
                        cv2.rectangle(canvas, (start_x, start_y), (last_x, last_y), current_draw_color, draw_thickness)
                    elif line_style == 'circle':
                        radius = int(np.hypot(last_x - start_x, last_y - start_y))
                        cv2.circle(canvas, (start_x, start_y), radius, current_draw_color, draw_thickness)
                prev_x, prev_y = None, None
                start_x, start_y = None, None
                last_x, last_y = None, None
                smooth_x, smooth_y = None, None

        # Opaque overlay technique using mask
        canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(canvas_gray, 5, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        
        if bg_mode == 'black':
            frame = np.zeros_like(frame)
        elif bg_mode == 'white':
            frame = np.ones_like(frame) * 255
            
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        frame = cv2.add(frame_bg, canvas)
        
        if start_x is not None and last_x is not None and lost_frames <= max_lost_frames:
            if line_style == 'straight':
                cv2.line(frame, (start_x, start_y), (last_x, last_y), current_draw_color, draw_thickness)
            elif line_style == 'rectangle':
                cv2.rectangle(frame, (start_x, start_y), (last_x, last_y), current_draw_color, draw_thickness)
            elif line_style == 'circle':
                radius = int(np.hypot(last_x - start_x, last_y - start_y))
                cv2.circle(frame, (start_x, start_y), radius, current_draw_color, draw_thickness)
        
        last_frame = frame.copy()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_color/<color>')
def set_color(color):
    global draw_color, draw_thickness
    draw_thickness = 6
    if color == 'red': draw_color = (0, 0, 255)
    elif color == 'blue': draw_color = (255, 0, 0)
    elif color == 'green': draw_color = (0, 255, 0)
    elif color == 'yellow': draw_color = (0, 255, 255)
    elif color == 'purple': draw_color = (255, 0, 255)
    elif color == 'pink': draw_color = (203, 192, 255)
    elif color == 'cyan': draw_color = (255, 255, 0)
    elif color == 'orange': draw_color = (0, 165, 255)
    elif color == 'white': draw_color = (255, 255, 255)
    elif color == 'rainbow': draw_color = 'rainbow'
    elif color == 'eraser':
        draw_color = (0, 0, 0)
        draw_thickness = 40
    elif color.startswith('hex_'):
        hex_val = color[4:]
        if len(hex_val) == 6:
            try:
                r = int(hex_val[0:2], 16)
                g = int(hex_val[2:4], 16)
                b = int(hex_val[4:6], 16)
                draw_color = (b, g, r)
            except ValueError:
                pass
    return '{"status": "success"}'

@app.route('/set_bg/<mode>')
def set_bg(mode):
    global bg_mode
    bg_mode = mode
    return '{"status": "success"}'

@app.route('/set_style/<style>')
def set_style(style):
    global line_style, start_x, start_y, last_x, last_y, prev_x, prev_y
    line_style = style
    start_x, start_y = None, None
    last_x, last_y = None, None
    prev_x, prev_y = None, None
    return '{"status": "success"}'

@app.route('/set_thickness/<int:size>')
def set_thickness(size):
    global draw_thickness
    draw_thickness = size
    return '{"status": "success"}'

@app.route('/save_image')
def save_image():
    global last_frame
    if last_frame is not None:
        ret, buffer = cv2.imencode('.png', last_frame)
        response = Response(buffer.tobytes(), mimetype='image/png')
        response.headers['Content-Disposition'] = 'attachment; filename=auradraw_masterpiece.png'
        return response
    return '{"status": "error"}'

@app.route('/clear')
def clear_board():
    global canvas, undo_stack, redo_stack
    if canvas is not None:
        if canvas.max() > 0:
            undo_stack.append(canvas.copy())
            if len(undo_stack) > 20:
                undo_stack.pop(0)
            redo_stack.clear()
        canvas[:] = 0
    return '{"status": "success"}'

@app.route('/undo')
def undo():
    global canvas, undo_stack, redo_stack
    if undo_stack and canvas is not None:
        redo_stack.append(canvas.copy())
        canvas = undo_stack.pop()
    return '{"status": "success"}'

@app.route('/redo')
def redo():
    global canvas, undo_stack, redo_stack
    if redo_stack and canvas is not None:
        undo_stack.append(canvas.copy())
        canvas = redo_stack.pop()
    return '{"status": "success"}'

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)