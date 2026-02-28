import cv2
import math
from hand_tracker import HandTracker
from test_tube import TestTube
from litmus_paper import LitmusPaper

CHEMICALS = {
    'HCl':  {'name': 'HCl (Acid)',  'liquid': (60, 60, 220),   'type': 'acid'},
    'NaOH': {'name': 'NaOH (Base)', 'liquid': (200, 80, 40),   'type': 'base'},
    'H2O':  {'name': 'H2O (Water)', 'liquid': (200, 200, 255), 'type': 'neutral'},
}

REACTION_TEXT = {
    ('red',  'acid'):    None,
    ('red',  'base'):    'Red litmus turns BLUE in presence of a Base!',
    ('red',  'neutral'): None,
    ('blue', 'acid'):    'Blue litmus turns RED in presence of an Acid!',
    ('blue', 'base'):    None,
    ('blue', 'neutral'): None,
}

# Litmus paper starting colors
LITMUS_COLORS = {
    'red':  (60,  60,  220),   # BGR red
    'blue': (200, 80,  40),    # BGR blue
}

# What color the paper should CHANGE TO on reaction
REACTION_COLORS = {
    ('red',  'base'): (200, 80,  40),   # red paper → turns blue (BGR)
    ('blue', 'acid'): (60,  60,  220),  # blue paper → turns red (BGR)
}

BUTTON_W, BUTTON_H = 140, 38
BUTTON_GAP         = 10
BUTTONS_START_X    = 10
BUTTONS_START_Y    = 10

# Litmus toggle button sits to the right of chemical buttons
LITMUS_BTN_X = BUTTONS_START_X + 3 * (BUTTON_W + BUTTON_GAP) + 20
LITMUS_BTN_Y = BUTTONS_START_Y
LITMUS_BTN_W = 160
LITMUS_BTN_H = 38


def get_buttons():
    buttons = []
    for i, (chem_id, chem) in enumerate(CHEMICALS.items()):
        x = BUTTONS_START_X + i * (BUTTON_W + BUTTON_GAP)
        buttons.append({'id': chem_id, 'chem': chem, 'x': x, 'y': BUTTONS_START_Y})
    return buttons


def draw_buttons(frame, buttons, active_id):
    for btn in buttons:
        x, y   = btn['x'], btn['y']
        chem   = btn['chem']
        active = btn['id'] == active_id
        accent = (60,60,220) if chem['type']=='acid' else (200,80,40) if chem['type']=='base' else (180,180,180)
        cv2.rectangle(frame, (x, y), (x+BUTTON_W, y+BUTTON_H), accent if active else (40,40,40), -1)
        cv2.rectangle(frame, (x, y), (x+BUTTON_W, y+BUTTON_H), accent, 2)
        cv2.putText(frame, chem['name'], (x+8, y+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)


def draw_litmus_button(frame, litmus_type):
    x, y = LITMUS_BTN_X, LITMUS_BTN_Y
    # Show which is active, toggle hint for the other
    label  = f'Litmus: {litmus_type.upper()}'
    accent = (60,60,220) if litmus_type == 'red' else (200,80,40)
    cv2.rectangle(frame, (x, y), (x+LITMUS_BTN_W, y+LITMUS_BTN_H), accent, -1)
    cv2.rectangle(frame, (x, y), (x+LITMUS_BTN_W, y+LITMUS_BTN_H), (255,255,255), 2)
    cv2.putText(frame, label, (x+10, y+25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    # Small hint
    cv2.putText(frame, '[click to toggle]', (x+10, y+LITMUS_BTN_H+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180,180,180), 1, cv2.LINE_AA)


def draw_reaction_text(frame, litmus_type, chemical_type, reacted):
    if not reacted:
        return
    msg = REACTION_TEXT.get((litmus_type, chemical_type))
    if not msg:
        return
    h, w   = frame.shape[:2]
    banner_y = h - 60
    cv2.rectangle(frame, (0, banner_y), (w, h), (20,20,20), -1)
    color = (200,80,40) if 'BLUE' in msg else (60,60,220)
    cv2.putText(frame, msg,
                (w//2 - len(msg)*4, banner_y+38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def make_paper(litmus_type):
    paper = LitmusPaper(x=310, y=420, width=90, height=130)
    c = list(LITMUS_COLORS[litmus_type])
    paper.base_color    = tuple(c)
    paper.current_color = c[:]
    paper.target_color  = c[:]
    return paper


def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    state   = param['state']
    buttons = param['buttons']

    # Check chemical buttons
    for btn in buttons:
        bx, by = btn['x'], btn['y']
        if bx < x < bx+BUTTON_W and by < y < by+BUTTON_H:
            if btn['id'] != state['active_id']:
                state['active_id'] = btn['id']
                state['reset']     = True
            return

    # Check litmus toggle button
    if LITMUS_BTN_X < x < LITMUS_BTN_X+LITMUS_BTN_W and LITMUS_BTN_Y < y < LITMUS_BTN_Y+LITMUS_BTN_H:
        state['litmus_type'] = 'blue' if state['litmus_type'] == 'red' else 'red'
        state['reset']       = True


def main():
    cap     = cv2.VideoCapture(0)
    tracker = HandTracker()

    state = {
        'active_id':   'H2O',
        'litmus_type': 'red',
        'reset':       False,
    }

    buttons = get_buttons()
    tube    = TestTube(x=350, y=150, width=60, height=200)
    paper   = make_paper(state['litmus_type'])
    reacted = False

    cv2.namedWindow('Virtual Chemistry Lab')
    cv2.setMouseCallback('Virtual Chemistry Lab', on_mouse,
                         {'state': state, 'buttons': buttons})

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # ── Reset on chemical or litmus change ────────────────────────────────
        if state['reset']:
            state['reset'] = False
            reacted        = False
            tube           = TestTube(x=350, y=150, width=60, height=200)
            paper          = make_paper(state['litmus_type'])

        litmus_type = state['litmus_type']
        chem        = CHEMICALS[state['active_id']]
        tube.liquid_color = chem['liquid']

        # ── If this combo causes a reaction, force paper toward reaction color ─
        reaction_color = REACTION_COLORS.get((litmus_type, chem['type']))
        if reacted and reaction_color:
            # Keep pushing target to fully change the paper color
            paper.target_color = list(reaction_color)

        # ── Hand tracking ─────────────────────────────────────────────────────
        frame = tracker.find_hands(frame)
        angle = tracker.get_hand_angle(frame)
        tube.set_angle(angle)

        # ── Draw ──────────────────────────────────────────────────────────────
        frame = paper.draw(frame)
        frame = tube.draw(frame)

        # ── Pour → reaction ───────────────────────────────────────────────────
        if tube.is_pouring and tube.liquid_level > 0:
            angle_rad   = math.radians(tube.display_angle)
            pivot_x     = tube.x + tube.width // 2
            pivot_y     = tube.y
            mouth_off_x = -(tube.width // 2)
            stream_x    = int(pivot_x + mouth_off_x * math.cos(angle_rad))
            stream_y    = int(pivot_y + mouth_off_x * math.sin(angle_rad))
            end_x       = stream_x - 45
            end_y       = stream_y + 130
            paper.receive_liquid(end_x, end_y + 85, chem['liquid'])

            if REACTION_TEXT.get((litmus_type, chem['type'])):
                reacted = True

        # ── UI ────────────────────────────────────────────────────────────────
        draw_buttons(frame, buttons, state['active_id'])
        draw_litmus_button(frame, litmus_type)
        draw_reaction_text(frame, litmus_type, chem['type'], reacted)

        cv2.imshow('Virtual Chemistry Lab', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()