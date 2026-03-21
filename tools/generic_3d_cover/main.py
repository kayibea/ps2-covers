import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
COVERS_DEFAULT = os.path.join(REPO_ROOT, 'covers', 'default')
COVERS_3D = os.path.join(REPO_ROOT, 'covers', '3d')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'generic_3d_cover')
GAME_INDEX = os.path.join(REPO_ROOT, 'tools', 'GameIndex.yaml')
TEMPLATE = os.path.join(SCRIPT_DIR, 'ps2-template.png') # template https://forums.launchbox-app.com/files/file/5484-sony-ps2-3d-box-template-ntsc/
FONT_PATH = 'YuGothB.ttc'
FONT_PATH_KOREAN = 'malgun.ttf'


pts_cover = np.array([
    [47, 27],
    [543, 73],
    [543, 832],
    [47, 857],
], dtype=np.float32)

pts_spine = np.array([
    [7, 254],
    [39, 254],
    [39, 858],
    [8, 848],
], dtype=np.float32)

SPINE_FONT_SIZE = 50
SPINE_TEXT_COLOR = (0, 0, 0, 255)
SPINE_WIDTH = 100
spine_w = float(np.linalg.norm(pts_spine[1] - pts_spine[0]))
spine_h = float(np.linalg.norm(pts_spine[3] - pts_spine[0]))
spine_height = max(100, int(SPINE_WIDTH * spine_h / spine_w))

MARGIN = 8
SERIAL_FONT_SIZE = 25
START_OFFSET = int(spine_height * 0.01)
SERIAL_RESERVE = 0.18

def pick_font(text, default_font, korean_font):
    # Try to detect if the text contains Korean characters and pick the appropriate font
    for ch in text:
        cp = ord(ch)
        if (0xAC00 <= cp <= 0xD7A3 or
                0x1100 <= cp <= 0x11FF or
                0x3130 <= cp <= 0x318F):
            return korean_font
    return default_font

def load_game_index(path):
    print(f"Loading GameIndex from {path} ...")
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict):
        return data
    return {}


def get_game_info(game_index, serial):
    entry = game_index.get(serial)
    if not entry:
        return None, None
    name = entry.get('name') or entry.get('name-en')
    region = entry.get('region', '')
    return name, region


def existing_3d_serials(covers_3d_dir):
    serials = set()
    if not os.path.isdir(covers_3d_dir):
        return serials
    for fname in os.listdir(covers_3d_dir):
        base, _ = os.path.splitext(fname)
        serials.add(base.upper())
    return serials


def default_covers(covers_default_dir):
    covers = []
    for fname in os.listdir(covers_default_dir):
        base, ext = os.path.splitext(fname)
        if ext.lower() in ('.jpg', '.jpeg', '.png'):
            covers.append((base.upper(), os.path.join(covers_default_dir, fname)))
    return covers


def render_3d_cover(cover_2d_path, game_name, serial, template_path, font_path, font_path_korean=None):
    background = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    cover_2d = cv2.imread(cover_2d_path)

    if background is None:
        raise FileNotFoundError(f"Template not found: {template_path}")
    if cover_2d is None:
        raise FileNotFoundError(f"Cover not found: {cover_2d_path}")

    if background.shape[2] == 4:
        alpha_background = background[:, :, 3].copy()
        background = background[:, :, :3]
    else:
        alpha_background = np.full((background.shape[0], background.shape[1]), 255, dtype=np.uint8)

    H_out, W_out = background.shape[:2]
    SCALE = 4

    h, w = cover_2d.shape[:2]
    pts_origen = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    pts_cover_2x = pts_cover * SCALE
    M_2x = cv2.getPerspectiveTransform(pts_origen, pts_cover_2x)
    warp_2x = cv2.warpPerspective(cover_2d, M_2x, (W_out * SCALE, H_out * SCALE),
                                   flags=cv2.INTER_LANCZOS4)
    background_2x = cv2.resize(background, (W_out * SCALE, H_out * SCALE), interpolation=cv2.INTER_LINEAR)
    mask_2x = np.zeros((H_out * SCALE, W_out * SCALE), dtype=np.uint8)
    cv2.fillConvexPoly(mask_2x, pts_cover_2x.astype(int), 255)

    comp_2x = background_2x.copy()
    comp_2x[mask_2x == 255] = warp_2x[mask_2x == 255]

    result = cv2.resize(comp_2x, (W_out, H_out), interpolation=cv2.INTER_AREA)
    alpha_mask_1x = cv2.resize(mask_2x, (W_out, H_out), interpolation=cv2.INTER_AREA)
    alpha_result = np.maximum(alpha_background, alpha_mask_1x)

    def compose_spine(img, warp_np):
        rgb = warp_np[:, :, :3]
        alpha = warp_np[:, :, 3]
        mask = alpha.astype(np.float32) / 255.0
        for c in range(3):
            img[:, :, c] = (img[:, :, c] * (1 - mask) + rgb[:, :, c] * mask).astype(np.uint8)
        return img, np.maximum(alpha_result, alpha)

    def warp_rgba(canvas_np, M, shape):
        kw = dict(flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return (cv2.warpPerspective(canvas_np[:, :, :3], M, shape, **kw),
                cv2.warpPerspective(canvas_np[:, :, 3],  M, shape, **kw))

    out_shape = (background.shape[1], background.shape[0])

    SS = 2
    spine_img = Image.new('RGBA', (spine_height * SS, SPINE_WIDTH * SS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(spine_img)

    title_space = int((spine_height * (1 - SERIAL_RESERVE) - START_OFFSET - MARGIN) * SS)
    active_font = pick_font(game_name, font_path, font_path_korean or font_path)
    font_size = SPINE_FONT_SIZE * SS
    while font_size >= 8:
        try:    font = ImageFont.truetype(active_font, font_size)
        except: font = ImageFont.load_default(); break
        bbox   = draw.textbbox((0, 0), game_name, font=font)
        w_text = bbox[2] - bbox[0]
        h_text = bbox[3] - bbox[1]
        if w_text <= title_space and h_text <= (SPINE_WIDTH - MARGIN * 2) * SS:
            break
        font_size -= 1

    draw.text(((MARGIN + START_OFFSET) * SS, (SPINE_WIDTH * SS - h_text) // 2),
              game_name, font=font, fill=SPINE_TEXT_COLOR)

    spine_img = spine_img.filter(ImageFilter.GaussianBlur(radius=0.6))
    spine_img_1x = spine_img.resize((spine_height, SPINE_WIDTH), Image.LANCZOS)
    spine_img = spine_img_1x.transpose(Image.ROTATE_270)
    spine_np = np.array(spine_img)
    rot_height, rot_width = spine_np.shape[:2]

    pts_src = np.array([[0, 0], [rot_width, 0], [rot_width, rot_height], [0, rot_height]], dtype=np.float32)
    M_spine = cv2.getPerspectiveTransform(pts_src, pts_spine.copy())

    rgb_t, alpha_t = warp_rgba(spine_np, M_spine, out_shape)
    result, alpha_result = compose_spine(result, np.dstack([rgb_t, alpha_t]))

    if serial:
        parts = serial.split('-', 1) if '-' in serial else [serial, '']

        try:    font_r = ImageFont.truetype(font_path, SERIAL_FONT_SIZE * SS)
        except: font_r = ImageFont.load_default()

        draw_tmp = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
        bb1 = draw_tmp.textbbox((0, 0), parts[0], font=font_r)
        bb2 = draw_tmp.textbbox((0, 0), parts[1], font=font_r) if parts[1] else (0, 0, 0, 0)
        w1, h1 = bb1[2] - bb1[0], bb1[3] - bb1[1]
        w2, h2 = (bb2[2] - bb2[0], bb2[3] - bb2[1]) if parts[1] else (0, 0)

        gap = 3 * SS
        total_h = h1 + (gap + h2 if parts[1] else 0)
        serial_w = SPINE_WIDTH * SS
        serial_h = total_h + MARGIN * SS * 2

        img_serial_2x = Image.new('RGBA', (serial_w, serial_h), (0, 0, 0, 0))
        dr = ImageDraw.Draw(img_serial_2x)
        dr.text(((serial_w - w1) // 2, MARGIN * SS), parts[0], font=font_r, fill=SPINE_TEXT_COLOR)
        if parts[1]:
            dr.text(((serial_w - w2) // 2, MARGIN * SS + h1 + gap), parts[1], font=font_r, fill=SPINE_TEXT_COLOR)

        img_serial_2x = img_serial_2x.filter(ImageFilter.GaussianBlur(radius=0.6))
        img_serial = img_serial_2x.resize((SPINE_WIDTH, serial_h // SS), Image.LANCZOS)
        serial_h = img_serial.height
        serial_w = img_serial.width

        frac = (spine_w * serial_h) / (spine_h * float(serial_w))
        SERIAL_OFFSET = 0.02
        v_left = (pts_spine[3] - pts_spine[0]) * (1 - frac - SERIAL_OFFSET)
        v_right = (pts_spine[2] - pts_spine[1]) * (1 - frac - SERIAL_OFFSET)
        v_left_b = (pts_spine[3] - pts_spine[0]) * (1 - SERIAL_OFFSET)
        v_right_b = (pts_spine[2] - pts_spine[1]) * (1 - SERIAL_OFFSET)
        pts_serial_dst = np.array([
            pts_spine[0] + v_left,
            pts_spine[1] + v_right,
            pts_spine[1] + v_right_b,
            pts_spine[0] + v_left_b,
        ], dtype=np.float32)

        serial_np = np.array(img_serial)
        pts_serial_src = np.array([[0, 0], [serial_w, 0], [serial_w, serial_h], [0, serial_h]], dtype=np.float32)
        M_serial = cv2.getPerspectiveTransform(pts_serial_src, pts_serial_dst)
        rgb_r, alpha_r = warp_rgba(serial_np, M_serial, out_shape)
        result, alpha_result = compose_spine(result, np.dstack([rgb_r, alpha_r]))

    result_rgba = cv2.merge([result[:, :, 0], result[:, :, 1], result[:, :, 2], alpha_result])
    return result_rgba


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    game_index = load_game_index(GAME_INDEX)

    existing_3d = existing_3d_serials(COVERS_3D)
    print(f"Found {len(existing_3d)} existing 3D covers in covers/3d/")

    covers = default_covers(COVERS_DEFAULT)
    print(f"Found {len(covers)} default covers in covers/default/")

    skipped = 0
    generated = 0
    errors = []
    generated_serials = []

    for serial, cover_path in sorted(covers):
        if serial in existing_3d:
            skipped += 1
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{serial}.png")
        if os.path.exists(out_path):
            skipped += 1
            continue

        game_name, _ = get_game_info(game_index, serial)
        if not game_name:
            print(f"[SKIP] {serial} — not found in GameIndex.yaml")
            errors.append(serial)
            continue

        print(f"[GEN]  {serial} — {game_name}")
        try:
            result_rgba = render_3d_cover(cover_path, game_name, serial, TEMPLATE, FONT_PATH, FONT_PATH_KOREAN)
            cv2.imwrite(out_path, result_rgba)
            generated_serials.append(serial)
            generated += 1
        except Exception as e:
            print(f"[ERR]  {serial} — {e}")
            errors.append(serial)

    if generated_serials:
        list_path = os.path.join(OUTPUT_DIR, 'generic_3d_cover_list.txt')
        with open(list_path, 'w') as f:
            f.write('\n'.join(generated_serials))

    print()
    print(f"Done! Generated: {generated} | Skipped: {skipped} | Errors: {len(errors)}")
    if errors:
        print(f"Could not process: {errors}")


if __name__ == "__main__":
    main()