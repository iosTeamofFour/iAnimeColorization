from operation.ai import *
from operation.tricks import *


def handle_colorization(pool):
    sketch, points, path = pool
    improved_sketch = sketch.copy()
    improved_sketch = min_resize(improved_sketch, 512)
    improved_sketch = cv_denoise(improved_sketch)
    improved_sketch = sensitive(improved_sketch, s=5.0)
    improved_sketch = go_tail(improved_sketch)

    std = cal_std(improved_sketch)
    if std > 100.0:
        improved_sketch = go_passline(improved_sketch)
        improved_sketch = min_k_down_c(improved_sketch, 2)
        improved_sketch = cv_denoise(improved_sketch)
        improved_sketch = go_tail(improved_sketch)
        improved_sketch = sensitive(improved_sketch, s=5.0)

    improved_sketch = min_black(improved_sketch)
    improved_sketch = cv2.cvtColor(improved_sketch, cv2.COLOR_BGR2GRAY)
    sketch_1024 = k_resize(improved_sketch, 64)
    sketch_256 = mini_norm(k_resize(min_k_down(sketch_1024, 2), 16))
    sketch_128 = hard_norm(sk_resize(min_k_down(sketch_1024, 4), 32))

    baby = go_baby(sketch_128, opreate_normal_hint(ini_hint(sketch_128), points, type=0, length=1))
    baby = de_line(baby, sketch_128)
    for _ in range(16):
        baby = blur_line(baby, sketch_128)
    baby = go_tail(baby)
    baby = clip_15(baby)

    composition = go_gird(sketch=sketch_256, latent=d_resize(baby, sketch_256.shape), hint=ini_hint(sketch_256))
    composition = go_tail(composition)

    painting_function = go_head
    reference = None
    alpha = 0
    result = painting_function(
        sketch=sketch_1024,
        global_hint=k_resize(composition, 14),
        local_hint=opreate_normal_hint(ini_hint(sketch_1024), points, type=2, length=2),
        global_hint_x=k_resize(reference, 14) if reference is not None else k_resize(composition, 14),
        alpha=(1 - alpha) if reference is not None else 1
    )
    result = go_tail(result)
    cv2.imwrite(path, result)

