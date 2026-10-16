def iou(a, b):
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])

    inter = max(0, right - left) * max(0, bottom - top)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union


def match_boxes(true_boxes, pred_boxes, pred_confs, iou_threshold=0.5):
    order = sorted(range(len(pred_boxes)), key=lambda i: pred_confs[i], reverse=True)

    matched_true = set()
    tp, fp = [], []

    for i in order:
        best_iou, best_j = 0, None
        for j, t in enumerate(true_boxes):
            if j in matched_true:
                continue
            value = iou(pred_boxes[i], t)
            if value > best_iou:
                best_iou, best_j = value, j

        if best_j is not None and best_iou >= iou_threshold:
            matched_true.add(best_j)
            tp.append((i, best_j, best_iou))
        else:
            fp.append(i)

    fn = [j for j in range(len(true_boxes)) if j not in matched_true]
    return tp, fp, fn