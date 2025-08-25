import numpy as np, cv2

def _safe_pt(kpts, key):
    v = kpts.get(key, None)
    if v is None: return None
    x, y = v
    return np.array([float(x), float(y)])

def build_dual_tri_roi_masks(kpts, img_shape,
                             roi_chest_down_ratio=0.28,
                             neck_up_ratio=0.12,
                             shoulder_inset_ratio=0.15):
    H, W = img_shape[:2]
    ls = np.array(kpts['left_shoulder'], dtype=float)
    rs = np.array(kpts['right_shoulder'], dtype=float)
    mid = (ls + rs) / 2.0
    shoulder_w = np.linalg.norm(ls - rs) + 1e-6

    def inward(p, to, r): return p + r * (to - p)
    ls_in = inward(ls, mid, shoulder_inset_ratio)
    rs_in = inward(rs, mid, shoulder_inset_ratio)

    chest = mid + np.array([0.0, roi_chest_down_ratio * shoulder_w])
    neck  = mid - np.array([0.0, neck_up_ratio        * shoulder_w])

    tri_down = np.vstack([ls_in, rs_in, chest]).astype(np.int32)
    tri_up   = np.vstack([ls_in, rs_in, neck ]).astype(np.int32)

    m_down = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(m_down, tri_down, 1)
    m_up   = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(m_up,   tri_up,   1)
    return m_down, m_up, tri_down, tri_up

def _circle_mask(shape, center, radius):
    H, W = shape[:2]
    yy, xx = np.ogrid[:H, :W]
    cx, cy = center
    return ((xx - cx)**2 + (yy - cy)**2) <= radius*radius

def sample_face_skin_region(img_bgr, kpts, radius_ratio=0.28):
    H, W = img_bgr.shape[:2]
    ls = _safe_pt(kpts, 'left_shoulder'); rs = _safe_pt(kpts, 'right_shoulder')
    nose = _safe_pt(kpts, 'nose'); le = _safe_pt(kpts, 'left_eye'); re = _safe_pt(kpts, 'right_eye')

    if ls is not None and rs is not None:
        mid = (ls + rs)/2.0
        shoulder_w = np.linalg.norm(ls - rs) + 1e-6
    else:
        mid = np.array([W/2, H/2]); shoulder_w = min(W, H)/4.0

    pts = [p for p in [nose, le, re] if p is not None]
    center = np.mean(pts, axis=0) if len(pts)>0 else (mid - np.array([0, 0.6*shoulder_w]))

    r = float(radius_ratio) * shoulder_w
    full = _circle_mask((H, W, 3), center[::-1], r)
    cy = int(center[1])
    lower = np.zeros_like(full, np.uint8)
    lower[cy:, :] = full[cy:, :].astype(np.uint8)
    lower = cv2.erode(lower.astype(np.uint8), np.ones((3,3), np.uint8), 1)
    ys, xs = np.where(lower > 0)
    pixels = img_bgr[ys, xs] if len(xs)>0 else np.empty((0,3), dtype=np.uint8)
    return lower, pixels

def _to_lab_ab(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return lab[...,1:3].astype(np.float32)

def fit_skin_color_gaussian(img_bgr, kpts, radius_ratio=0.28):
    mask, pix = sample_face_skin_region(img_bgr, kpts, radius_ratio)
    if pix.shape[0] < 50: return None, None, mask
    lab = cv2.cvtColor(pix.reshape(-1,1,3), cv2.COLOR_BGR2LAB).reshape(-1,3)
    ab = lab[:,1:2+1].astype(np.float32)
    mu = np.median(ab, axis=0)
    dif = ab - mu
    cov = np.cov(dif.T) + np.eye(2, dtype=np.float32)*1e-3
    return mu, cov, mask

def mahalanobis_distance_ab(img_bgr, mu, cov):
    H, W = img_bgr.shape[:2]
    inv = np.linalg.inv(cov)
    ab = _to_lab_ab(img_bgr).reshape(-1,2)
    dif = ab - mu[None,:]
    m = np.einsum('ij,ij->i', dif @ inv, dif)
    return np.sqrt(np.maximum(m,0)).reshape(H, W)

def filter_mask_by_skincolor(img_bgr, mask_bin, mu, cov, thresh=4.0, mode='component'):
    if mu is None or cov is None: return mask_bin
    dist = mahalanobis_distance_ab(img_bgr, mu, cov)
    if mode=='pixel':
        return (mask_bin.astype(np.uint8) & (dist<thresh).astype(np.uint8)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin.astype(np.uint8), 8)
    out = np.zeros_like(mask_bin, np.uint8)
    for i in range(1, n):
        x,y,w,h,area = stats[i]
        if area < 8: continue
        comp = (labels==i).astype(np.uint8)
        dvals = dist[y:y+h, x:x+w][comp[y:y+h,x:x+w]>0]
        if dvals.size == 0: continue
        if np.median(dvals) < thresh:
            out[y:y+h, x:x+w][comp[y:y+h,x:x+w]>0] = 1
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    return out

def _shoulder_line_and_axis(kpts):
    ls = _safe_pt(kpts, 'left_shoulder'); rs = _safe_pt(kpts, 'right_shoulder')
    if ls is None or rs is None: return None, None, None, None
    mid = (ls + rs)/2.0
    v = rs - ls; L = np.linalg.norm(v) + 1e-6
    axis = np.array([-v[1], v[0]])/L
    return ls, rs, mid, axis

def _apex_angle(poly):
    if poly.shape[0] < 3: return None, None
    def angle(p, a, b):
        v1 = p - a; v2 = b - a
        c = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)
        c = np.clip(c, -1, 1)
        return np.degrees(np.arccos(c))
    k = len(poly); angs=[]
    for i in range(k):
        a = poly[i]; b = poly[(i-1)%k]; c = poly[(i+1)%k]
        angs.append(angle(b, a, c))
    idx = int(np.argmin(angs)); return float(angs[idx]), idx

def extract_small_v_subtriangle(img_bgr, kpts, roi_mask, mu=None, cov=None,
                                color_thresh=4.0, min_area_px=24,
                                max_area_ratio=0.035, prefer_up_or_down='auto'):
    H, W = img_bgr.shape[:2]
    cand = roi_mask.astype(np.uint8)
    if (mu is not None) and (cov is not None):
        dist = mahalanobis_distance_ab(img_bgr, mu, cov)
        cand = (cand & (dist < float(color_thresh))).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    best = np.zeros_like(cand, np.uint8); best_score=-1
    if n<=1: return best

    ls, rs, mid, axis = _shoulder_line_and_axis(kpts)
    total = H*W
    for i in range(1, n):
        x,y,w,h,area = stats[i]
        if area < min_area_px or area > max_area_ratio*total: continue
        comp = (labels==i).astype(np.uint8)
        cnts,_ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        cnt = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon=0.04*peri + 1e-6, closed=True).reshape(-1,2)
        k = len(approx)
        tri_like = 1.0/(1.0+abs(3-k))
        ang, idx = _apex_angle(approx) if k>=3 else (None,None)
        angle_ok = 0.0
        if ang is not None and 25<=ang<=95:
            angle_ok = 1.0 - abs(60 - ang)/60.0
        orient_ok = 1.0
        if axis is not None and ang is not None:
            apex = approx[idx]; vec = apex - mid
            proj = float(np.dot(vec, axis))
            if   prefer_up_or_down=='down': orient_ok = 1.0 if proj>0 else 0.2
            elif prefer_up_or_down=='up'  : orient_ok = 1.0 if proj<0 else 0.2
        small_pref = 1.0 - (area / (max_area_ratio*total + 1e-6))
        score = 0.45*tri_like + 0.25*angle_ok + 0.20*orient_ok + 0.10*small_pref
        if score > best_score: best_score=score; best=comp
    return best

def split_instances_with_pose(kpts_list):
    out=[]
    for k in kpts_list:
        ls=_safe_pt(k,'left_shoulder'); rs=_safe_pt(k,'right_shoulder')
        if ls is None or rs is None: continue
        sw = np.linalg.norm(ls-rs)
        x0=int(max(0, min(ls[0],rs[0]) - 0.6*sw))
        x1=int(max(ls[0],rs[0]) + 0.6*sw)
        y0=int(max(0, min(ls[1],rs[1]) - 1.2*sw))
        y1=int(max(ls[1],rs[1]) + 1.6*sw)
        out.append({'kpts':k, 'bbox':(x0,y0,x1,y1)})
    return out