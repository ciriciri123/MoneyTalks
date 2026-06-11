import os
import io
import re
from collections import Counter
import cv2
import numpy as np
import joblib
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models', 'proposed')

_models_loaded = False
_kmeans_model = None
_tfidf = None
_svm = None

DENOMINATION_MAP = {
    '100000': 'idr_100000',
    '50000': 'idr_50000',
    '20000': 'idr_20000',
    '10000': 'idr_10000',
    '5000': 'idr_5000',
    '2000': 'idr_2000',
    '1000': 'idr_1000'
}

DENOMINATION_KEYS = sorted(DENOMINATION_MAP.keys(), key=len, reverse=True)

def _normalize_ocr_text(text):
    cleaned = text.replace('\n', ' ').replace('\r', ' ')
    cleaned = re.sub(r'[^0-9 ]+', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def _levenshtein_distance(a, b):
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = curr[j - 1] + 1
            delete_cost = prev[j] + 1
            replace_cost = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(insert_cost, delete_cost, replace_cost))
        prev = curr
    return prev[-1]

def _extract_denom_candidates(cleaned_text):
    if not cleaned_text: return []
    tokens = re.findall(r'\d+', cleaned_text)
    if not tokens: return []
    token_set = set(tokens)
    candidates = set(tokens)
    max_window = min(4, len(tokens))
    for window in range(2, max_window + 1):
        for i in range(len(tokens) - window + 1):
            merged = ''.join(tokens[i:i + window])
            candidates.add(merged)
    matched = []
    for denom in DENOMINATION_KEYS:
        if denom in candidates:
            matched.append(denom)
    filtered = []
    for denom in matched:
        overshadowed = any(
            (other != denom and len(other) > len(denom) and other.startswith(denom))
            for other in matched
        )
        is_direct_token = denom in token_set
        if not overshadowed or is_direct_token:
            filtered.append(denom)
    return filtered

def _extract_fuzzy_denom_scores(cleaned_text):
    if not cleaned_text: return {}
    tokens = re.findall(r'\d+', cleaned_text)
    if not tokens: return {}
    candidates = set(tokens)
    max_window = min(4, len(tokens))
    for window in range(2, max_window + 1):
        for i in range(len(tokens) - window + 1):
            candidates.add(''.join(tokens[i:i + window]))
    fuzzy_scores = Counter()
    for cand in candidates:
        if len(cand) < 4 or len(cand) > 6: continue
        for denom in DENOMINATION_KEYS:
            if abs(len(cand) - len(denom)) > 1: continue
            if denom.startswith(cand) or cand.startswith(denom):
                continue
            dist = _levenshtein_distance(cand, denom)
            if dist == 1:
                if len(cand) == len(denom) and cand[1:] == denom[1:]:
                    continue
                fuzzy_scores[denom] += 0.50
            elif dist == 2 and len(denom) >= 5:
                fuzzy_scores[denom] += 0.20
    return dict(fuzzy_scores)

def _get_ocr_rois(zoom_ocr):
    h, w = zoom_ocr.shape[:2]
    center_roi = zoom_ocr[int(h * 0.20):int(h * 0.80), int(w * 0.20):int(w * 0.80)]
    return [
        ("full", zoom_ocr),
        ("center", center_roi)
    ]

def _denom_from_label(label):
    if not label or not label.startswith('idr_'): return ''
    return label.split('_', 1)[1]

def _predict_with_ocr(ocr_crop):
    h_ocr, w_ocr = ocr_crop.shape[:2]
    zoom_factor = 2 
    zoom_ocr = cv2.resize(
        ocr_crop,
        (w_ocr * zoom_factor, h_ocr * zoom_factor),
        interpolation=cv2.INTER_LINEAR
    )
    
    ocr_configs = ['--psm 11']

    candidate_score = Counter()
    candidate_hits = Counter()
    pass_count = 0
    digit_pass_count = 0
    candidate_pass_count = 0

    # OPTIMASI SPEED: Hapus CLAHE untuk OCR, cukup Gray dan Otsu Threshold
    for roi_name, roi_img in _get_ocr_rois(zoom_ocr):
        gray_ocr = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        _, otsu_bin = cv2.threshold(gray_ocr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        preprocessed_images = [
            ("gray", gray_ocr),
            ("otsu", otsu_bin) 
        ]

        for prep_name, ocr_img in preprocessed_images:
            for config in ocr_configs:
                try:
                    ocr_text = pytesseract.image_to_string(ocr_img, config=config)
                    cleaned_text = _normalize_ocr_text(ocr_text)
                    pass_count += 1

                    if cleaned_text:
                        digit_pass_count += 1

                    matched_denoms = _extract_denom_candidates(cleaned_text)
                    if matched_denoms:
                        candidate_pass_count += 1
                        tokens = re.findall(r'\d+', cleaned_text)
                        token_set = set(tokens)
                        collapsed = ''.join(tokens)

                        for denom in set(matched_denoms):
                            score = 1.0
                            if denom in token_set:
                                score += 0.9
                                score += 0.5
                            elif denom in collapsed: score += 0.35
                            score += 0.12 * max(len(denom) - 4, 0)

                            candidate_score[denom] += score
                            candidate_hits[denom] += 1

                    fuzzy_scores = _extract_fuzzy_denom_scores(cleaned_text)
                    for denom, fuzzy_score in fuzzy_scores.items():
                        candidate_score[denom] += fuzzy_score
                except Exception:
                    continue

    if not candidate_score:
        return None, 0.0, pass_count

    best_denom = max(
        candidate_score,
        key=lambda d: (candidate_hits[d], candidate_score[d])
    )
    hits = candidate_hits[best_denom]
    vote_ratio = hits / max(candidate_pass_count, 1)
    ocr_label = DENOMINATION_MAP[best_denom]

    return ocr_label, float(vote_ratio), pass_count

def load_models():
    global _models_loaded, _kmeans_model, _tfidf, _svm
    if _models_loaded: return
    bovw_path = os.path.join(MODELS_DIR, 'bovw_dictionary.pkl')
    tfidf_path = os.path.join(MODELS_DIR, 'tfidf_scaler.pkl')
    svm_path = os.path.join(MODELS_DIR, 'svm_model.pkl')
    
    _kmeans_model = joblib.load(bovw_path)
    _tfidf = joblib.load(tfidf_path)
    _svm = joblib.load(svm_path)
    _models_loaded = True

def preprocess_image(image_bytes):
    np_img = np.frombuffer(image_bytes, np.uint8)
    cv_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if cv_img is None:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return cv_img

def get_orb_and_color_features(img, max_features=2000):
    H, W = img.shape[:2]
    orb = cv2.ORB_create(nfeatures=max_features)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    keypoints_rough, _ = orb.detectAndCompute(blurred, None)
    
    box_coords = None
    money_crop = img
    
    if keypoints_rough and len(keypoints_rough) >= 30:
        pts = np.array([kp.pt for kp in keypoints_rough])
        
        x_min, y_min = np.percentile(pts, 8, axis=0) 
        x_max, y_max = np.percentile(pts, 92, axis=0)
        
        pad_x = int((x_max - x_min) * 0.18)
        pad_y = int((y_max - y_min) * 0.18)
        
        x1 = max(0, int(x_min) - pad_x)
        y1 = max(0, int(y_min) - pad_y)
        x2 = min(W, int(x_max) + pad_x)
        y2 = min(H, int(y_max) + pad_y)
        
        money_crop = img[y1:y2, x1:x2]
        box_coords = [float(x1/W), float(y1/H), float((x2-x1)/W), float((y2-y1)/H)]
            
    if money_crop.shape[0] > money_crop.shape[1]:
        money_crop = cv2.rotate(money_crop, cv2.ROTATE_90_CLOCKWISE)
        
    ocr_crop = money_crop.copy()
    svm_crop = cv2.resize(money_crop, (800, 400))
    
    gray_svm = cv2.cvtColor(svm_crop, cv2.COLOR_BGR2GRAY)
    enhanced_svm = clahe.apply(gray_svm)
    blurred_svm = cv2.GaussianBlur(enhanced_svm, (5, 5), 0)
    
    final_keypoints, final_descriptors = orb.detectAndCompute(blurred_svm, None)
    if final_descriptors is None: final_descriptors = np.array([])
        
    hsv = cv2.cvtColor(svm_crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist) 
    color_hist = hist.flatten()
        
    return final_keypoints, final_descriptors, color_hist, box_coords, ocr_crop

def get_bovw_histogram(descriptors, kmeans_model):
    num_clusters = kmeans_model.n_clusters
    if len(descriptors) > 0:
        words = kmeans_model.predict(descriptors)
        hist, _ = np.histogram(words, bins=np.arange(num_clusters + 1))
    else:
        hist = np.zeros(num_clusters)
    return hist

def predict_currency(image_bytes):
    if not _models_loaded: load_models()

    try:
        img = preprocess_image(image_bytes)
        keypoints, descriptors, color_hist, box_coords, ocr_crop = get_orb_and_color_features(img)

        if len(descriptors) < 50:
            return {"label": "none", "confidence": 0.0}

        # --- 1. PREDIKSI SVM ---
        bovw_raw = get_bovw_histogram(descriptors, _kmeans_model)
        bovw_tfidf = _tfidf.transform([bovw_raw]).toarray()[0]
        bovw_tfidf = bovw_tfidf * 3.0
        
        fused = np.hstack((bovw_tfidf, color_hist))
        raw_label = _svm.predict([fused])[0]
        svm_label = str(raw_label)
        
        proba = _svm.predict_proba([fused])[0]
        confidence = float(np.max(proba))
        sorted_proba = np.sort(proba)
        svm_margin = float(sorted_proba[-1] - sorted_proba[-2]) if len(sorted_proba) >= 2 else float(confidence)

        # --- 2. PREDIKSI OCR ---
        ocr_label = None
        ocr_vote_ratio = 0.0
        try:
            ocr_label_1, ocr_vote_ratio_1, ocr_pass_count_1 = _predict_with_ocr(ocr_crop)
            
            if ocr_vote_ratio_1 < 0.25:
                ocr_crop_flipped = cv2.rotate(ocr_crop, cv2.ROTATE_180)
                ocr_label_2, ocr_vote_ratio_2, ocr_pass_count_2 = _predict_with_ocr(ocr_crop_flipped)
                
                if ocr_label_2 and ocr_vote_ratio_2 > ocr_vote_ratio_1:
                    ocr_label = ocr_label_2
                    ocr_vote_ratio = ocr_vote_ratio_2
                else:
                    ocr_label = ocr_label_1
                    ocr_vote_ratio = ocr_vote_ratio_1
            else:
                ocr_label = ocr_label_1
                ocr_vote_ratio = ocr_vote_ratio_1

            if (not ocr_label or ocr_vote_ratio < 0.30) and ocr_crop is not img:
                full_frame_crop = img.copy()
                if full_frame_crop.shape[0] > full_frame_crop.shape[1]:
                    full_frame_crop = cv2.rotate(full_frame_crop, cv2.ROTATE_90_CLOCKWISE)
                ocr_label_ff, ocr_vote_ratio_ff, _ = _predict_with_ocr(full_frame_crop)
                ocr_vote_ratio_ff_adj = ocr_vote_ratio_ff * 0.70  # noise penalty
                if ocr_label_ff and ocr_vote_ratio_ff_adj > (ocr_vote_ratio or 0):
                    ocr_label = ocr_label_ff
                    ocr_vote_ratio = ocr_vote_ratio_ff_adj
                    print(f"[OCR] Full-frame fallback used: {ocr_label_ff} (ratio={ocr_vote_ratio_ff:.2f})")
                
        except Exception as e:
            print(f"[OCR WARNING] {e}")

        # --- 3. DECISION FUSION ---
        final_label = svm_label
        
        if ocr_label:
            if ocr_label == svm_label:
                confidence = min(0.995, max(confidence, 0.72 + (0.25 * ocr_vote_ratio)))
            else:
                if (svm_label == 'idr_2000' and ocr_label == 'idr_10000') or \
                   (svm_label == 'idr_10000' and ocr_label == 'idr_2000'):
                    if ocr_vote_ratio >= 0.85 and confidence < 0.60:
                        final_label = ocr_label
                        confidence = 0.76

                elif (svm_label == 'idr_10000' and ocr_label == 'idr_100000') or \
                     (svm_label == 'idr_100000' and ocr_label == 'idr_10000'):
                    if ocr_vote_ratio >= 0.90 and confidence < 0.55:
                        final_label = ocr_label
                        confidence = 0.76

                elif (svm_label == 'idr_2000' and ocr_label == 'idr_100000') or \
                     (svm_label == 'idr_100000' and ocr_label == 'idr_2000'):
                    if ocr_vote_ratio >= 0.95 and confidence < 0.50:
                        final_label = ocr_label
                        confidence = 0.76

                elif (svm_label == 'idr_1000' and ocr_label == 'idr_100000') or \
                     (svm_label == 'idr_100000' and ocr_label == 'idr_1000'):
                    if ocr_vote_ratio >= 0.95 and confidence < 0.50:
                        final_label = ocr_label
                        confidence = 0.76

                elif (svm_label == 'idr_1000' and ocr_label == 'idr_50000') or \
                     (svm_label == 'idr_50000' and ocr_label == 'idr_1000'):
                    if ocr_vote_ratio >= 0.60 and confidence < 0.80:
                        final_label = ocr_label
                        confidence = min(0.90, max(confidence, 0.65 + (0.25 * ocr_vote_ratio)))

                elif (svm_label == 'idr_2000' and ocr_label == 'idr_50000') or \
                     (svm_label == 'idr_50000' and ocr_label == 'idr_2000'):
                    if ocr_vote_ratio >= 0.60 and confidence < 0.80:
                        final_label = ocr_label
                        confidence = min(0.90, max(confidence, 0.65 + (0.25 * ocr_vote_ratio)))

                elif (svm_label == 'idr_1000' and ocr_label == 'idr_20000') or \
                     (svm_label == 'idr_20000' and ocr_label == 'idr_1000') or \
                     (svm_label == 'idr_2000' and ocr_label == 'idr_20000') or \
                     (svm_label == 'idr_20000' and ocr_label == 'idr_2000'):
                    if ocr_vote_ratio >= 0.65 and confidence < 0.75:
                        final_label = ocr_label
                        confidence = min(0.90, max(confidence, 0.65 + (0.25 * ocr_vote_ratio)))

                elif (svm_label == 'idr_5000' and ocr_label == 'idr_10000') or \
                     (svm_label == 'idr_10000' and ocr_label == 'idr_5000'):
                    if ocr_vote_ratio >= 0.55 and confidence < 0.82:
                        final_label = ocr_label
                        confidence = min(0.90, max(confidence, 0.65 + (0.25 * ocr_vote_ratio)))

                elif (svm_label == 'idr_10000' and ocr_label == 'idr_50000') or \
                     (svm_label == 'idr_50000' and ocr_label == 'idr_10000'):
                    if ocr_vote_ratio >= 0.55 and confidence < 0.82:
                        final_label = ocr_label
                        confidence = min(0.90, max(confidence, 0.65 + (0.25 * ocr_vote_ratio)))

                elif (svm_label == 'idr_5000' and ocr_label == 'idr_2000') or \
                     (svm_label == 'idr_2000' and ocr_label == 'idr_5000') or \
                     (svm_label == 'idr_5000' and ocr_label == 'idr_1000') or \
                     (svm_label == 'idr_1000' and ocr_label == 'idr_5000'):
                    if ocr_vote_ratio >= 0.60 and confidence < 0.78:
                        final_label = ocr_label
                        confidence = min(0.90, max(confidence, 0.63 + (0.25 * ocr_vote_ratio)))

                else:
                    ocr_proposes_100k = (ocr_label == 'idr_100000')
                    conf_threshold   = 0.55 if ocr_proposes_100k else 0.65
                    vote_threshold   = 0.75 if ocr_proposes_100k else 0.60
                    margin_threshold = 0.10 if ocr_proposes_100k else 0.15
                    vote_margin_thr  = 0.65 if ocr_proposes_100k else 0.50

                    if confidence < conf_threshold and ocr_vote_ratio >= vote_threshold:
                        final_label = ocr_label
                        confidence = min(0.95, max(confidence, 0.60 + (0.35 * ocr_vote_ratio)))
                    elif svm_margin < margin_threshold and ocr_vote_ratio >= vote_margin_thr:
                        final_label = ocr_label
                        confidence = min(0.90, max(confidence, 0.58 + (0.40 * ocr_vote_ratio)))

        confidence = float(min(max(confidence, 0.01), 0.995))

        result = {
            'label': final_label,
            'confidence': float(confidence)
        }
        if box_coords: result['box'] = box_coords
            
        return result
        
    except Exception as e:
        return {"error": str(e)}