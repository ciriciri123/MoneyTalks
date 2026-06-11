import io
import os
import zipfile
import tempfile

import bcrypt
from dotenv import load_dotenv
from flask import (Flask, render_template, request, jsonify,
                   send_file, redirect, url_for, flash)
from flask_login import (LoginManager, UserMixin,
                         login_user, logout_user, login_required, current_user)
from flask_wtf.csrf import CSRFProtect
from gtts import gTTS

import supabase_client
from inference import predict_currency, reload_models

load_dotenv()

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-change-me')
app.config['MAX_CONTENT_LENGTH'] = 600 * 1024 * 1024  # 600 MB upload limit

csrf = CSRFProtect(app)

# ── Flask-Login ───────────────────────────────────────────────────────────────

login_manager = LoginManager(app)
login_manager.login_view = 'admin_login'
login_manager.login_message = 'Please log in to access the admin panel.'


class AdminUser(UserMixin):
    def __init__(self, user_id: str, email: str):
        self.id = user_id
        self.email = email


@login_manager.user_loader
def load_user(user_id: str):
    try:
        data = supabase_client.get_admin_by_id(user_id)
        if data:
            return AdminUser(data['id'], data['email'])
    except Exception:
        pass
    return None


# ── Label map ─────────────────────────────────────────────────────────────────

AMOUNTS = {
    'idr_1000':   'Seribu Rupiah',
    'idr_2000':   'Dua Ribu Rupiah',
    'idr_5000':   'Lima Ribu Rupiah',
    'idr_10000':  'Sepuluh Ribu Rupiah',
    'idr_20000':  'Dua Puluh Ribu Rupiah',
    'idr_50000':  'Lima Puluh Ribu Rupiah',
    'idr_100000': 'Seratus Ribu Rupiah',
}

CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', '0.75'))


# ── Guest routes ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/test')
def test_page():
    return render_template('test.html')


@app.route('/api/detect', methods=['POST'])
@csrf.exempt
def detect():
    if 'frame' not in request.files:
        return jsonify({"error": "No frame received"}), 400

    image_bytes = request.files['frame'].read()
    if not image_bytes:
        return jsonify({"error": "Empty frame"}), 400

    result = predict_currency(image_bytes)
    if "error" in result:
        return jsonify({"error": result["error"]}), 500

    label      = result['label']
    confidence = result['confidence']
    friendly   = AMOUNTS.get(label, "Uang tidak dikenali")

    body = {"confidence": confidence, "raw_label": label}

    if confidence >= CONFIDENCE_THRESHOLD:
        body['message'] = friendly
        body['valid']   = True
        if "box" in result:
            body["box"] = result["box"]
    else:
        body['message'] = "Tolong dekatkan uang ke kamera"
        body['valid']   = False

    return jsonify(body)


@app.route('/api/upload-image', methods=['POST'])
@csrf.exempt
def upload_image():
    """Persist a detected scan frame to Supabase (FR-04)."""
    if not supabase_client.is_configured():
        return jsonify({"error": "Supabase not configured"}), 503

    if 'frame' not in request.files:
        return jsonify({"error": "No frame"}), 400

    image_bytes  = request.files['frame'].read()
    denomination = request.form.get('denomination', 'unknown')
    confidence   = float(request.form.get('confidence', 0.0))

    if not image_bytes:
        return jsonify({"error": "Empty frame"}), 400

    try:
        record = supabase_client.upload_scan(image_bytes, denomination, confidence)
        return jsonify({"ok": True, "id": record.get('id')})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/tts')
def tts():
    text = request.args.get('text', '').strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    buf = io.BytesIO()
    gTTS(text=text, lang='id').write_to_fp(buf)
    buf.seek(0)
    return send_file(buf, mimetype='audio/mpeg', as_attachment=False)


# ── Admin: authentication ─────────────────────────────────────────────────────

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if current_user.is_authenticated:
        return redirect(url_for('admin_dashboard'))

    error = None
    if request.method == 'POST':
        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '').encode()

        try:
            admin = supabase_client.get_admin_by_email(email)
            if admin and bcrypt.checkpw(password, admin['password_hash'].encode()):
                login_user(AdminUser(admin['id'], admin['email']), remember=False)
                return redirect(url_for('admin_dashboard'))
            error = "Email atau password salah."
        except Exception as e:
            error = f"Login gagal: {e}"

    return render_template('admin/login.html', error=error)


@app.route('/admin/logout')
@login_required
def admin_logout():
    logout_user()
    return redirect(url_for('admin_login'))


# ── Admin: dashboard ──────────────────────────────────────────────────────────

@app.route('/admin/')
@app.route('/admin')
@login_required
def admin_dashboard():
    stats = {}
    deployed = None
    try:
        stats['total_scans'] = supabase_client.get_scan_count()
        deployed = supabase_client.get_deployed_model()
    except Exception:
        pass

    return render_template('admin/dashboard.html',
                           stats=stats,
                           deployed=deployed,
                           admin=current_user)


# ── Admin: scan images ────────────────────────────────────────────────────────

@app.route('/admin/images')
@login_required
def admin_images():
    page      = int(request.args.get('page', 1))
    from_date = request.args.get('from', '')
    to_date   = request.args.get('to', '')
    per_page  = 20
    offset    = (page - 1) * per_page

    scans = []
    try:
        scans = supabase_client.get_scans(
            from_date=from_date or None,
            to_date=to_date or None,
            limit=per_page,
            offset=offset,
        )
    except Exception:
        pass

    return render_template('admin/images.html',
                           scans=scans,
                           page=page,
                           from_date=from_date,
                           to_date=to_date,
                           per_page=per_page,
                           admin=current_user)


@app.route('/admin/images/view/<path:image_path>')
@login_required
def admin_image_view(image_path: str):
    """Proxy a private Supabase Storage image to the browser."""
    try:
        img_bytes = supabase_client.download_scan_image(image_path)
        return send_file(io.BytesIO(img_bytes), mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@app.route('/admin/images/download')
@login_required
def admin_images_download():
    from_date = request.args.get('from', '')
    to_date   = request.args.get('to', '')

    try:
        scans = supabase_client.get_scans(
            from_date=from_date or None,
            to_date=to_date or None,
            limit=1000,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for scan in scans:
            if not scan.get('image_path'):
                continue
            try:
                img_bytes = supabase_client.download_scan_image(scan['image_path'])
                filename  = scan['image_path'].replace('/', '_')
                zf.writestr(filename, img_bytes)
            except Exception:
                continue

    zip_buf.seek(0)
    label = f"scanned-images-{from_date or 'all'}-{to_date or 'now'}.zip"
    return send_file(zip_buf, mimetype='application/zip',
                     as_attachment=True, download_name=label)


# ── Admin: model versions ─────────────────────────────────────────────────────

@app.route('/admin/models')
@login_required
def admin_models():
    versions = []
    try:
        versions = supabase_client.get_model_versions()
    except Exception:
        pass
    return render_template('admin/models.html',
                           versions=versions,
                           admin=current_user)


@app.route('/admin/models/upload', methods=['POST'])
@login_required
def admin_models_upload():
    if 'model_zip' not in request.files:
        flash('Pilih file .zip model terlebih dahulu.', 'error')
        return redirect(url_for('admin_models'))

    version_string = request.form.get('version_string', '').strip()
    if not version_string:
        flash('Versi model tidak boleh kosong.', 'error')
        return redirect(url_for('admin_models'))

    file_bytes = request.files['model_zip'].read()
    if not file_bytes:
        flash('File kosong.', 'error')
        return redirect(url_for('admin_models'))

    # Validate zip contains the required pkl files
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            names = zf.namelist()
            required = {'bovw_dictionary.pkl', 'svm_model.pkl', 'tfidf_scaler.pkl'}
            if not required.issubset(set(names)):
                missing = required - set(names)
                flash(f'ZIP harus berisi: {", ".join(missing)}', 'error')
                return redirect(url_for('admin_models'))
    except zipfile.BadZipFile:
        flash('File bukan ZIP yang valid.', 'error')
        return redirect(url_for('admin_models'))

    try:
        supabase_client.upload_model_zip(
            file_bytes, version_string, current_user.email
        )
        flash(f'Model v{version_string} berhasil diunggah.', 'success')
    except Exception as e:
        flash(f'Upload gagal: {e}', 'error')

    return redirect(url_for('admin_models'))


@app.route('/admin/models/<version_id>/deploy', methods=['POST'])
@login_required
def admin_models_deploy(version_id: str):
    try:
        # Mark as deployed in DB
        supabase_client.deploy_model_version(version_id)

        # Download zip and hot-swap in-memory models
        version = next(
            (v for v in supabase_client.get_model_versions() if v['id'] == version_id),
            None
        )
        if version:
            zip_bytes = supabase_client.download_model_zip(version['file_path'])
            with tempfile.TemporaryDirectory() as tmp:
                with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                    zf.extractall(tmp)
                reload_models(
                    bovw_path=os.path.join(tmp, 'bovw_dictionary.pkl'),
                    tfidf_path=os.path.join(tmp, 'tfidf_scaler.pkl'),
                    svm_path=os.path.join(tmp, 'svm_model.pkl'),
                )
                # Persist new models to disk so they survive server restarts
                models_dir = os.path.join(os.path.dirname(__file__), 'models')
                import shutil
                for pkl in ('bovw_dictionary.pkl', 'tfidf_scaler.pkl', 'svm_model.pkl'):
                    shutil.copy(os.path.join(tmp, pkl), os.path.join(models_dir, pkl))

        flash('Model berhasil di-deploy dan langsung aktif.', 'success')
    except Exception as e:
        flash(f'Deploy gagal: {e}', 'error')

    return redirect(url_for('admin_models'))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5003)
