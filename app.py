from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, Response
from flask_session import Session
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, f1_score, make_scorer, recall_score, roc_auc_score, roc_curve, mean_squared_error, r2_score, mean_absolute_error, explained_variance_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.utils.multiclass import type_of_target
from tpot import TPOTClassifier, TPOTRegressor
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import matplotlib.pyplot as plt
import io
import os
import base64
import pickle
import json
import os
import tempfile
from datetime import datetime
from datetime import timedelta
import time
from functools import wraps
import firebase_admin 
from firebase_admin import credentials, auth, firestore, initialize_app, storage
from google.cloud.firestore_v1.base_query import FieldFilter
from google.cloud import storage as gcs_storage
import stripe

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlxs', 'xls'}
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = True      

app.config.update(
    SESSION_TYPE='filesystem',
    SESSION_FILE_DIR=tempfile.mkdtemp(),
    SESSION_PERMANENT=False,
    PERMANENT_SESSION_LIFETIME=timedelta(hours=1),
    MAX_CONTENT_LENGTH=16 * 1024 * 1024
)
Session(app)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

firebase_json = os.getenv("FIREBASE_CREDENTIALS_JSON")

if firebase_json:
    cred_dict = json.loads(firebase_json)
    cred = credentials.Certificate(cred_dict)
else:
    firebase_key_path = os.getenv("FIREBASE_KEY_PATH", "firebase_key.json")
    cred = credentials.Certificate(firebase_key_path)

firebase_admin.initialize_app(cred, {
    'storageBucket': os.getenv("FIREBASE_STORAGE_BUCKET", "fypautoml.firebasestorage.app"),
    'projectId': os.getenv("FIREBASE_PROJECT_ID", "fypautoml")
})

db = firestore.client()
bucket = storage.bucket()

stripe.api_key = os.getenv('STRIPE_SECRET_KEY', 'sk_test_51RSADZIEZSCBrBXvwlRklvjYUAXqfgKGxX6y02Y8Uzihwr0k7myIwlVV9gwQCFEFMWOI8ahGaiEK7mkBuNRf9H5H00vjcHSPKV')
stripe_webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET', 'whsec_Z0QgZ3Ck7bgf4Vs9KcKOYEzkMd9AgZ29')
IS_CLOUD = os.getenv("KOYEB_PUBLIC_DOMAIN") is not None

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'pdf'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_dataset_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
    
def detect_task_type(y):
    target_type = type_of_target(y)
    
    if target_type in ['binary', 'multiclass']:
        return 'classification'
    
    unique_values = sorted(y.unique())
    if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
        return 'classification'

    if target_type in ['continuous', 'continuous-multioutput']:
        return 'regression'

    raise ValueError(f"Unsupported target type: {target_type}")

def calculate_classification_metrics(confusion_matrix, classes, y_true=None, y_pred_proba=None):
    """Calculate specificity and sensitivity from confusion matrix"""
    metrics = {}
    sensitivities = []
    specificities = []
    
    # For all classification types (binary and multi-class)
    for i in range(len(classes)):
        tp = confusion_matrix[i][i]
        fn = sum(confusion_matrix[i]) - tp
        fp = sum(row[i] for row in confusion_matrix) - tp
        tn = sum(sum(row) for row in confusion_matrix) - tp - fp - fn
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        sensitivities.append(sensitivity)
        specificities.append(specificity)
    
    metrics.update({
        'sensitivity': np.mean(sensitivities),
        'specificity': np.mean(specificities),
    })
    
    # Calculate AUC if probabilities are available
    if y_pred_proba is not None and y_true is not None:
        try:
            if len(classes) == 2:  # Binary classification
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:  # Multiclass classification
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
        except Exception as e:
            print(f"Could not calculate AUC: {str(e)}")
            metrics['auc'] = None
    else:
        metrics['auc'] = None

    return metrics

def calculate_regression_metrics(y_true, y_pred):
    """Calculate regression-specific metrics"""
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'explained_variance': explained_variance_score(y_true, y_pred)
    }

def evaluate_model_cv(model, X, y, cv=5, task_type='classification'):
                             
    # Get confusion matrix from one of the CV splits
    model.fit(X, y)  # Fit on full data for confusion matrix

    if task_type == 'classification':
        y_pred = model.predict(X)

        # Get predicted probabilities if available
        y_pred_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None

        # Ensure y is properly encoded for confusion matrix
        if isinstance(y, (pd.Series, pd.DataFrame)):  # Check if pandas object
            if y.dtype == 'object' or isinstance(y.iloc[0], str):
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                y_pred_encoded = le.transform(y_pred)
                cm = confusion_matrix(y_encoded, y_pred_encoded)
                classes = le.classes_.tolist()
            else:
                y_encoded = y
                cm = confusion_matrix(y, y_pred)
                classes = sorted(np.unique(y).tolist())
        else:  # Handle numpy arrays
            if y.dtype == object or isinstance(y[0], str):
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                y_pred_encoded = le.transform(y_pred)
                cm = confusion_matrix(y_encoded, y_pred_encoded)
                classes = le.classes_.tolist()
            else:
                y_encoded = y
                cm = confusion_matrix(y, y_pred)
                classes = sorted(np.unique(y).tolist())

        additional_metrics = calculate_classification_metrics(cm, classes, y_true=y_encoded, y_pred_proba=y_pred_proba)

        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='weighted', zero_division=0),
            'f1_score': make_scorer(f1_score, average='weighted'),
            'recall': make_scorer(recall_score, average='weighted', zero_division=0)
        }

        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)

        return {
            'accuracy': np.mean(cv_results['test_accuracy']),
            'precision': np.mean(cv_results['test_precision']),
            'f1_score': np.mean(cv_results['test_f1_score']),
            'sensitivity': np.mean(cv_results['test_recall']),  
            'specificity': additional_metrics['specificity'],
            'auc': additional_metrics['auc'], 
            'cv_accuracy': cv_results['test_accuracy'].tolist(),
            'cv_precision': cv_results['test_precision'].tolist(),
            'cv_f1_score': cv_results['test_f1_score'].tolist(),
            'cv_recall': cv_results['test_recall'].tolist(),
            'cv_folds': cv,
            'confusion_matrix': cm.tolist(),  
            'classes': classes, 
            'task_type' : 'classification'
            }
    
    else:
        y_pred = model.predict(X)
        metrics = calculate_regression_metrics(y, y_pred)
        
        scoring = {
            'neg_mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
            'r2': make_scorer(r2_score),
            'neg_mean_absolute_error': make_scorer(mean_absolute_error, greater_is_better=False),
            'explained_variance': make_scorer(explained_variance_score)
        }

        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
        
        # Convert negative MSE and MAE back to positive
        cv_results['test_mean_squared_error'] = -cv_results['test_neg_mean_squared_error']
        cv_results['test_mean_absolute_error'] = -cv_results['test_neg_mean_absolute_error']
        
        return {
            'mse': np.mean(cv_results['test_mean_squared_error']),
            'rmse': np.sqrt(np.mean(cv_results['test_mean_squared_error'])),
            'mae': np.mean(cv_results['test_mean_absolute_error']),
            'r2': np.mean(cv_results['test_r2']),
            'explained_variance': np.mean(cv_results['test_explained_variance']),
            'cv_mse': cv_results['test_mean_squared_error'].tolist(),
            'cv_rmse': np.sqrt(cv_results['test_mean_squared_error']).tolist(),
            'cv_mae': cv_results['test_mean_absolute_error'].tolist(),
            'cv_r2': cv_results['test_r2'].tolist(),
            'cv_folds': cv,
            'task_type': 'regression'
        }

def create_feature_importance_plot(importances, feature_names, title):
    # Normalize importances to 0-1 range
    importances = np.array(importances)
    if importances.min() < 0:
        # For permutation importance which can have negative values
        importances = (importances - importances.min()) / (importances.max() - importances.min())
    else:
        importances = importances / importances.max()
    
    # Sort features by importance
    indices = np.argsort(importances)
    features = [feature_names[i] for i in indices]
    importance_values = importances[indices]
    
    # Create plot with improved styling
    plt.figure(figsize=(10, max(6, len(features)*0.3)))
    bars = plt.barh(range(len(features)), importance_values, 
                   align='center', 
                   color='#1f77b4',
                   alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(importance_values):
        plt.text(v + 0.01, i, f"{v:.2f}", color='black', va='center')
    
    plt.yticks(range(len(features)), features)
    plt.title(title, pad=20)
    plt.xlabel('Normalized Importance Score')
    plt.xlim(0, 1.1)
    
    # Add grid
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Remove spines
    for spine in ['top', 'right', 'bottom']:
        plt.gca().spines[spine].set_visible(False)
    
    plt.tight_layout()
    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    # Encode as base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def get_tpot_feature_importance(tpot_model, X, y):
    try:
        # Get the final estimator from the pipeline
        final_estimator = tpot_model.fitted_pipeline_.steps[-1][1]
        
        # Try different methods to get feature importance
        if hasattr(final_estimator, 'feature_importances_'):
            return final_estimator.feature_importances_
        elif hasattr(final_estimator, 'coef_'):
            return np.mean(np.abs(final_estimator.coef_), axis=0)
        else:
            # Fall back to permutation importance
            result = permutation_importance(
                final_estimator, X, y, n_repeats=10, random_state=42
            )
            return result.importances_mean
    except Exception as e:
        print(f"Error getting TPOT feature importance: {str(e)}")
        # Return uniform importance if we can't calculate
        return np.ones(X.shape[1]) / X.shape[1]
    
def handle_large_session(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except RequestEntityTooLarge:
            flash('Session data too large. Please use smaller datasets or fewer algorithms.')
            return redirect(url_for('index'))
    return decorated_function

def verify_firebase_token(id_token):
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token
    except Exception as e:
        print("Token verification failed:", e)
        return None

@app.route('/')
def home():
    return render_template('register.html')

@app.route('/login_token', methods=['POST'])
def login_token():
    data = request.get_json()
    id_token = data.get('idToken')

    if not id_token:
        return jsonify({'status': 'error', 'message': 'Missing ID token'}), 400

    try:
        decoded = auth.verify_id_token(id_token, clock_skew_seconds=10)
        uid = decoded['uid']

        user_ref = db.collection('Users').document(uid)
        user_doc = user_ref.get()

        if not user_doc.exists:
            return jsonify({'status': 'pending'})
        
        user_data = user_doc.to_dict()
        status = user_data.get('status', 'pending')
        session['uid'] = uid
        session['license_type'] = user_data.get('license_type', 1)
        print("[/login_token] Session content after setting license_type:", dict(session))

        if status == 'approved':
            return jsonify({'status': 'approved', 'redirect': '/index'})
        elif status == 'waiting_review':
            return jsonify({'status': 'waiting_review', 'redirect': '/waiting_page'})
        elif status == 'rejected':
            return jsonify({'status': 'rejected', 'redirect': '/upload_receipt', 'message': 'Your receipt was rejected. Please upload a valid one for verification.'})
        else:
            return jsonify({'status': 'pending', 'redirect': '/select_license'})
        
    except Exception as e:

        import traceback
        print("Redirect check failed:", e)
        traceback.print_exc()
        
        print("[/login_token] Session content after setting license_type:", dict(session))
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/select_license', methods=['GET', 'POST'])
def select_license():
    if request.method == 'GET':
        return render_template('select_license.html')
    
    elif request.method == 'POST':
        uid = request.form.get('uid')
        license_type = request.form.get('license_type_id')

        if not uid or not license_type:
            return "Missing UID or License Type", 400
        
        try:
            user_ref = db.collection('Users').document(uid)
            user_ref.update({
                'license_type': license_type,
                'status': 'pending'
            })

            db.collection('Payments').add({
                'uid': uid,
                'license_type': license_type,
                'status': 'pending',
                'created_at': datetime.utcnow()
            })

            # Store in session
            session['user_id'] = uid
            session['license_type_id'] = license_type

            session['payment_id'] = datetime.utcnow

            return redirect(url_for('upload_receipt'))
        
        except Exception as e:
            print("DB Error:", e)
            return "Database error", 500
        
@app.route('/create-checkout-session', methods=['POST'])
def create_checkout_session():
    data = request.get_json()
    
    if not data:
            print("Error: No JSON data provided in request")
            return jsonify({'error': 'No JSON data provided'}), 400
    
    uid = data.get('uid')
    license_type = data.get('license_type')

    print(f"Received request with uid: {uid}, license_type: {license_type}")

    # Validate inputs
    if not uid:
        print("Error: Missing uid in request")
        return jsonify({'error': 'User ID is required'}), 400
    if not license_type:
        print("Error: Missing license_type in request")
        return jsonify({'error': 'License type is required'}), 400

    price_map = {
        '1': 'price_1RUOeQIEZSCBrBXvv0dxdm3t',  # Normal License $10.00
        '2': 'price_1RUOeuIEZSCBrBXvGv8yYG3S',  # Advanced License $15.00
    }
    price_id = price_map.get(license_type)
    if not price_id:
        print(f"Error: Invalid license type. License Type: {license_type}")
        return jsonify({'error': 'Invalid license type'}), 400

    try:
        print("Creating Stripe checkout session...")
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': price_id,
                'quantity': 1, 
            }],
            mode='subscription',
            success_url=f"https://fypproject-945914686130.asia-southeast3.run.app/payment-success?session_id={{CHECKOUT_SESSION_ID}}&uid={uid}&license_type={license_type}",
            cancel_url="https://fypproject-945914686130.asia-southeast3.run.app/payment-cancelled",
            metadata={
                'uid': uid,
                'license_type': license_type,
            }
        )
        print("Checkout session created successfully:", session.id)
        return jsonify({'id': session.id})
    except Exception as e:
        print("Error while creating checkout session:", e)
        return jsonify(error=str(e)), 500

@app.route('/payment-success')
def payment_success():
    session_id = request.args.get('session_id')
    uid = request.args.get('uid')
    license_type = request.args.get('license_type')

    # Retrieve session to verify
    try:
        session = stripe.checkout.Session.retrieve(session_id)


        # Log the session object for debugging
        print(f"Stripe session: {session}")  # Debugging line

        invoice_url = None
        if session.invoice:
            invoice_obj = stripe.Invoice.retrieve(session.invoice)
            invoice_url = invoice_obj.hosted_invoice_url
            print(f"Invoice URL retrieved: {invoice_url}")  # Debugging line

        # TODO: Update Firestore Payments and Users collection here:
        # Example Firestore update:
        if session.payment_status == 'paid':
            if license_type: 
                payments_ref = db.collection('Payments')
                query = payments_ref.where('uid', '==', uid).where('license_type', '==', license_type).limit(1).stream()
                payment_doc = next(query, None)

                if payment_doc:
                    # Update the existing document with the invoice_url
                    payments_ref.document(payment_doc.id).update({
                        'status': 'paid', 
                        'payment_intent': session.payment_intent, 
                        'paid_at': datetime.utcnow(),
                        'invoice_url': invoice_url
                    })
                else:
                    # Create a new document if it doesn't exist
                    payments_ref.add({
                        'uid': uid,
                        'license_type': license_type,
                        'status': 'paid',
                        'payment_intent': session.payment_intent,
                        'paid_at': datetime.utcnow(),
                        'invoice_url': invoice_url
                    })

                # Update the user's license type in Firestore
                users_ref = db.collection('Users').document(uid)
                users_ref.update({
                                'license_type': license_type
                                })
                
                # Update session with new license type
                session['license_type'] = int(license_type)  # Ensure the session reflects the updated license type
                print(f"Updated session after payment: {session}")
                print(f"Payment for user {uid} with license type {license_type} marked as paid in Firestore.")


                if invoice_url:
                    users_ref.update({'invoice_url': invoice_url})
                    print(f"Invoice URL saved to Firestore: {invoice_url}")  # Debugging line

            return redirect(url_for('upload_receipt'))

        else:
            flash('Payment failed. Please try again.', 'error')
            return redirect(url_for('payment-cancelled'))
    except Exception as e:
        return f'Error verifying payment: {e}', 500


@app.route('/payment-cancelled')
def payment_cancelled():
    return 'Payment cancelled. You can retry purchasing a license anytime.'


@app.route('/stripe-webhook', methods=['POST'])
def stripe_webhook():
    event = None
    payload = request.data
    sig_header = request.headers.get('stripe-signature')

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, stripe_webhook_secret
        )
    except ValueError:
        # Invalid payload
        return '', 400
    except stripe.error.SignatureVerificationError:
        # Invalid signature
        return '', 400

    # Handle checkout session completed event
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        uid = session['metadata'].get('uid')  # Get uid from metadata
        license_type = session['metadata'].get('license_type')

        if not uid:
            print(f"Missing UID or License Type in metadata: {session['metadata']}")
            return '', 400  # If no uid found in metadata, exit with an error
        
        # Check if the user document exists in Firestore
        users_ref = db.collection('Users').document(uid)
        user_doc = users_ref.get()

        if user_doc.exists:
            invoice_url = None
            if session.get('invoice'):
                try:
                    invoice_obj = stripe.Invoice.retrieve(session['invoice'])
                    invoice_url = invoice_obj.hosted_invoice_url
                except Exception as e:
                    print(f"Error retrieving invoice: {e}")

            # Update Firestore to mark payment as paid
            payments_ref = db.collection('Payments')
            payments_ref.add({
            'uid': uid,
            'license_type': license_type,
            'status': 'paid',  # Payment status should be 'paid' after successful session
            'payment_intent': session['payment_intent'],  # You can store the payment intent ID
            'created_at': firestore.SERVER_TIMESTAMP, # Timestamp for when payment was completed
            'invoice_url': invoice_url
        })
            query = payments_ref.where(
                filter=FieldFilter('uid', '==', uid)
            ).where(
                filter=FieldFilter('license_type', '==', license_type)
            ).limit(1).stream()

            for doc in query:
                payments_ref.document(doc.id).update({
                    'status': 'paid',
                    'payment_intent': session.payment_intent,
                    'paid_at': datetime.utcnow(),
                    'invoice_url': invoice_url
                })

                update_data = {
                    'status': 'waiting_review',
                    'license_type': license_type
                }

                if invoice_url:
                    update_data['invoice_url'] = invoice_url
                users_ref.update(update_data)
                
        else:
            # User doesn't exist, handle the error (log or notify)
            print(f"User with UID {uid} not found. Skipping update.")
            return '', 400
        
    return '', 200

@app.route('/upload_receipt', methods=['GET', 'POST'])
def upload_receipt():
    if request.method == 'GET':
        return render_template('upload_receipt.html')
    
    else:
        try:
            uid = request.form['uid']

            if not uid:
                return jsonify({"error": "UID is missing"}), 400
            # Check if the file is in the request
            if 'receipt' not in request.files:
                return jsonify({"error": "No file part"}), 400

            file = request.files['receipt']
            
            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join("uploads", filename)

                # Upload file to Firebase Storage
                blob = bucket.blob(f"receipts/{filename}")
                blob.upload_from_file(file)

                # Get file metadata (URL, size)
                file.seek(0, os.SEEK_END)  # Move to end to get size
                file_size = file.tell()
                file_url = blob.public_url

                # Save file metadata to Firestore
                data = {
                    'uid': uid,
                    'file_name': filename,
                    'file_size': file_size,
                    'file_url': file_url,
                    'upload_time': firestore.SERVER_TIMESTAMP
                }
                db.collection('receipts').add(data)

                # Update the user's status to 'waiting_review'
                user_ref = db.collection('Users').document(uid)
                user_ref.update({'status': 'waiting_review'})

                return redirect(url_for('waiting_page'))
            
            return jsonify({"error": "File type not allowed"}), 400

        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/waiting_page')
def waiting_page():
    uid = session.get('uid')
    # Get the user's status from Firestore
    user_ref = db.collection('Users').document(uid)
    user_doc = user_ref.get()

    if user_doc.exists:
        user_status = user_doc.to_dict().get('status', '')
        if user_status == 'approved':
            return redirect(url_for('index'))  # Redirect to index if approved
        elif user_status != 'waiting_review':
            return redirect(url_for('register'))  # Redirect if not waiting_review

    return render_template('waiting_page')

@app.route('/admin_dashboard')
def admin_dashboard():
    # Fetch users where status is 'waiting_review'
    users_ref = db.collection('Users').where('status', '==', 'waiting_review')
    users = users_ref.stream()
    
    total_users = 0
    pending_users = 0
    approved_users = 0
    rejected_users = 0

    # Get user data
    users_data = []
    for user in users:
        data = user.to_dict()
        user_id = user.id
        data['id'] = user_id  # ✅ This line is essential
        data['status'] = data.get('status', 'unknown')
        
        payments_query = db.collection('Payments').where('uid', '==', user_id).where('status', '==', 'paid').limit(1)        

        invoice_url = None
        for payment_doc in payments_query.stream():
            payment_data = payment_doc.to_dict()
            invoice_url = payment_data.get('invoice_url')
            break

        print(f"Invoice URL for user {user_id}: {invoice_url}")  # Debugging line
        data['invoice_url'] = invoice_url

        data['file_url'] = None
        receipt_query  = db.collection('receipts').where('uid', '==', user.id).limit(1).stream()

        for doc in receipt_query:
            receipt_data = doc.to_dict()
            data['file_url'] = receipt_data.get('file_url')
            break  # we only want the first matching document
        

        users_data.append(data)
        total_users += 1

        status = data.get('status', '').lower()
        if status in ['pending', 'waiting_review']:
            pending_users += 1
        elif status == 'approved':
            approved_users += 1
        elif status == 'rejected':
            rejected_users += 1
    
    return render_template('admin_dashboard.html', 
                            users=users_data, 
                            total_users=total_users,
                            pending_users=pending_users,
                            approved_users=approved_users,
                            rejected_users=rejected_users)

@app.route('/admin/approve_receipt', methods=['POST'])
def approve_receipt():
    uid = request.form.get('uid')
    
    if not uid:
        return jsonify({"error": "Missing UID"}), 400
    
    try:
        # Get user reference
        user_ref = db.collection('Users').document(uid)

        # Check if the user exists
        user_doc = user_ref.get()
        if not user_doc.exists:
            return jsonify({"error": "User not found"}), 404

        # Update the user's status to 'approved'
        user_ref.update({'status': 'approved'})

        return jsonify({"success": f"User {uid} approved successfully."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/admin/reject_receipt', methods=['POST'])
def reject_receipt():
    uid = request.form.get('uid')

    if not uid:
        return jsonify({"error": "Missing UID"}), 400
    
    try:
        # Get user reference
        user_ref = db.collection('Users').document(uid)

        # Check if the user exists
        user_doc = user_ref.get()
        if not user_doc.exists:
            return jsonify({"error": "User not found"}), 404

        # Update the user's status to 'rejected'
        user_ref.update({'status': 'rejected'})

        return jsonify({"success": f"User {uid} rejected successfully."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/get-receipt-url/<uid>', methods=['GET'])
def get_receipt_url(uid):
    try:
       # Get receipt document based on UID
        receipts_ref = db.collection('receipts')
        query = receipts_ref.where('uid', '==', uid).limit(1).stream()
        receipt_doc = next(query, None)
        
        if not receipt_doc:
            print(f"[DEBUG] No receipt document found for UID: {uid}")
            return jsonify({'error': 'No receipt found for this user.'}), 404

        receipt_data = receipt_doc.to_dict()
        file_name = receipt_data.get('file_name')
        print(f"Receipt data for UID {uid}: {receipt_data}")

        if not file_name:
            print(f"No file_name found in receipt document for UID: {uid}")
            return jsonify({'error': 'Receipt file name not found.'}), 404

        blob = bucket.blob(f'receipts/{file_name}')
        print(f"Checking blob: receipts/{file_name}")

        if not blob.exists():
            print(f"File receipts/{file_name} not found in bucket fypautoml.firebasestorage.app")
            return jsonify({'error': 'File not found in storage.'}), 404
        
        # Dynamically patch the content type based on file extension
        if file_name.endswith('.pdf'):
            blob.content_type = 'application/pdf'
        elif file_name.endswith('.png'):
            blob.content_type = 'image/png'
        elif file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
            blob.content_type = 'image/jpeg'
        else:
            blob.content_type = 'application/octet-stream'  # fallback

        blob.patch()  # 🔁 Apply the content type update

        signed_url = blob.generate_signed_url(
            expiration=3600,  # 1 hour in seconds
            method='GET',
            version='v4',
            response_disposition='inline'
        )
        print(f"Generated signed URL for UID {uid}: {signed_url}")

        return jsonify({'signed_url': signed_url})

    except Exception as e:
        print(f"[ERROR] Failed to generate signed URL: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print("[/index] session:", dict(session))
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_dataset_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)


            session['filename'] = filename
            license_type = session.get('license_type')
            if license_type is None:
                license_type = session.get('license_type_id', 1)

            
            try:
                data = pd.read_csv(filepath)
                data.columns = data.columns.str.strip()
                columns = data.columns.tolist()
                preview = data.head(5).to_html(classes='preview-table', index=False)
                numeric_data = data.select_dtypes(include=['number'])
                corr_html = numeric_data.corr().round(2).to_html(classes='correlation-matrix', index=True) if not numeric_data.empty else "<p>No numeric columns for correlation matrix</p>"

                print("License type in session:", session.get('license_type'))

                return render_template('select_target.html',
                                    filename=filename, 
                                    columns=columns, 
                                    preview=preview,
                                    corr_matrix=corr_html,
                                    license_type=license_type)
            except Exception as e:
                print("\nERROR:", str(e))
                flash(f'Error reading file: {str(e)}')
                return redirect(url_for('index'))
    
    return render_template('index.html')

@app.route('/select_target', methods=['GET', 'POST'])
@handle_large_session
def select_target():
    # Handle GET request (initial page load)
    if request.method == 'GET':
        if 'filename' not in session:
            flash('Please upload a dataset first')
            return redirect(url_for('index'))
        
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], session['filename'])
            data = pd.read_csv(filepath)
            data.columns = data.columns.str.strip()
            columns = data.columns.tolist()
            preview = data.head(5).to_html(classes='preview-table', index=False)
            numeric_data = data.select_dtypes(include=['number'])
            corr_html = numeric_data.corr().round(2).to_html(classes='correlation-matrix', index=True) if not numeric_data.empty else "<p>No numeric columns for correlation matrix</p>"
            
            # Retrieve previous selections from session if available
            selected_target = session.get('target_column', '')
            selected_algorithms = session.get('selected_algorithms', [])
            selected_features = session.get('selected_features', [])
            license_type = session.get('license_type')
            
            return render_template('select_target.html',
                                filename=session['filename'],
                                columns=columns,
                                preview=preview,
                                corr_matrix=corr_html,
                                selected_target=selected_target,
                                selected_algorithms=selected_algorithms,
                                selected_features=selected_features,
                                license_type=license_type
                                )
            
        except Exception as e:
            flash(f'Error reading file: {str(e)}')
            return redirect(url_for('index'))

    # Handle POST request (form submission)
    if request.method == 'POST':
        # Validate required fields
        target_column = request.form.get('target_column')
        if not target_column:
            flash('Please select a target column')
            return redirect(url_for('select_target'))
        
        selected_algorithms = request.form.getlist('algorithms')

        license_type = str(session.get('license_type') or session.get('license_type_id') or '1')

        if license_type == '1' and 'tpot' in selected_algorithms:
            selected_algorithms.remove('tpot')

        if not selected_algorithms:
            flash('Please select at least one algorithm')
            return redirect(url_for('select_target'))
        
        # Get the selected features (it will be a JSON string)
        selected_features = request.form.get('selected_features')
        if not selected_features:
            flash('Please select at least one feature')
            return redirect(url_for('select_target'))
        
        selected_features = json.loads(selected_features)

        # Store selections in session
        session['target_column'] = target_column
        session['selected_algorithms'] = selected_algorithms
        session['selected_features'] = selected_features

        # Detect task type
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], session['filename'])
            data = pd.read_csv(filepath)
            data.columns = data.columns.str.strip()
            y = data[target_column]
            session['task_type'] = detect_task_type(y)
        except Exception as e:
            flash(f'Error detecting task type: {str(e)}')
            return redirect(url_for('select_target'))
            
        return redirect(url_for('configure_parameters'))

@app.route('/configure_parameters', methods=['GET', 'POST'])
def configure_parameters():
    if 'filename' not in session or 'selected_algorithms' not in session or 'target_column' not in session:
        flash('Missing required session data. Please start over.')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        try:
            # Validate TPOT parameters if selected
            if 'tpot' in session['selected_algorithms']:
                crossover_rate = request.form.get('crossover_rate', type=float)
                mutation_rate = request.form.get('tpot_mutation_rate', type=float)
                
                if crossover_rate is None or mutation_rate is None:
                    flash('Please provide both crossover rate and mutation rates for TPOT')
                    return redirect(url_for('configure_parameters'))
                
                if not (0 <= crossover_rate <= 1) or not (0 <= mutation_rate <= 1):
                    flash('Crossover probability and mutation rates must be between 0 and 1')
                    return redirect(url_for('configure_parameters'))
            
            # Store all parameters in session
            parameters = {
                'cv_value': request.form.get('cv_value', default=5, type=int),
                'random_forest': {
                    'n_estimators': request.form.get('rf_n_estimators', default=100, type=int),
                    'max_depth': request.form.get('rf_max_depth', type=int),
                    'min_samples_split': request.form.get('rf_min_samples_split', default=2, type=int),
                    'min_samples_leaf': request.form.get('rf_min_samples_leaf', default=1, type=int),
                    'max_features': request.form.get('rf_max_features', default='sqrt')
                },
                'svm': {
                    'C': request.form.get('svm_C', default=1.0, type=float),
                    'kernel': request.form.get('svm_kernel', default='rbf'),
                    'degree': request.form.get('svm_degree', default=3, type=int),
                    'gamma': request.form.get('svm_gamma', default='scale')
                },
                'knn': {
                    'n_neighbors': request.form.get('knn_n_neighbors', default=5, type=int),
                    'weights': request.form.get('knn_weights', default='uniform'),
                    'algorithm': request.form.get('knn_algorithm', default='auto'),
                    'p': request.form.get('knn_p', default=2, type=int)
                },
                'naive_bayes': {
                    'priors': request.form.get('nb_priors'),
                    'var_smoothing': request.form.get('nb_var_smoothing', default=1e-9, type=float)
                },
                'tpot': None
            }

            # Only process TPOT parameters if TPOT is selected
            if 'tpot' in session['selected_algorithms']:
                crossover_rate = request.form.get('crossover_rate', type=float)
                mutation_rate = request.form.get('tpot_mutation_rate', type=float)
                
                if crossover_rate is None or mutation_rate is None:
                    flash('Please provide both crossover rate and mutation rates for TPOT')
                    return redirect(url_for('configure_parameters'))
                
                if not (0 <= crossover_rate <= 1) or not (0 <= mutation_rate <= 1):
                    flash('Crossover probability and mutation rates must be between 0 and 1')
                    return redirect(url_for('configure_parameters'))
                
                parameters['tpot'] = {
                    'generations': request.form.get('tpot_generations', default=5, type=int),
                    'population_size': request.form.get('tpot_population_size', default=20, type=int),
                    'cv': request.form.get('tpot_cv', default=5, type=int),
                    'crossover_rate': crossover_rate,
                    'mutation_rate': mutation_rate,
                    'max_time_mins': request.form.get('tpot_max_time_mins', default=None, type=int)
                }

            session['parameters'] = parameters
            return redirect(url_for('results'))

        except Exception as e:
            print(f"Error in configure_parameters: {str(e)}")
            flash(f'An error occurred: {str(e)}')
            return redirect(url_for('configure_parameters'))
    
    return render_template('configure_parameters.html', 
                         selected_algorithms=session['selected_algorithms'])

@app.route('/results')
@handle_large_session
def results():
    if 'filename' not in session or 'selected_algorithms' not in session or 'parameters' not in session:
        return redirect(url_for('index'))

    filename = session['filename']
    target_column = session['target_column']
    selected_algorithms = session['selected_algorithms']
    selected_features = session.get('selected_features', [])
    parameters = session['parameters']
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data = pd.read_csv(filepath)
    data.columns = data.columns.str.strip()

    if selected_features:
        # Make sure selected features exist in the dataset
        print("=== RESULTS DEBUG ===")
        print("filename:", filename)
        print("target_column:", target_column)
        print("selected_features:", selected_features)
        print("data columns:", data.columns.tolist())
        missing_features = [feature for feature in selected_features if feature not in data.columns]
        if missing_features:
            print("MISSING FEATURES:", missing_features)
            flash(f"The following features are missing from the dataset: {', '.join(missing_features)}")
            return redirect(url_for('select_target'))
        data = data[selected_features + [target_column]]
    else:
        # If no features selected, fall back to using the full dataset
        data = data[[target_column] + list(data.columns.difference([target_column]))]

    X = pd.get_dummies(data.drop(columns=[target_column]))
    y = data[target_column]

    if y.isnull().any():
        flash('Target column contains missing values. Please check your training dataset.')
        return redirect(url_for('index'))

    if X.isnull().any().any():
        flash('Data contains missing values. Please clean your data first.')
        return redirect(url_for('index'))

    if X.shape[1] == 0:
        flash('No features remaining after preprocessing. Check your data.')
        return redirect(url_for('index'))

    task_type = detect_task_type(y)
    session['task_type'] = task_type

    if task_type == 'classification':
        min_class_count = y.value_counts().min()
        safe_cv = max(2, min(parameters['cv_value'], int(min_class_count)))
        parameters['cv_value'] = safe_cv

        if parameters.get('tpot'):
            parameters['tpot']['cv'] = max(2, min(parameters['tpot']['cv'], int(min_class_count)))

    results = {}
    feature_importance_plots = {}
    tpot_info = None
    models = {}
    plt.switch_backend('Agg')

    for algo in selected_algorithms:
        try:
            if algo == 'random_forest':
                p = parameters['random_forest']
                model = (RandomForestClassifier if task_type == 'classification' else RandomForestRegressor)(
                    n_estimators=p['n_estimators'],
                    max_depth=p['max_depth'],
                    min_samples_split=p['min_samples_split'],
                    min_samples_leaf=p['min_samples_leaf'],
                    max_features=p['max_features'],
                    random_state=42
                )

            elif algo == 'svm':
                p = parameters['svm']
                if task_type == 'classification':
                    model = SVC(
                        C=p['C'],
                        kernel=p['kernel'],
                        degree=p['degree'],
                        gamma=p['gamma'],
                        random_state=42,
                        probability=True
                    )
                else:
                    if p['kernel'] in ['poly', 'sigmoid']:
                        model = SVR(
                            C=p['C'],
                            kernel=p['kernel'],
                            degree=p['degree'],
                            gamma=p['gamma'],
                            random_state=42
                        )
                    else:
                        model = SVR(
                            C=p['C'],
                            kernel=p['kernel'],
                            degree=p['degree'],
                            gamma=p['gamma']
                        )

            elif algo == 'knn':
                p = parameters['knn']
                model = (KNeighborsClassifier if task_type == 'classification' else KNeighborsRegressor)(
                    n_neighbors=p['n_neighbors'],
                    weights=p['weights'],
                    algorithm=p['algorithm'],
                    p=p['p']
                )

            elif algo == 'naive_bayes' and task_type == 'classification':
                p = parameters['naive_bayes']
                model = GaussianNB(
                    priors=eval(p['priors']) if p['priors'] else None,
                    var_smoothing=p['var_smoothing']
                )

            elif algo == 'tpot':
                p = parameters['tpot']
                model = (TPOTClassifier if task_type == 'classification' else TPOTRegressor)(
                    generations=p['generations'],
                    population_size=p['population_size'],
                    cv=p['cv'],
                    crossover_rate=p['crossover_rate'],
                    mutation_rate=p['mutation_rate'],
                    max_time_mins=p['max_time_mins'],
                    verbosity=2,
                    random_state=42,
                    n_jobs=-1,
                    early_stop=5,
                    max_eval_time_mins=5
                )
                le = LabelEncoder()
                y_encoded = le.fit_transform(y.astype(str)) if task_type == 'classification' else y

                start_time = time.time()
                print("=== TPOT DEBUG ===")
                print("task_type:", task_type)
                print("X shape:", X.shape)
                print("y classes/counts:")
                print(pd.Series(y_encoded).value_counts())
                print("TPOT params:", p)
                print("X dtypes:")
                print(X.dtypes.value_counts())
                model.fit(X, y_encoded)
                end_time = time.time()

                elapsed_time = round(end_time - start_time, 2)
                print(f"[TPOT] ✅ Population Size {p['population_size']}: {elapsed_time}s completed")

                tpot_pipeline = str(model.fitted_pipeline_)
                tpot_evaluated_individuals = model.evaluated_individuals_
                pipeline_scores = {k: v.get('internal_cv_score', 0) for k, v in tpot_evaluated_individuals.items()}
                best_pipeline, best_score = max(pipeline_scores.items(), key=lambda item: item[1], default=('', 0))

                results['TPOT (GP-AutoML)'] = evaluate_model_cv(model, X, y_encoded, cv=parameters['cv_value'], task_type=task_type)

                models['TPOT (GP-AutoML)'] = {'model': model.fitted_pipeline_, 'type': 'tpot', 'pipeline': tpot_pipeline}

                feature_importance_plots['TPOT (GP-AutoML)'] = create_feature_importance_plot(
                    get_tpot_feature_importance(model, X, y_encoded), X.columns, 'TPOT Feature Importance'
                )
                
                tpot_info = {
                    'pipeline': tpot_pipeline,
                    'pipeline_scores': pipeline_scores,
                    'generations': p['generations'],
                    'population_size': p['population_size'],
                    'total_pipelines': len(pipeline_scores),
                    'best_score': best_score,
                    'best_pipeline': best_pipeline
                }
                continue

            model.fit(X, y)

            display_name = algo.upper() if algo != 'svm' else 'SVM'
            models[display_name] = {'model': model, 'type': algo, 'parameters': model.get_params()}
            results[display_name] = evaluate_model_cv(model, X, y, cv=parameters['cv_value'], task_type=task_type)
            result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
            feature_importance_plots[display_name] = create_feature_importance_plot(
                result.importances_mean, X.columns, f'{display_name} Permutation Importance'
            )

        except Exception as e:
            flash(f'Error training {algo}: {str(e)}')
            app.logger.error(f'Error training {algo}: {str(e)}')

    session_models = {}
    for name, info in models.items():
        try:
            encoded_model = base64.b64encode(pickle.dumps(info['model'])).decode('utf-8')
            session_models[name] = {
                'model': encoded_model,
                'type': info['type'],
                'pipeline': info.get('pipeline', '') if info['type'] == 'tpot' else '',
                'parameters': info.get('parameters', {}) if info['type'] != 'tpot' else {}
            }
        except Exception as e:
            app.logger.error(f"Failed to serialize model {name}: {str(e)}")

    session['models'] = session_models

    # 🔵 Clean prediction preview without probabilities
    prediction_preview = None
    prediction_columns = None
    if 'prediction_data' in session:
        try:
            pred_df = pd.DataFrame(json.loads(session['prediction_data']))
            pred_df = pd.get_dummies(pred_df)
            for col in set(X.columns) - set(pred_df.columns):
                pred_df[col] = 0
            pred_df = pred_df[X.columns]

            sample_pred_df = pred_df.copy()
            for model_name in session_models.keys():
                model = pickle.loads(base64.b64decode(session_models[model_name]['model'].encode('utf-8')))
                sample_pred_df[f'{model_name}_prediction'] = model.predict(pred_df)

            prediction_preview = sample_pred_df.head(5).to_dict(orient='records')
            prediction_columns = sample_pred_df.columns.tolist()
        except Exception as e:
            flash(f'Prediction preview error: {str(e)}')

    return render_template('result.html',
                           results=results,
                           feature_importance_plots=feature_importance_plots,
                           filename=filename,
                           target_column=target_column,
                           cv_value=parameters['cv_value'],
                           tpot_info=tpot_info if 'tpot' in selected_algorithms else None,
                           task_type=task_type,
                           prediction_data=prediction_preview,
                           prediction_columns=prediction_columns,
                           prediction_info={'filename': session.get('prediction_filename')})

# app.py - update the predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if we have a model to use for prediction
        if 'results' not in session:
            flash('Please run the comparison first before making predictions')
            return redirect(url_for('results'))

        if 'prediction_file' not in request.files:
            flash('No prediction file provided')
            return redirect(url_for('results'))
            
        file = request.files['prediction_file']
        
        if file.filename == '':
            flash('No file selected')
            return redirect(url_for('results'))
            
        if not allowed_file(file.filename):
            flash('Invalid file type. Only CSV files are accepted')
            return redirect(url_for('results'))

        # Save the uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load training data to get feature structure
        training_filepath = os.path.join(app.config['UPLOAD_FOLDER'], session['filename'])
        training_data = pd.read_csv(training_filepath)
        training_data.columns = training_data.columns.str.strip()
        target_column = session['target_column']
        
        if target_column not in training_data.columns:
            flash(f'Target column "{target_column}" not found in training data')
            return redirect(url_for('results'))

        X_train = pd.get_dummies(training_data.drop(columns=[target_column]))
        
        # Load and prepare prediction data
        pred_data = pd.read_csv(filepath)
        pred_data.columns = pred_data.columns.str.strip()
        pred_data = pd.get_dummies(pred_data)

        selected_features = session.get('selected_features', [])

        if selected_features:
            pred_data = pred_data[selected_features]  # Keep only the selected features in prediction data

        # Align features with training data
        missing_cols = set(X_train.columns) - set(pred_data.columns)
        extra_cols = set(pred_data.columns) - set(X_train.columns)

        # Add missing columns with 0 values
        for col in missing_cols:
            pred_data[col] = 0

        # Remove extra columns
        pred_data = pred_data[X_train.columns]

        # Store results in session
        session['predictionData'] = pred_data.to_dict(orient='records')
        session['predictionColumns'] = pred_data.columns.tolist()
        
        # Clean up temporary file
        os.remove(filepath)

        return redirect(url_for('results'))

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        flash(f'An error occurred during prediction: {str(e)}')
        return redirect(url_for('results'))

@app.route('/download_predictions')
@handle_large_session
def download_predictions():
    # Check required data exists
    if 'prediction_data' not in session or 'models' not in session:
        flash('No prediction data available or models not trained')
        return redirect(url_for('results'))

    try:
        # Load prediction data
        pred_data_json = session['prediction_data']
        pred_data_list = json.loads(pred_data_json)  # <-- fix here

        pred_data = pd.DataFrame(pred_data_list)
        
        # Load original training data for feature alignment
        training_filepath = os.path.join(app.config['UPLOAD_FOLDER'], session['filename'])
        training_data = pd.read_csv(training_filepath)
        training_data.columns = training_data.columns.str.strip()
        X_train = pd.get_dummies(training_data.drop(columns=[session['target_column']]))
        
        selected_features = session.get('selected_features', [])

        # **Ensure prediction data contains only the selected features** [Highlighted Change]
        if selected_features:
            pred_data = pred_data[selected_features]  # Keep only the selected features in prediction data

        # Preprocess prediction data to match training features
        pred_data = pd.get_dummies(pred_data)
        missing_cols = set(X_train.columns) - set(pred_data.columns)
        for col in missing_cols:
            pred_data[col] = 0
        pred_data = pred_data[X_train.columns]

        # Initialize results with original features
        results = pred_data.copy()
        
        # Generate predictions for each model
        for algo_name, model_info in session['models'].items():
            try:
                # Deserialize model
                model = pickle.loads(base64.b64decode(model_info['model'].encode('utf-8')))
                
                # Make predictions based on task type
                if session['task_type'] == 'classification':
                    # Add class probabilities if available
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(pred_data)
                        for i in range(proba.shape[1]):
                            results[f'{algo_name}_class{i}_probability'] = proba[:, i]
                    # Add class predictions
                    results[f'{algo_name}_prediction'] = model.predict(pred_data)
                else:
                    # Add regression predictions
                    results[f'{algo_name}_prediction'] = model.predict(pred_data)
                    
            except Exception as e:
                flash(f'Error generating predictions for {algo_name}: {str(e)}')
                continue

        # Create CSV output
        output = io.StringIO()
        results.to_csv(output, index=False)
        output.seek(0)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        download_filename = f"predictions_{session['filename'].replace('.csv', '')}_{timestamp}.csv" 
        
        return Response(
            output,
            mimetype="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={download_filename}",
                "Cache-Control": "no-cache"
            }
        )

    except Exception as e:
        flash(f'Error generating predictions: {str(e)}')
        app.logger.error(f"Prediction error: {str(e)}")
        return redirect(url_for('results'))
    
# Add these new routes
@app.route('/upload_prediction', methods=['POST'])
@handle_large_session
def upload_prediction():
    print("UPLOAD_PREDICTION route triggered")  # Debug line

    if 'prediction_file' not in request.files:
        print("No prediction_file part in request.files")  # Debug
        flash('No file uploaded')
        return redirect(url_for('select_target'))
    
    file = request.files['prediction_file']
    print(f"Uploaded file: {file.filename}")  # Debug

    if file.filename == '':
        print("Empty filename")  # Debug
        flash('No selected file')
        return jsonify(success=False, error='No selected file'), 400
    
    if file and allowed_dataset_file(file.filename):
        try:
            # Read and store prediction data
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()
            session['prediction_data'] = df.to_json(orient='records')
            session['prediction_columns'] = df.columns.tolist()
            filename = secure_filename(file.filename)
            session['prediction_filename'] = filename
            print(f"Prediction data received with {len(df)} rows and columns: {df.columns.tolist()}")
            flash('Prediction dataset uploaded successfully.')
            return jsonify(success=True, filename=filename), 200
        except Exception as e:
            print(f"Error reading prediction file: {str(e)}")
            flash(f'Error reading file: {str(e)}')
            return jsonify(success=False, error=f'Error reading file: {str(e)}'), 500

    print("File type not allowed")  # Debug
    flash('Invalid file type.')  
    return jsonify(success=False, error='Invalid file type.'), 400

@app.route('/clear_prediction', methods=['POST'])
def clear_prediction():
    session.pop('prediction_data', None)
    session.pop('prediction_columns', None)
    session.pop('prediction_filename', None)
    return redirect(url_for('results'))

@app.route('/restart')
def restart():
    # Save license_type before clearing session
    license_type = session.get('license_type')
    session.clear()
    if license_type is not None:
        session['license_type'] = license_type
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True)