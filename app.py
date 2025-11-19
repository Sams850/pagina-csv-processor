from flask import Flask, request, render_template, redirect, url_for, send_file, flash
import pandas as pd
import numpy as np
import io
import os
from werkzeug.utils import secure_filename
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

app = Flask(__name__)
app.secret_key = os.urandom(24)
ALLOWED_EXTENSIONS = {"csv"}
SESSION_STORE = {}
PROCESSED_FILES = {}
ORIGINAL_DATA = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash("No file part")
        return redirect(url_for('index'))

    f = request.files['file']
    if f.filename == "":
        flash("No file selected")
        return redirect(url_for('index'))
    if not allowed_file(f.filename):
        flash("Solo se permiten CSV")
        return redirect(url_for('index'))

    filename = secure_filename(f.filename)

    try:
        df = pd.read_csv(f)
    except Exception as e:
        flash(f"Error leyendo CSV: {e}")
        return redirect(url_for('index'))

    session_id = os.urandom(12).hex()
    SESSION_STORE[session_id] = df
    ORIGINAL_DATA[session_id] = df.copy()

    preview = df.head(50).copy()
    rows = preview.to_dict(orient="records")
    columns = list(preview.columns)
    numeric_columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    
    missing_info = {}
    missing_numeric_cols = []
    missing_categorical_cols = []
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_info[col] = missing_count
            if pd.api.types.is_numeric_dtype(df[col]):
                missing_numeric_cols.append(col)
            else:
                missing_categorical_cols.append(col)
    
    recommendations = []
    if missing_numeric_cols:
        recommendations.append("Rellenar valores faltantes con método 'mean' o 'median' para columnas numéricas")
        if len(missing_numeric_cols) > 0:
            recommendations.append("Predicción con Árboles de Decisión para columnas numéricas con valores faltantes")
    if missing_categorical_cols:
        recommendations.append("Rellenar valores faltantes con método 'mode' para columnas categóricas")
        if len(missing_categorical_cols) > 0:
            recommendations.append("Predicción con Árboles de Decisión para columnas categóricas con valores faltantes")
    if missing_info:
        recommendations.append("La predicción con Árboles de Decisión es recomendada para obtener valores más precisos")

    return render_template(
        "preview.html",
        filename=filename,
        columns=columns,
        rows=rows,
        session_id=session_id,
        numeric_columns=numeric_columns,
        missing_info=missing_info,
        missing_numeric_cols=missing_numeric_cols,
        missing_categorical_cols=missing_categorical_cols,
        recommendations=recommendations
    )

@app.route('/apply', methods=["POST"])
def apply():
    session_id = request.form.get("session_id")
    if not session_id or session_id not in SESSION_STORE:
        flash("Sesión inválida, sube el CSV nuevamente.")
        return redirect(url_for('index'))

    df = SESSION_STORE[session_id].copy()

    if request.form.get("fill_missing_on"):
        method = request.form.get("fill_method")
        const_val = request.form.get("constant_value")

        if method == "mean":
            for c in df.columns:
                if pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = df[c].fillna(df[c].mean())

        elif method == "median":
            for c in df.columns:
                if pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = df[c].fillna(df[c].median())

        elif method == "mode":
            for c in df.columns:
                if df[c].isnull().any():
                    m = df[c].mode()
                    if len(m) > 0:
                        df[c] = df[c].fillna(m.iloc[0])

        elif method == "ffill":
            df = df.fillna(method="ffill")

        elif method == "bfill":
            df = df.fillna(method="bfill")

        elif method == "constant":
            df = df.fillna(const_val if const_val else "")

    if request.form.get("discretize_on"):
        cols = request.form.getlist("discretize_cols")
        n_bins = int(request.form.get("n_bins") or 5)
        
        for c in cols:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                df[c] = pd.cut(df[c], bins=n_bins, labels=False, duplicates="drop")

    if request.form.get("normalize_on"):
        cols = request.form.getlist("normalize_cols")
        cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

        if len(cols) > 0:
            scaler = MinMaxScaler()
            df[cols] = scaler.fit_transform(df[cols].astype(float))

    if request.form.get("predict_on"):
        predict_cols = request.form.getlist("predict_cols")
        use_forest = request.form.get("use_forest") == "on"
        
        for target_col in predict_cols:
            if target_col not in df.columns:
                continue
                
            missing_mask = df[target_col].isnull()
            if not missing_mask.any():
                continue
                
            df_model = df.copy()
            X = df_model.drop(columns=[target_col])
            y = df_model[target_col]
            
            if X.empty:
                continue
                
            encoders = {}
            for col in X.columns:
                if X[col].dtype == object or X[col].dtype.name == 'category':
                    X[col] = X[col].astype(str).fillna("MISSING")
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                    encoders[col] = le
                else:
                    X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
            
            train_mask = ~y.isnull()
            
            if train_mask.sum() < 2:
                continue
                
            X_train = X[train_mask]
            y_train = y[train_mask]
            X_predict = X[missing_mask]
            
            is_numeric_target = pd.api.types.is_numeric_dtype(y_train)
            target_encoder = None
            
            if not is_numeric_target:
                y_train = y_train.astype(str).fillna("MISSING")
                target_encoder = LabelEncoder()
                y_train = target_encoder.fit_transform(y_train)
            
            try:
                if use_forest:
                    if is_numeric_target:
                        model = RandomForestRegressor(n_estimators=50, random_state=0)
                    else:
                        model = RandomForestClassifier(n_estimators=50, random_state=0)
                else:
                    if is_numeric_target:
                        model = DecisionTreeRegressor(random_state=0)
                    else:
                        model = DecisionTreeClassifier(random_state=0)
                
                model.fit(X_train, y_train)
                preds = model.predict(X_predict)
                
                if target_encoder:
                    preds = target_encoder.inverse_transform(preds.astype(int))
                
                df.loc[missing_mask, target_col] = preds
            except Exception as e:
                continue

    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val.iloc[0])
                else:
                    df[col] = df[col].fillna("")

    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    filename = f"processed_{session_id}.csv"
    
    PROCESSED_FILES[session_id] = buf.getvalue()
    
    from flask import Response
    response = Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "X-Session-ID": session_id
        }
    )
    
    return response

@app.route('/success/<session_id>')
def success(session_id):
    if session_id not in PROCESSED_FILES:
        flash("Sesión no encontrada")
        return redirect(url_for('index'))
    
    filename = f"processed_{session_id}.csv"
    has_original = session_id in ORIGINAL_DATA
    
    return render_template("success.html", session_id=session_id, filename=filename, has_original=has_original)

@app.route('/back/<session_id>')
def back(session_id):
    if session_id not in ORIGINAL_DATA:
        flash("Datos originales no encontrados")
        return redirect(url_for('index'))
    
    df = ORIGINAL_DATA[session_id].copy()
    SESSION_STORE[session_id] = df
    
    preview = df.head(50).copy()
    rows = preview.to_dict(orient="records")
    columns = list(preview.columns)
    numeric_columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    
    missing_info = {}
    missing_numeric_cols = []
    missing_categorical_cols = []
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_info[col] = missing_count
            if pd.api.types.is_numeric_dtype(df[col]):
                missing_numeric_cols.append(col)
            else:
                missing_categorical_cols.append(col)
    
    recommendations = []
    if missing_numeric_cols:
        recommendations.append("Rellenar valores faltantes con método 'mean' o 'median' para columnas numéricas")
        if len(missing_numeric_cols) > 0:
            recommendations.append("Predicción con Árboles de Decisión para columnas numéricas con valores faltantes")
    if missing_categorical_cols:
        recommendations.append("Rellenar valores faltantes con método 'mode' para columnas categóricas")
        if len(missing_categorical_cols) > 0:
            recommendations.append("Predicción con Árboles de Decisión para columnas categóricas con valores faltantes")
    if missing_info:
        recommendations.append("La predicción con Árboles de Decisión es recomendada para obtener valores más precisos")

    return render_template(
        "preview.html",
        filename=f"original_{session_id}.csv",
        columns=columns,
        rows=rows,
        session_id=session_id,
        numeric_columns=numeric_columns,
        missing_info=missing_info,
        missing_numeric_cols=missing_numeric_cols,
        missing_categorical_cols=missing_categorical_cols,
        recommendations=recommendations
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
