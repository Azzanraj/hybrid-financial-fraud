import os
import uuid
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

from flask import Flask, render_template, request, send_from_directory, flash
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

# =========================
# App config
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"csv"}

app = Flask(__name__)
app.secret_key = "fraud_detection_secret_key_change_this"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Model definitions
# =========================
class FinalAE(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class GATModel(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.gat1 = GATConv(in_dim, 64, heads=4, dropout=0.3)
        self.gat2 = GATConv(64 * 4, 32, heads=2, dropout=0.3)
        self.fc = nn.Linear(32 * 2, 1)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        x = self.fc(x)
        return x.view(-1)


# =========================
# Load artifacts once
# =========================
LGB_MODEL = joblib.load(os.path.join(MODEL_DIR, "have_up_lgb_model.pkl"))
LGB_FEATURES = joblib.load(os.path.join(MODEL_DIR, "have_up_lgb_features.pkl"))

AE_SCALER = joblib.load(os.path.join(MODEL_DIR, "have_up_ae_scaler.pkl"))
AE_FEATURES = joblib.load(os.path.join(MODEL_DIR, "have_up_ae_features.pkl"))
AE_MODEL = FinalAE(len(AE_FEATURES)).to(device)
AE_MODEL.load_state_dict(torch.load(os.path.join(MODEL_DIR, "have_up_ae_model.pth"), map_location=device))
AE_MODEL.eval()

GAT_SCALER = joblib.load(os.path.join(MODEL_DIR, "have_up_gat_scaler.pkl"))
GAT_FEATURES = joblib.load(os.path.join(MODEL_DIR, "have_up_gat_features.pkl"))
GAT_MODEL = GATModel(len(GAT_FEATURES)).to(device)
GAT_MODEL.load_state_dict(torch.load(os.path.join(MODEL_DIR, "have_up_gat_model.pth"), map_location=device))
GAT_MODEL.eval()


# =========================
# Helpers
# =========================
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mn = np.min(x)
    mx = np.max(x)
    if abs(mx - mn) < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mn) / (mx - mn)


def preprocess_uploaded_csv(df: pd.DataFrame):
    df = df.copy()

    transaction_ids = None
    if "TransactionID" in df.columns:
        transaction_ids = df["TransactionID"].copy()
    else:
        transaction_ids = pd.Series(np.arange(len(df)), name="TransactionID")

    y_true = None
    if "isFraud" in df.columns:
        y_true = df["isFraud"].copy()
        df = df.drop(columns=["isFraud"])

    if "TransactionID" in df.columns:
        df = df.drop(columns=["TransactionID"])

    # Numeric / categorical cleanup
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    df[num_cols] = df[num_cols].fillna(-999)
    for c in cat_cols:
        df[c] = df[c].fillna("missing").astype("category").cat.codes.astype(np.int32)

    # Temporal features
    if "TransactionDT" in df.columns:
        df["hour"] = (df["TransactionDT"] // 3600) % 24
        df["day"] = (df["TransactionDT"] // (3600 * 24)) % 7
    else:
        df["hour"] = 0
        df["day"] = 0

    if "card1" in df.columns and "TransactionDT" in df.columns:
        df["time_diff"] = df.groupby("card1")["TransactionDT"].diff().fillna(0)
    else:
        df["time_diff"] = 0

    # Behavioral features
    if "card1" in df.columns and "TransactionAmt" in df.columns:
        card_mean = df.groupby("card1")["TransactionAmt"].mean()
        card_std = df.groupby("card1")["TransactionAmt"].std()
        card_count = df.groupby("card1")["TransactionAmt"].count()

        df["amt_mean"] = df["card1"].map(card_mean)
        df["amt_std"] = df["card1"].map(card_std).fillna(0)
        df["tx_count"] = df["card1"].map(card_count)
        df["amt_norm"] = df["TransactionAmt"] / (df["amt_mean"] + 1)
    else:
        df["amt_mean"] = 0
        df["amt_std"] = 0
        df["tx_count"] = 0
        df["amt_norm"] = 0

    # Advanced features
    if "card1" in df.columns and "DeviceInfo" in df.columns:
        device_count = df.groupby("card1")["DeviceInfo"].nunique()
        df["device_count"] = df["card1"].map(device_count).fillna(0)
    else:
        df["device_count"] = 0

    if "card1" in df.columns and "P_emaildomain" in df.columns:
        email_count = df.groupby("card1")["P_emaildomain"].nunique()
        df["email_count"] = df["card1"].map(email_count).fillna(0)
    else:
        df["email_count"] = 0

    # Relational feature
    if "card1" in df.columns and "addr1" in df.columns:
        tmp = df["card1"].astype(str) + "_" + df["addr1"].astype(str)
        df["card_addr"] = tmp.astype("category").cat.codes.astype(np.int32)
    else:
        df["card_addr"] = 0

    # Final cleanup
    df = df.replace([np.inf, -np.inf], -999)
    df = df.fillna(-999)
    df = df.copy()

    return df, transaction_ids, y_true


def align_features(df: pd.DataFrame, feature_list):
    out = df.copy()
    for c in feature_list:
        if c not in out.columns:
            out[c] = -999
    out = out[feature_list].copy()
    return out


def ae_reconstruction_scores(X_np: np.ndarray) -> np.ndarray:
    X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)
    X_tensor = torch.tensor(X_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        recon = AE_MODEL(X_tensor)
        score = ((X_tensor - recon) ** 2).mean(dim=1).detach().cpu().numpy()
    return score


def build_edge_index(df_for_graph: pd.DataFrame, gat_x_np: np.ndarray):
    edge_set = set()
    rel_cols = [c for c in ["card1", "card_addr", "addr1", "addr2", "DeviceInfo", "P_emaildomain"] if c in df_for_graph.columns]

    for col in rel_cols:
        groups = df_for_graph.groupby(col).indices
        for _, idxs in groups.items():
            idxs = list(idxs)
            if len(idxs) < 2:
                continue
            limit = min(len(idxs) - 1, 3)
            for i in range(limit):
                u, v = int(idxs[i]), int(idxs[i + 1])
                if u != v:
                    edge_set.add((u, v))
                    edge_set.add((v, u))

    if len(edge_set) == 0:
        n = len(gat_x_np)
        if n <= 1:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        k = min(8, n)
        nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(gat_x_np)
        _, nn_idx = nbrs.kneighbors(gat_x_np)
        for i in range(len(nn_idx)):
            for j in nn_idx[i][1:]:
                edge_set.add((i, int(j)))
                edge_set.add((int(j), i))

    edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
    return edge_index.to(device)


def gat_scores(df_raw: pd.DataFrame, df_processed: pd.DataFrame) -> np.ndarray:
    X_gat = align_features(df_processed, GAT_FEATURES)
    X_gat = GAT_SCALER.transform(X_gat).astype(np.float32)
    X_gat = np.nan_to_num(X_gat, nan=0.0, posinf=0.0, neginf=0.0)

    edge_index = build_edge_index(df_processed, X_gat)

    x_tensor = torch.tensor(X_gat, dtype=torch.float32, device=device)
    data = Data(x=x_tensor, edge_index=edge_index)

    with torch.no_grad():
        logits = GAT_MODEL(data.x, data.edge_index)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

    return probs


def run_pipeline(df_raw: pd.DataFrame, threshold: float = 0.5):
    processed, transaction_ids, y_true = preprocess_uploaded_csv(df_raw)

    # LightGBM
    X_lgb = align_features(processed, LGB_FEATURES)
    lgb_scores = LGB_MODEL.predict_proba(X_lgb)[:, 1]

    # Autoencoder
    X_ae = align_features(processed, AE_FEATURES)
    X_ae = AE_SCALER.transform(X_ae).astype(np.float32)
    ae_scores = ae_reconstruction_scores(X_ae)

    # GAT
    gat_scores_arr = gat_scores(df_raw, processed)

    # Demo ensemble: normalize each score source to [0,1] and average
    lgb_n = minmax(lgb_scores)
    ae_n = minmax(ae_scores)
    gat_n = minmax(gat_scores_arr)

    final_score = (lgb_n + ae_n + gat_n) / 3.0
    final_pred = (final_score >= threshold).astype(int)

    result_df = pd.DataFrame({
        "TransactionID": transaction_ids.values if hasattr(transaction_ids, "values") else transaction_ids,
        "LGB_Score": lgb_scores,
        "AE_Score": ae_scores,
        "GAT_Score": gat_scores_arr,
        "Final_Risk_Score": final_score,
        "Predicted_Label": final_pred
    })

    result_df["Risk_Level"] = np.where(
        result_df["Final_Risk_Score"] >= 0.75, "High",
        np.where(result_df["Final_Risk_Score"] >= 0.45, "Medium", "Low")
    )



    fraud_count = int(result_df["Predicted_Label"].sum())
    total_count = len(result_df)

    summary = {
        "total_count": total_count,
        "fraud_count": fraud_count,
        "normal_count": total_count - fraud_count,
        "fraud_rate": round((fraud_count / total_count) * 100, 2) if total_count else 0.0,
        "threshold": threshold
    }

    return result_df, summary


# =========================
# Routes
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part found.", "danger")
            return render_template("index.html")

        file = request.files["file"]
        threshold = float(request.form.get("threshold", 0.5))

        if file.filename == "":
            flash("Please choose a CSV file.", "danger")
            return render_template("index.html")

        if not allowed_file(file.filename):
            flash("Only CSV files are allowed.", "danger")
            return render_template("index.html")

        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        upload_path = os.path.join(UPLOAD_DIR, unique_name)
        file.save(upload_path)

        df_raw = pd.read_csv(upload_path)
        result_df, summary,  = run_pipeline(df_raw, threshold=threshold)

        out_name = f"fraud_results_{uuid.uuid4().hex}.csv"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        result_df.to_csv(out_path, index=False)

        table_rows = result_df.head(300).to_dict(orient="records")

        return render_template(
            "results.html",
            rows=table_rows,
            summary=summary,
            download_file=out_name
        )

    return render_template("index.html")


@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)