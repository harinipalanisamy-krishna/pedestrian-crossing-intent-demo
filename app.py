import json
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Pedestrian Crossing Intent", layout="wide")


@st.cache_resource
def load_all():
    import tensorflow as tf
    model = tf.keras.models.load_model("lstm_jaad.keras")
    nz = np.load("norm_jaad.npz")
    data = np.load("demo_data.npz")
    meta = json.load(open("clips.json"))
    return model, nz["mu"], nz["sd"], data["x"], meta


model, mu, sd, X, meta = load_all()

st.title("Pedestrian Crossing Intent Prediction")
st.warning("Stage 1 prototype: trained on JAAD only (preliminary). PIE and Indian road data are the next stages.")

st.markdown("**Pipeline:** Video → YOLO11-pose → 17 keypoints → 30-frame window → LSTM → crossing probability → alert")

tab_demo, tab_res, tab_road = st.tabs(["Live demo", "Results", "Roadmap"])

with tab_demo:
    st.sidebar.header("Alert settings")
    thr = st.sidebar.slider("Alert threshold", 0.1, 0.9, 0.5, 0.05)
    st.sidebar.caption("Lower threshold: fewer misses, more false alarms. Higher: the opposite.")

    names = [m["title"] for m in meta]
    pick = st.selectbox("Choose a sample clip (JAAD test set)", names)
    i = names.index(pick)

    left, right = st.columns([3, 2])
    with left:
        st.video(meta[i]["file"])
        st.caption("Green skeleton = YOLO11-pose keypoints, blue box = tracked pedestrian. The model sees this 1-second window, which ends 1 second before the crossing frame.")
    with right:
        if meta[i]["ok"] < 0.5:
            st.error("Low visibility: skeleton unreliable, prediction not trusted.")
        x = ((X[i] - mu) / sd)[None].astype("float32")
        p = float(model.predict(x, verbose=0)[0, 0])
        st.metric("Crossing probability", f"{p:.2f}")
        st.progress(min(max(p, 0.0), 1.0))
        if p >= thr:
            st.error(f"ALERT: pedestrian likely to cross (p = {p:.2f} ≥ {thr:.2f})")
        else:
            st.success(f"No alert (p = {p:.2f} < {thr:.2f})")
        if st.checkbox("Reveal what actually happened"):
            st.info("The pedestrian DID cross." if meta[i]["label"] == 1 else "The pedestrian did NOT cross.")
    st.caption("These clips are selected examples. Overall test performance is on the Results tab.")

with tab_res:
    st.subheader("JAAD test set (93 windows)")
    c = st.columns(5)
    for col, (k, v) in zip(c, [("Accuracy", "0.677"), ("Precision", "0.776"), ("Recall", "0.667"),
                               ("F1", "0.717"), ("ROC-AUC", "0.721")]):
        col.metric(k, v)
    st.write("Confusion matrix (rows = true, columns = predicted)")
    st.dataframe(pd.DataFrame([[25, 11], [19, 38]], index=["true no cross", "true cross"],
                              columns=["pred no cross", "pred cross"]))
    st.caption("Small dataset (248 windows from JAAD). Preliminary result, not a final accuracy.")

with tab_road:
    st.markdown("""
1. **Add PIE** (1,842 pedestrians, vehicle speed) for pretraining
2. **Collect and label Indian road clips**, fine-tune the model
3. **Low-light handling** (CLAHE/gamma, NoIR camera)
4. **Edge deployment** on Raspberry Pi (TFLite, FPS and latency)
5. **Ablation:** rule-based → LSTM → + motion → + vehicle speed
""")
