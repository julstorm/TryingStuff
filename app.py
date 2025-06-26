
# app_v2.py - Enhanced Streamlit app with clustering optimization

import streamlit as st
import pandas as pd
import numpy as np
import gzip
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import SDMolSupplier
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import plotly.express as px
import matplotlib.pyplot as plt
import io

st.set_page_config(layout="wide")
st.title("🧬 Enhanced Docking Cluster Viewer (v2)")

st.sidebar.header("Upload Input Files")
csv_file = st.sidebar.file_uploader("Docking Results (.csv)", type=["csv"])
sdf_file = st.sidebar.file_uploader("Ligand Library (.sdf.gz)", type=["gz"])

if csv_file and sdf_file:
    with st.spinner("Loading CSV..."):
        df_csv = pd.read_csv(csv_file)
        if "title" not in df_csv.columns or "r_i_glide_gscore" not in df_csv.columns:
            st.error("CSV must include 'title' and 'r_i_glide_gscore' columns.")
            st.stop()

    with st.spinner("Loading molecules from SDF..."):
        mols = []
        titles = []
        smiles = []
        gzip_stream = gzip.open(sdf_file)
        suppl = SDMolSupplier(gzip_stream)
        for mol in suppl:
            if mol is not None:
                mols.append(mol)
                try:
                    titles.append(mol.GetProp("title"))
                except:
                    titles.append("")
                smiles.append(Chem.MolToSmiles(mol))

        df_sdf = pd.DataFrame({"title": titles, "SMILES": smiles})
        df = pd.merge(df_sdf, df_csv, on="title", how="inner")

    st.success(f"Loaded {len(df)} molecules.")

    with st.spinner("Generating fingerprints and PCA..."):
        fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=1024) for smi in df["SMILES"]]
        fps_array = np.array([list(fp) for fp in fps])
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(fps_array)
        df["PC1"] = pca_result[:, 0]
        df["PC2"] = pca_result[:, 1]

    st.markdown("### 🧠 Optimize DBSCAN Clustering")

    eps_range = np.arange(0.2, 1.2, 0.2)
    min_samples = 5
    sil_scores = []

    for eps in eps_range:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(df[["PC1", "PC2"]])
        labels = db.labels_
        if len(set(labels)) > 1 and -1 not in set(labels):
            score = silhouette_score(df[["PC1", "PC2"]], labels)
            sil_scores.append((eps, score))

    if sil_scores:
        best_eps = max(sil_scores, key=lambda x: x[1])[0]
        st.write(f"✅ Best DBSCAN eps: `{best_eps}` (based on silhouette score)")
    else:
        best_eps = 0.5
        st.warning("Could not optimize eps from data; using default 0.5.")

    db = DBSCAN(eps=best_eps, min_samples=min_samples).fit(df[["PC1", "PC2"]])
    df["Cluster"] = db.labels_.astype(str)

    st.markdown("### 📊 PCA + Optimized DBSCAN Clustering")
    fig = px.scatter(df, x="PC1", y="PC2", color="Cluster", hover_data=["title", "r_i_glide_gscore"])
    st.plotly_chart(fig, use_container_width=True)

    cluster_stats = df.groupby("Cluster")["r_i_glide_gscore"].agg(["count", "mean", "std"]).reset_index()
    st.markdown("### 📈 Docking Score Stats per Cluster")
    st.dataframe(cluster_stats)

    cluster_options = df["Cluster"].unique()
    selected_cluster = st.selectbox("Select Cluster", cluster_options)
    df_selected = df[df["Cluster"] == selected_cluster]

    st.markdown(f"### 📁 Cluster {selected_cluster} Details ({len(df_selected)} ligands)")
    st.dataframe(df_selected[["title", "r_i_glide_gscore", "SMILES"]])

    st.markdown("### 🔬 Substructure Viewer")
    num_mols = st.slider("How many molecules to display?", 1, min(20, len(df_selected)), 6)
    sub_mols = [Chem.MolFromSmiles(smi) for smi in df_selected["SMILES"].head(num_mols)]
    st.image(Draw.MolsToGridImage(sub_mols, molsPerRow=3, subImgSize=(200, 200)))

    st.markdown("### 💾 Download Cluster as SDF")
    def write_sdf(df_rows):
        buf = io.StringIO()
        writer = Chem.SDWriter(buf)
        for smi in df_rows["SMILES"]:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                writer.write(mol)
        writer.close()
        return buf.getvalue()

    sdf_string = write_sdf(df_selected)
    st.download_button("Download .sdf of this cluster", data=sdf_string, file_name=f"cluster_{selected_cluster}.sdf")

else:
    st.info("Upload a `.csv` and `.sdf.gz` to begin.")
