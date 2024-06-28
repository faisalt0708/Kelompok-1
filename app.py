import streamlit as st
import pandas as pd
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import uuid

def show_data():
    st.header('Menu Data')
    st.write('Ini adalah halaman untuk menampilkan data dari Excel.')

    # Inisialisasi state untuk menyimpan data yang diunggah dan jumlah upload
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
        st.session_state.upload_count = 0

    # Hanya izinkan maksimal 3 upload
    if st.session_state.upload_count < 3:
        uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx", "xls"], key=f"file_uploader_{st.session_state.upload_count}")

        if uploaded_file is not None:
            # Simpan file yang diunggah ke dalam session state
            st.session_state.uploaded_files.append(uploaded_file)
            st.session_state.upload_count += 1

            # Baca data Excel
            df = pd.read_excel(uploaded_file, sheet_name='Daftar Peserta Didik')

            # Simpan DataFrame ke dalam session state
            st.session_state[f'data_{st.session_state.upload_count}'] = df

    else:
        st.write("Anda telah mengunggah maksimal 3 file.")

    # Tampilkan data yang diunggah
    for i in range(1, st.session_state.upload_count + 1):
        df = st.session_state.get(f'data_{i}')
        if df is not None:
            st.write(f"Data dari Upload {i}:")
            st.dataframe(df)
            
            # Opsional: Tampilkan informasi statistik ringkas
            st.write("Informasi statistik ringkas:")
            st.write(df.describe())

def show_kategorial():
    st.header('Menu Kategorial')
    st.write('Ini adalah halaman untuk menampilkan data kategorial dari Excel.')

    # Pastikan ada file yang diunggah
    if st.session_state.uploaded_files:
        # Ambil file pertama dari daftar yang diunggah
        uploaded_file = st.session_state.uploaded_files[0]

        # Baca data Kategorial dari file yang diunggah
        df_kategorial = pd.read_excel(uploaded_file, sheet_name='kategorial')

        # Tampilkan data Kategorial
        st.write("Data Kategorial dari Excel:")
        st.dataframe(df_kategorial)

        # Opsional: Tampilkan informasi statistik ringkas
        # st.write("Informasi statistik ringkas:")
        # st.write(df_kategorial.describe())
    else:
        st.write("Silakan unggah file Excel terlebih dahulu di menu Data.")

def show_perhitungan():
    st.header('Menu Perhitungan')
    st.write('Ini adalah halaman untuk menghitung data menggunakan metode K-Medoids.')

    # Pastikan ada file yang diunggah untuk perhitungan
    if st.session_state.uploaded_files:
        # Ambil file pertama dari daftar yang diunggah
        uploaded_file = st.session_state.uploaded_files[0]

        # Baca data Perhitungan dari file yang diunggah
        df_perhitungan = pd.read_excel(uploaded_file, sheet_name='number')

        # Tampilkan data Perhitungan
        st.write("Data Perhitungan dari Excel:")
        st.dataframe(df_perhitungan)

        # Hapus kolom yang tidak relevan atau tidak dapat diproses
        columns_to_drop = []  # Tambahkan nama kolom yang ingin dihapus
        df_perhitungan = df_perhitungan.drop(columns=columns_to_drop)

        # Lakukan penanganan nilai NaN
        if df_perhitungan.isnull().values.any():
            st.warning("Data mengandung nilai NaN. Melakukan penggantian NaN dengan nilai rata-rata kolom.")
            
            # Ganti NaN dengan nilai rata-rata kolom
            df_perhitungan = df_perhitungan.fillna(df_perhitungan.mean())

        # Cek tipe data kolom dan konversi jika perlu
        for column in df_perhitungan.columns:
            if df_perhitungan[column].dtype == 'object':
                df_perhitungan[column] = pd.to_numeric(df_perhitungan[column], errors='coerce')

        # Input untuk menentukan jumlah cluster
        n_clusters = st.slider('Pilih jumlah cluster', min_value=2, max_value=10, value=3)

        if st.button('Lakukan Perhitungan K-Medoids'):
            try:
                # Lakukan clustering menggunakan K-Medoids
                kmedoids = KMedoids(n_clusters=n_clusters, random_state=0)
                labels = kmedoids.fit_predict(df_perhitungan)

                # Tambahkan label cluster ke DataFrame
                df_perhitungan['Cluster'] = labels

                # Tampilkan hasil clustering
                st.write("Hasil Clustering:")
                st.dataframe(df_perhitungan)

                # Plot hasil clustering
                plot_clusters(df_perhitungan, kmedoids)

            except ValueError as e:
                st.error(f"Terjadi kesalahan saat melakukan clustering: {str(e)}")

    else:
        st.write("Silakan unggah file Excel terlebih dahulu di menu Data.")

def plot_clusters(df, kmedoids):
    # Plot hasil clustering menggunakan Scatter Plot
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 6))

    unique_labels = np.unique(kmedoids.labels_)
    colors = sns.color_palette('husl', n_colors=len(unique_labels))

    for label, color in zip(unique_labels, colors):
        subset = df[df['Cluster'] == label]
        plt.scatter(subset.iloc[:, 0], subset.iloc[:, 1], label=f'Cluster {label}', color=color, alpha=0.7)

    plt.title('Clustering menggunakan K-Medoids')
    plt.xlabel('Fitur 1')
    plt.ylabel('Fitur 2')
    plt.legend()
    
    st.pyplot(plt)

def main():
    st.title('Clustering Data Siswa Menggunakan Algoritma K-Medoid')
    menu = ['Data', 'Kategorial', 'Perhitungan']
    choice = st.sidebar.selectbox('Pilih Menu', menu)

    if choice == 'Data':
        show_data()
    elif choice == 'Kategorial':
        show_kategorial()
    elif choice == 'Perhitungan':
        show_perhitungan()

if __name__ == "__main__":
    main()