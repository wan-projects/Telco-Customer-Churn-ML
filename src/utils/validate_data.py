import great_expectations as gx
from typing import Tuple, List


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Validasi data yang komprehensif untuk dataset Telco Customer Churn menggunakan Great Expectations.

    Fungsi ini mengimplementasikan pengecekan kualitas data yang krusial dan harus lolos
    sebelum training model. Fungsi ini memvalidasi integritas data, batasan logika bisnis,
    dan properti statistik yang diharapkan oleh model ML.

    Ditulis untuk Great Expectations versi 1.x (Fluent API / GX Core).
    """
    print("🔍 Memulai validasi data dengan Great Expectations...")

    # === SETUP CONTEXT, DATA SOURCE, DATA ASSET, BATCH (khusus GE 1.x) ===
    # Context menyimpan konfigurasi GE selama proses berjalan (di memori, tidak disimpan ke disk)
    context = gx.get_context(mode="ephemeral")

    # Daftarkan pandas sebagai sumber data
    data_source = context.data_sources.add_pandas("telco_pandas_datasource")

    # Daftarkan DataFrame yang mau divalidasi sebagai "data asset"
    data_asset = data_source.add_dataframe_asset(name="telco_data_asset")

    # Definisikan "batch" - representasi satu kumpulan data yang akan divalidasi
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "telco_batch_definition"
    )

    # Ambil batch aktual dengan menyisipkan DataFrame kita ke dalamnya
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    # === DEFINISIKAN SEMUA EXPECTATION ===
    print("   📋 Menyusun daftar expectation (skema, logika bisnis, rentang, konsistensi)...")

    expectations = [
        # --- VALIDASI SKEMA - KOLOM WAJIB ---
        # Identifier customer harus ada (wajib untuk operasional bisnis)
        gx.expectations.ExpectColumnToExist(column="customerID"),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="customerID"),

        # Fitur demografi utama
        gx.expectations.ExpectColumnToExist(column="gender"),
        gx.expectations.ExpectColumnToExist(column="Partner"),
        gx.expectations.ExpectColumnToExist(column="Dependents"),

        # Fitur layanan (krusial untuk analisis churn)
        gx.expectations.ExpectColumnToExist(column="PhoneService"),
        gx.expectations.ExpectColumnToExist(column="InternetService"),
        gx.expectations.ExpectColumnToExist(column="Contract"),

        # Fitur finansial (prediktor utama churn)
        gx.expectations.ExpectColumnToExist(column="tenure"),
        gx.expectations.ExpectColumnToExist(column="MonthlyCharges"),
        gx.expectations.ExpectColumnToExist(column="TotalCharges"),

        # --- VALIDASI LOGIKA BISNIS ---
        # Gender harus salah satu dari nilai yang diharapkan (integritas data)
        gx.expectations.ExpectColumnValuesToBeInSet(column="gender", value_set=["Male", "Female"]),

        # Field Yes/No harus memiliki nilai yang valid
        gx.expectations.ExpectColumnValuesToBeInSet(column="Partner", value_set=["Yes", "No"]),
        gx.expectations.ExpectColumnValuesToBeInSet(column="Dependents", value_set=["Yes", "No"]),
        gx.expectations.ExpectColumnValuesToBeInSet(column="PhoneService", value_set=["Yes", "No"]),

        # Tipe kontrak harus valid (batasan bisnis)
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="Contract",
            value_set=["Month-to-month", "One year", "Two year"],
        ),

        # Tipe layanan internet (batasan bisnis)
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="InternetService",
            value_set=["DSL", "Fiber optic", "No"],
        ),

        # --- VALIDASI RENTANG NUMERIK & STATISTIK ---
        # Tenure harus masuk akal: non-negatif, maksimum ~10 tahun (120 bulan)
        gx.expectations.ExpectColumnValuesToBeBetween(column="tenure", min_value=0, max_value=120),

        # Monthly charges harus dalam rentang bisnis yang masuk akal
        gx.expectations.ExpectColumnValuesToBeBetween(column="MonthlyCharges", min_value=0, max_value=200),

        # Total charges harus tidak negatif
        gx.expectations.ExpectColumnValuesToBeBetween(column="TotalCharges", min_value=0),

        # Tidak boleh ada nilai kosong di fitur numerik yang krusial
        gx.expectations.ExpectColumnValuesToNotBeNull(column="tenure"),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="MonthlyCharges"),

        # --- PENGECEKAN KONSISTENSI DATA ---
        # Total charges umumnya harus >= Monthly charges (kecuali pelanggan yang sangat baru)
        # Toleransi 5% pengecualian untuk kasus edge case
        gx.expectations.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="TotalCharges",
            column_B="MonthlyCharges",
            or_equal=True,
            mostly=0.95,
        ),
    ]

    # === JALANKAN SETIAP EXPECTATION SATU PER SATU ===
    print("   ⚙️  Menjalankan suite validasi lengkap...")

    failed_expectations = []
    passed_checks = 0
    total_checks = len(expectations)

    for expectation in expectations:
        result = batch.validate(expectation)
        if result.success:
            passed_checks += 1
        else:
            # Ambil nama class expectation sebagai identitas check yang gagal
            failed_expectations.append(type(expectation).__name__)

    failed_checks = total_checks - passed_checks
    overall_success = len(failed_expectations) == 0

    # === CETAK RINGKASAN HASIL VALIDASI ===
    if overall_success:
        print(f"✅ Validasi data BERHASIL: {passed_checks}/{total_checks} pengecekan sukses")
    else:
        print(f"❌ Validasi data GAGAL: {failed_checks}/{total_checks} pengecekan gagal")
        print(f"   Expectation yang gagal: {failed_expectations}")

    return overall_success, failed_expectations