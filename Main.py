# ============================================================
# Materia : Programación para Procesamiento de Datos
# Producto : P5 - Script ETL con Transformación de Datos
# Autor    : Victor Hugo Barraza Gonzalez
# Fecha    : 04/03/2026
# Dataset  : ds_salaries.csv  (607 registros, 12 columnas)
# ============================================================

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os

# ── Rutas del proyecto ──────────────────────────────────────
INPUT_FOLDER = 'Input'
OUTPUT_FOLDER = 'Output'
FILE_NAME = 'ds_salaries.csv'
OUTPUT_FILE = 'ds_salaries_refined.csv'

os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("=" * 60)
print("  PROCESO ETL - DS SALARIES")
print("=" * 60)

# ============================================================
# FASE 1: EXTRACCIÓN
# Carga de materia prima tecnológica (raw data) desde CSV
# ============================================================
input_path = os.path.join(INPUT_FOLDER, FILE_NAME)
try:
    df = pd.read_csv(input_path)
    print(f"\n[EXTRACCIÓN] Archivo cargado: {input_path}")
    print(f"  → Registros: {df.shape[0]} | Columnas: {df.shape[1]}")
except FileNotFoundError:
    print(
        f"\n[ERROR] Coloque '{FILE_NAME}' "
        f"dentro de la carpeta '{INPUT_FOLDER}/'"
    )
    exit()

# ============================================================
# FASE 2: TRANSFORMACIÓN
# ============================================================
print("\n[TRANSFORMACIÓN] Iniciando limpieza y enriquecimiento...")

# ── T1. Eliminar columna índice redundante ──────────────────
# La columna 'Unnamed: 0' es un índice duplicado del CSV, no
# aporta información analítica y genera ruido en el dataset.
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)
    print("  T1 → Columna índice redundante 'Unnamed: 0' eliminada.")

# ── T2. Eliminación de duplicados ──────────────────────────
# Registros idénticos generan sesgo en promedios y conteos.
antes = len(df)
df.drop_duplicates(inplace=True)
eliminados = antes - len(df)
print(f"  T2 → Duplicados eliminados: {eliminados} "
      f"(registros restantes: {len(df)})")

# ── T3. Gestión de valores faltantes ───────────────────────
# Se detectan nulos en todas las columnas. Para salary_in_usd
# se aplica imputación con la mediana (más robusta que la
# media ante outliers). Para columnas categóricas, se imputa
# con la moda (valor más frecuente).
nulos_total = df.isnull().sum().sum()
print(f"  T3 → Valores nulos detectados: {nulos_total}")

if nulos_total > 0:
    # Numéricas → mediana
    for col in df.select_dtypes(include='number').columns:
        if df[col].isnull().any():
            mediana = df[col].median()
            df[col].fillna(mediana, inplace=True)
            print(
                f"       '{col}': nulos imputados "
                f"con mediana ({mediana:.2f})"
            )
    # Categóricas → moda
    for col in df.select_dtypes(include='object').columns:
        if df[col].isnull().any():
            moda = df[col].mode()[0]
            df[col].fillna(moda, inplace=True)
            print(f"       '{col}': nulos imputados con moda ('{moda}')")
else:
    print("       Sin valores nulos. Dataset íntegro.")

# ── T4. Estandarización de texto ───────────────────────────
# Las máquinas no distinguen "Data Scientist" de "data scientist".
# Normalizar a minúsculas garantiza coherencia en agrupaciones.
df['job_title'] = df['job_title'].str.lower().str.strip()
print("  T4 → job_title normalizado a minúsculas.")

# ── T5. Decodificación de variables categóricas ────────────
# Los códigos originales (EN, MI, SE, EX) no son autoexplicativos.
# Se reemplazan por etiquetas legibles para análisis e informes.
exp_map = {'EN': 'Entry-level', 'MI': 'Mid-level',
           'SE': 'Senior', 'EX': 'Executive'}
size_map = {'S': 'Small', 'M': 'Medium', 'L': 'Large'}
emp_map = {'FT': 'Full-Time', 'PT': 'Part-Time',
           'CT': 'Contract', 'FL': 'Freelance'}

df['experience_level'] = df['experience_level'].map(exp_map)
df['company_size'] = df['company_size'].map(size_map)
df['employment_type'] = df['employment_type'].map(emp_map)
print("  T5 → Códigos decodificados: experience_level, "
      "company_size, employment_type.")

# ── T6. Manejo de outliers en salary_in_usd ────────────────
# Se usa el método IQR (rango intercuartílico) para identificar
# valores extremos. Los outliers se recortan (clipping) al
# límite del rango normal en lugar de eliminarse, preservando
# la cantidad de registros.
Q1 = df['salary_in_usd'].quantile(0.25)
Q3 = df['salary_in_usd'].quantile(0.75)
IQR = Q3 - Q1
lim_inf = Q1 - 1.5 * IQR
lim_sup = Q3 + 1.5 * IQR
outliers = df[(df['salary_in_usd'] < lim_inf) |
              (df['salary_in_usd'] > lim_sup)].shape[0]
df['salary_in_usd'] = df['salary_in_usd'].clip(lim_inf, lim_sup)
print(f"  T6 → Outliers en salary_in_usd recortados: {outliers} registros "
      f"(rango válido: ${lim_inf:,.0f} – ${lim_sup:,.0f})")

# ── T7. Normalización Min-Max ──────────────────────────────
# Escala salary_in_usd al rango [0, 1] para hacerlo comparable
# con otras variables numéricas en algoritmos de distancia (KNN).
scaler_mm = MinMaxScaler()
df['salary_usd_norm'] = scaler_mm.fit_transform(df[['salary_in_usd']])
print("  T7 → Normalización Min-Max aplicada → 'salary_usd_norm' [0,1].")

# ── T8. Estandarización Z-score ───────────────────────────
# Centra los datos en media=0 y desv. estándar=1.
# Preferible para modelos como SVM, PCA o Regresión Logística.
scaler_std = StandardScaler()
df['salary_usd_zscore'] = scaler_std.fit_transform(df[['salary_in_usd']])
print("  T8 → Estandarización Z-score aplicada → 'salary_usd_zscore'.")

# ── T9. Derivación de nuevas variables (Feature Engineering) ─
# Se crean columnas que aportan conocimiento adicional:
# a) salary_level: categoriza el salario en rangos de negocio
# b) is_remote: simplifica remote_ratio a variable binaria
df['salary_level'] = df['salary_in_usd'].apply(
    lambda x: 'Bajo' if x < 60000 else ('Medio' if x < 140000 else 'Alto')
)
df['is_remote'] = df['remote_ratio'].apply(
    lambda x: 'Remoto' if x == 100
    else ('Híbrido' if x == 50 else 'Presencial')
)
print("  T9 → Nuevas variables derivadas: 'salary_level', 'is_remote'.")

# ── T10. Agregación por nivel de experiencia ───────────────
# Resumen estadístico que responde: ¿qué perfil salarial tiene
# cada nivel de experiencia en el mercado de datos?
resumen = df.groupby('experience_level')['salary_in_usd'].agg(
    Promedio='mean', Mediana='median', Maximo='max', Minimo='min'
).round(2)
resumen_path = os.path.join(OUTPUT_FOLDER, 'resumen_por_experiencia.csv')
resumen.to_csv(resumen_path)
print(f"  T10 → Agregación por experiencia guardada en '{resumen_path}'")
print(f"\n{resumen.to_string()}\n")

# ── Estadísticas finales del dataset transformado ──────────
print(f"  Dataset final: {df.shape[0]} registros | {df.shape[1]} columnas")

# ============================================================
# FASE 3: CARGA
# Persistencia del activo estratégico refinado
# ============================================================
output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)
df.to_csv(output_path, index=False)

print("\n[CARGA] Proceso ETL completado exitosamente.")
print(f"  → Dataset refinado guardado en: {output_path}")
print("=" * 60)
