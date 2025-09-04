import pandas as pd
import requests
import pingouin as pg
from scipy.stats import kruskal, spearmanr
import scikit_posthocs as sp
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from statsmodels.tools.sm_exceptions import HessianInversionWarning, ConvergenceWarning

# =======================================
# Constants & Configuration
# =======================================
DATA_CSV_PATH = 'data.csv'
OUTPUT_EXCEL_PATH = 'anime_violence_analysis_results.xlsx'
TABLES_OUTPUT_PATH = 'Tables.xlsx'
FIGURES_OUTPUT_DIR = 'figures'
ANILIST_API_URL = 'https://graphql.anilist.co'

# --- Analysis Parameters ---
SIGNIFICANCE_THRESHOLD = 0.05
MIN_CATEGORY_COUNT = 10  # Min episodes to be considered a standalone category
VIF_THRESHOLD = 10.0     # Standard threshold for detecting high multicollinearity
MIN_NON_ZERO_FOR_REGRESSION = 5 

# --- Column Definitions ---
VIOLENCE_TYPES = [
    'total_violence', 'verbal', 'fighting', 'weapons', 'human_torture',
    'animal_torture', 'violent_death', 'destruction', 'implied_aftermath',
    'sexual', 'terrorism', 'suicide', 'other'
]
CATEGORICAL_METADATA = ['rating', 'genres', 'tags']
CONTINUOUS_METADATA = ['averageScore', 'popularity', 'favourites', 'seasonYear']
# A reduced predictor set for the multivariate model to avoid multicollinearity
MULTIVARIATE_PREDICTORS = ['averageScore', 'popularity', 'seasonYear']

# --- API Query ---
GET_MEDIA_QUERY = '''
query Query($idIn: [Int], $page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo { hasNextPage }
    media(id_in: $idIn) {
      id, title { english, romaji }, averageScore, popularity,
      favourites, seasonYear, genres, tags { name }
    }
  }
}
'''

# =======================================
# Data Loading & Preparation
# =======================================

def fetch_anilist_metadata(id_list):
    """Fetches metadata from AniList for a given list of anime IDs, handling pagination."""
    print("Fetching metadata from AniList API...")
    all_media = []
    page = 1
    per_page = 50
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

    while True:
        variables = {"idIn": id_list, "page": page, "perPage": per_page}
        try:
            response = requests.post(ANILIST_API_URL, json={'query': GET_MEDIA_QUERY, 'variables': variables}, headers=headers)
            response.raise_for_status()
            data = response.json().get('data', {}).get('Page', {})
            media_data = data.get('media', [])
            if media_data:
                all_media.extend(media_data)
            if not data.get('pageInfo', {}).get('hasNextPage'):
                break
            page += 1
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from AniList API on page {page}: {e}")
            break
            
    print(f"Successfully fetched metadata for {len(all_media)} series.")
    return pd.DataFrame(all_media)

def load_and_prepare_data(filepath):
    """Loads violence data, fetches metadata, and merges them into analysis-ready dataframes."""
    print("--- Loading and Preparing Data ---")
    violence_df = pd.read_csv(filepath)
    for col in VIOLENCE_TYPES:
        violence_df[col] = pd.to_numeric(violence_df[col], errors='coerce')

    unique_ids = violence_df['anilist'].dropna().unique().tolist()
    metadata_df = fetch_anilist_metadata(unique_ids)
    if metadata_df.empty:
        raise SystemExit("Could not fetch metadata. Exiting.")

    metadata_df['title'] = metadata_df['title'].apply(lambda x: x.get('english') or x.get('romaji'))
    metadata_df['tags'] = metadata_df['tags'].apply(lambda x: [tag['name'] for tag in x] if isinstance(x, list) else [])

    merged_df = pd.merge(violence_df, metadata_df, left_on='anilist', right_on='id', how='left')

    agg_dict = {v_type: 'mean' for v_type in VIOLENCE_TYPES}
    agg_dict.update({col: 'first' for col in ['rating'] + CONTINUOUS_METADATA + ['genres', 'tags']})

    episode_df = merged_df.groupby(['show', 'anilist', 'season', 'episode']).agg(agg_dict).reset_index()
    for v_type in VIOLENCE_TYPES:
        episode_df[v_type] = episode_df[v_type].round().astype(int)

    print(f"Data prepared. {len(episode_df)} unique episodes for analysis.")
    return violence_df, episode_df

# =======================================
# Statistical Analysis Functions
# =======================================

def _prepare_categorical_data_for_analysis(episode_data, column_name):
    """Prepares categorical data by grouping infrequent categories into 'Other'."""
    print(f"Preparing '{column_name}' data for analysis...")
    
    if not episode_data[column_name].dropna().empty and isinstance(episode_data[column_name].dropna().iloc[0], list):
        exploded_df = episode_data.dropna(subset=[column_name]).explode(column_name)
    else:
        exploded_df = episode_data.dropna(subset=[column_name])
    if exploded_df.empty:
        print(f"Warning: No data found for column '{column_name}'.")
        return None

    counts = exploded_df[column_name].value_counts()
    categories_to_keep = counts[counts >= MIN_CATEGORY_COUNT].index
    
    grouped_col_name = f'{column_name}_grouped'
    exploded_df[grouped_col_name] = exploded_df[column_name].apply(lambda x: x if x in categories_to_keep else 'Other')
    if exploded_df[grouped_col_name].nunique() < 2:
        print(f"Warning: After grouping, less than two '{column_name}' categories remain. Skipping analysis.")
        return None
    
    return exploded_df

def calculate_icc(raw_data):
    """Calculates Intra-class Correlation Coefficient for inter-rater reliability."""
    print("Calculating Inter-rater Reliability (ICC)...")
    cleaned_data = raw_data.dropna(subset=['show', 'season', 'episode', 'reviewer']).copy()
    cleaned_data['episode_id'] = cleaned_data.apply(lambda row: f"{row['show']}_s{row['season']}_e{row['episode']}", axis=1)
    
    episodes_with_two_ratings = cleaned_data['episode_id'].value_counts()[lambda x: x == 2].index
    icc_data = cleaned_data[cleaned_data['episode_id'].isin(episodes_with_two_ratings)].copy()
    if icc_data.empty:
        print("ERROR: No episodes with exactly two ratings were found. Cannot calculate ICC.")
        return pd.DataFrame()

    # Rename raters generically to create a balanced dataset for the ICC function.
    icc_data['reviewer'] = 'Rater' + (icc_data.groupby('episode_id').cumcount() + 1).astype(str)
    
    results = []
    for v_type in VIOLENCE_TYPES:
        try:
            iter_data = icc_data[['episode_id', 'reviewer', v_type]].dropna()
            if iter_data[v_type].nunique() < 2: raise ValueError("Outcome has zero variance")
            
            icc_df = pg.intraclass_corr(data=iter_data, targets='episode_id', raters='reviewer', ratings=v_type).set_index('Type')
            icc_val = icc_df.loc['ICC2k']['ICC']
            interp = "Poor" if icc_val < 0.5 else "Moderate" if icc_val < 0.75 else "Good" if icc_val < 0.9 else "Excellent"
            results.append({'Violence Type': v_type, 'ICC': icc_val, 'Interpretation': interp})
        except Exception as e:
            print(f"Warning: Could not calculate ICC for '{v_type}'. Reason: {e}")
            results.append({'Violence Type': v_type, 'ICC': None, 'Interpretation': 'Calculation failed'})
    
    return pd.DataFrame(results)

def run_kruskal_wallis(data, outcome_variable, grouping_variable):
    """Runs the Kruskal-Wallis H-test for a single outcome and grouping variable."""
    desc = data.groupby(grouping_variable)[outcome_variable].agg(median='median', iqr=lambda x: x.quantile(0.75) - x.quantile(0.25))
    desc_row = desc.unstack().to_frame().T
    desc_row.columns = [f'{group}_{stat}' for stat, group in desc_row.columns]
    
    groups = [group[outcome_variable].dropna().values for _, group in data.groupby(grouping_variable) if not group.empty]
    if sum(1 for g in groups if pd.Series(g).nunique() > 1) < 2: return None, None 

    try:
        stat, p_val = kruskal(*groups)
    except ValueError:
        return None, None

    results_df = pd.DataFrame([{'Violence Type': outcome_variable, 'Statistic': stat, 'P-Value': p_val, 'Significant': 'Yes' if p_val < SIGNIFICANCE_THRESHOLD else 'No'}])
    combined_row = pd.concat([results_df.reset_index(drop=True), desc_row.reset_index(drop=True)], axis=1)
    dunn_df = sp.posthoc_dunn(data, val_col=outcome_variable, group_col=grouping_variable, p_adjust='bonferroni') if p_val < SIGNIFICANCE_THRESHOLD else None
    return combined_row, dunn_df

def _fit_and_extract_regression_results(model, predictor_names):
    """
    Helper to fit a statsmodels model with fallback solvers and safely extract results.
    This re-introduces the 'hail mary' fix of trying multiple solvers.
    """
    solvers = ['bfgs', 'newton'] # Primary and fallback solvers
    fitted_model = None
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=HessianInversionWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        for solver in solvers:
            try:
                fitted_model = model.fit(method=solver, disp=False, maxiter=200)
                if fitted_model.mle_retvals['converged']:
                    break 
            except Exception:
                fitted_model = None
    
    if fitted_model is None or not fitted_model.mle_retvals['converged']:
        return None, "Model failed to converge with all attempted solvers"

    try:
        conf_int_df = fitted_model.conf_int()
    except HessianInversionWarning:
        return None, "Hessian Inversion failed, cannot compute CIs"

    results = []
    for pred in predictor_names:
        if pred not in fitted_model.params.index: continue
        params = fitted_model.params.loc[pred]
        p_val = fitted_model.pvalues.loc[pred]
        conf_int = conf_int_df.loc[pred]
        if any(pd.isna([params, p_val])): continue

        results.append({
            'Predictor': pred, 'Coefficient': params, 'P-Value': p_val,
            'Significant': 'Yes' if p_val < SIGNIFICANCE_THRESHOLD else 'No',
            'Conf. Int. Lower': conf_int[0], 'Conf. Int. Upper': conf_int[1],
            'IRR': np.exp(params), 'IRR CI Lower': np.exp(conf_int[0]), 'IRR CI Upper': np.exp(conf_int[1])
        })
    return results, None

def run_correlation_and_regression(episode_data):
    """Runs Spearman correlation and robustly selects and fits regression models."""
    print("Running correlations and regressions...")
    spearman_results, uni_reg_results, multi_reg_results = [], [], []
    
    reg_data = episode_data.copy().dropna(subset=CONTINUOUS_METADATA)
    reg_data[CONTINUOUS_METADATA] = StandardScaler().fit_transform(reg_data[CONTINUOUS_METADATA])
    
    X_vif = sm.add_constant(reg_data[MULTIVARIATE_PREDICTORS])
    vif = pd.DataFrame({'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]}, index=X_vif.columns)
    print("Multicollinearity Check (VIF) for Multivariate Model:\n", vif)
    if (vif['VIF'][1:] > VIF_THRESHOLD).any():
        print(f"\nWARNING: High multicollinearity detected (VIF > {VIF_THRESHOLD}). Results may be unstable.\n")

    for v_type in VIOLENCE_TYPES:
        for pred in CONTINUOUS_METADATA:
            subset = episode_data[[pred, v_type]].dropna()
            if len(subset) > 5 and subset[pred].nunique() > 1 and subset[v_type].nunique() > 1:
                rho, p = spearmanr(subset[pred], subset[v_type])
                spearman_results.append({'Violence Type': v_type, 'Predictor': pred, 'Rho': rho, 'P-Value': p, 'Significant': 'Yes' if p < SIGNIFICANCE_THRESHOLD else 'No'})

        y = reg_data[v_type]
        if y.nunique() < 2:
            print(f"Skipping regression for '{v_type}': Outcome has no variance.")
            continue
        if (y > 0).sum() < MIN_NON_ZERO_FOR_REGRESSION:
            print(f"Skipping regression for '{v_type}': Insufficient non-zero data points for stable model fitting.")
            continue

        model_class = sm.ZeroInflatedNegativeBinomialP if (y == 0).any() else sm.NegativeBinomial
        infl_model = sm.add_constant(np.ones(len(y)), has_constant='add')
        
        for pred in CONTINUOUS_METADATA:
            X_uni = sm.add_constant(reg_data[pred])
            model = model_class(y, X_uni, exog_infl=infl_model) if model_class == sm.ZeroInflatedNegativeBinomialP else model_class(y, X_uni)
            res, err = _fit_and_extract_regression_results(model, [pred])
            if err: print(f"Could not fit univariate model for {v_type} ~ {pred}. Reason: {err}")
            elif res:
                for r in res: r['Violence Type'] = v_type
                uni_reg_results.extend(res)

        X_multi = sm.add_constant(reg_data[MULTIVARIATE_PREDICTORS])
        model = model_class(y, X_multi, exog_infl=infl_model) if model_class == sm.ZeroInflatedNegativeBinomialP else model_class(y, X_multi)
        res, err = _fit_and_extract_regression_results(model, MULTIVARIATE_PREDICTORS)
        if err: print(f"Could not fit multivariate model for {v_type}. Reason: {err}")
        elif res:
            for r in res: r['Violence Type'] = v_type
            multi_reg_results.extend(res)

    return pd.DataFrame(spearman_results), pd.DataFrame(uni_reg_results), pd.DataFrame(multi_reg_results)

def run_analysis(raw_data, episode_data):
    """Master function to run all statistical analyses."""
    print("\n--- Running All Analyses ---")
    all_results = {}
    all_results['icc'] = calculate_icc(raw_data)
    
    for cat_col in CATEGORICAL_METADATA:
        prepared_df = _prepare_categorical_data_for_analysis(episode_data, cat_col)
        all_results[f'{cat_col}_prepared_data'] = prepared_df
        if prepared_df is None:
            all_results[f'kruskal_wallis_{cat_col}'], all_results[f'dunn_{cat_col}'] = pd.DataFrame(), {}
            continue
        
        print(f"Running Kruskal-Wallis H tests for {cat_col}...")
        kw_list, dunn_dict = [], {}
        for v_type in VIOLENCE_TYPES:
            kw_res, dunn_res = run_kruskal_wallis(prepared_df, v_type, f'{cat_col}_grouped')
            if kw_res is not None: kw_list.append(kw_res)
            if dunn_res is not None: dunn_dict[v_type] = dunn_res
        
        all_results[f'kruskal_wallis_{cat_col}'] = pd.concat(kw_list, ignore_index=True) if kw_list else pd.DataFrame()
        all_results[f'dunn_{cat_col}'] = dunn_dict

    spearman, reg_uni, reg_multi = run_correlation_and_regression(episode_data)
    all_results.update({'spearman': spearman, 'reg_uni': reg_uni, 'reg_multi': reg_multi})
    print("All analyses complete.")
    return all_results

# =======================================
# Publication Output Functions
# =======================================

def format_correlation_for_publication(spearman_df):
    """Formats Spearman correlation results into a publication-ready matrix."""
    if spearman_df.empty: return pd.DataFrame()
    p_to_asterisks = lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    spearman_df['formatted'] = spearman_df.apply(lambda r: f"{r['Rho']:.2f}{p_to_asterisks(r['P-Value'])}", axis=1)
    return spearman_df.pivot_table(index='Violence Type', columns='Predictor', values='formatted', aggfunc='first')[CONTINUOUS_METADATA]

def format_regression_for_publication(regression_df, predictors):
    """Formats regression results into a publication-ready table."""
    if regression_df.empty: return pd.DataFrame()
    p_to_asterisks = lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    regression_df['formatted'] = regression_df.apply(lambda r: f"{r['IRR']:.2f} ({r['IRR CI Lower']:.2f}-{r['IRR CI Upper']:.2f}){p_to_asterisks(r['P-Value'])}", axis=1)
    pivot = regression_df.pivot_table(index='Violence Type', columns='Predictor', values='formatted', aggfunc='first')
    return pivot[predictors]

def generate_visualizations(episode_data, analysis_results):
    """Generates and saves all plots for the analysis."""
    print("\n--- Generating Visualizations ---")
    os.makedirs(FIGURES_OUTPUT_DIR, exist_ok=True)

    print("Generating correlation heatmap...")
    corr_df = episode_data[VIOLENCE_TYPES + CONTINUOUS_METADATA].corr(method='spearman')
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Spearman Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_OUTPUT_DIR, 'correlation_heatmap.png'), dpi=300)
    plt.close()

    top_v_types = episode_data[VIOLENCE_TYPES].sum().nlargest(5).index
    for cat in ['rating', 'genres']:
        df = analysis_results.get(f'{cat}_prepared_data')
        if df is None or df.empty: continue
        
        print(f"Generating boxplots for {cat}...")
        group_col = f'{cat}_grouped'
        order = df[group_col].value_counts().index
        for v_type in top_v_types:
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=df, x=group_col, y=v_type, order=order)
            plt.title(f'"{v_type.replace("_", " ").title()}" by {cat.capitalize()}', fontsize=16)
            plt.ylabel('Count per Episode'); plt.xlabel(cat.capitalize())
            plt.xticks(rotation=45, ha='right'); plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_OUTPUT_DIR, f'boxplot_{cat}_{v_type}.png'), dpi=300)
            plt.close()
            
    print("All visualizations have been saved.")

def write_publication_tables(episode_data, analysis_results):
    """Writes publication-ready summary tables to a separate Excel file."""
    print(f"\n--- Writing Publication Tables to {TABLES_OUTPUT_PATH} ---")
    with pd.ExcelWriter(TABLES_OUTPUT_PATH) as writer:
        for cat in CATEGORICAL_METADATA:
            df = analysis_results.get(f'{cat}_prepared_data')
            if df is None or df.empty: continue

            group_col = f'{cat}_grouped'
            n_counts = df.groupby(group_col).size().to_frame('N (Episodes)')
            summary_dfs = [n_counts]
            for v_type in episode_data[VIOLENCE_TYPES].sum().nlargest(5).index:
                desc = df.groupby(group_col)[v_type].agg(median='median', iqr=lambda x: x.quantile(0.75) - x.quantile(0.25)).round(2)
                v_type_clean = v_type.replace('_', ' ').title()
                desc.columns = [f"{v_type_clean} Median", f"{v_type_clean} IQR"]
                summary_dfs.append(desc)
            
            summary_table = pd.concat(summary_dfs, axis=1).sort_values(by='N (Episodes)', ascending=False)
            summary_table.index.name = cat.capitalize()
            summary_table.to_excel(writer, sheet_name=f'Desc - {cat.capitalize()}')

        format_correlation_for_publication(analysis_results['spearman']).to_excel(writer, sheet_name='Spearman Correlation')
        format_regression_for_publication(analysis_results['reg_uni'], CONTINUOUS_METADATA).to_excel(writer, sheet_name='Regression - Univariate')
        format_regression_for_publication(analysis_results['reg_multi'], MULTIVARIATE_PREDICTORS).to_excel(writer, sheet_name='Regression - Multivariate')

    print(f"Publication-ready tables have been saved to {TABLES_OUTPUT_PATH}")

def write_raw_results_to_excel(results, prepared_data):
    """Writes all raw analysis results to a multi-sheet Excel file."""
    print(f"\n--- Writing Raw Results to {OUTPUT_EXCEL_PATH} ---")
    with pd.ExcelWriter(OUTPUT_EXCEL_PATH) as writer:
        data_to_write = prepared_data.copy()
        for col in ['genres', 'tags']:
            if col in data_to_write.columns and not data_to_write[col].dropna().empty:
                data_to_write[col] = data_to_write[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        data_to_write.to_excel(writer, sheet_name='Prepared_Episode_Data', index=False)
        
        results['icc'].to_excel(writer, sheet_name='Inter-Rater_Reliability_ICC', index=False)
        
        for cat in CATEGORICAL_METADATA:
            if f'kruskal_wallis_{cat}' in results and not results[f'kruskal_wallis_{cat}'].empty:
                results[f'kruskal_wallis_{cat}'].to_excel(writer, sheet_name=f'KruskalWallis_{cat.capitalize()}', index=False)
            if f'dunn_{cat}' in results:
                for v, df in results[f'dunn_{cat}'].items():
                    df.to_excel(writer, sheet_name=f'Dunn_{cat.capitalize()}_{v[:15]}')
        
        results['spearman'].to_excel(writer, sheet_name='Spearman_Correlation', index=False)
        if 'reg_uni' in results and not results['reg_uni'].empty:
            results['reg_uni'].to_excel(writer, sheet_name='Regression_Univariate', index=False)
        if 'reg_multi' in results and not results['reg_multi'].empty:
            results['reg_multi'].to_excel(writer, sheet_name='Regression_Multivariate', index=False)

    print("Analysis results have been saved successfully.")

# =======================================
# Main Execution
# =======================================

def main():
    """Main function to run the entire analysis pipeline."""
    raw_data, episode_data = load_and_prepare_data(DATA_CSV_PATH)
    analysis_results = run_analysis(raw_data, episode_data)
    # Ensure raw results are written before tables, which might depend on them
    write_raw_results_to_excel(analysis_results, episode_data)
    write_publication_tables(episode_data, analysis_results)
    generate_visualizations(episode_data, analysis_results)

if __name__ == "__main__":
    main()

