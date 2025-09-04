import json

import pandas as pd
import scipy as sp
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from dotenv import load_dotenv
import requests

load_dotenv()

API_URL = 'https://graphql.anilist.co'

GET_MEDIA_QUERY = '''
query Query($idIn: [Int], $page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    media(id_in: $idIn) {
      averageScore
      duration
      episodes
      genres
      format
      favourites
      popularity
      seasonYear
      title {
        english
      }
      season
      tags {
        name
      }
      type
      id
    }
  }
}
'''

violence_types = ['total_violence', 'verbal', 'fighting', 'weapons', 'human_torture', 'animal_torture',
                  'violent_death', 'destruction', 'implied_aftermath', 'sexual', 'terrorism', 'suicide', 
                  'other']

# Load violence data
violence = pd.read_csv('master_data.csv')
reviewers = violence['reviewer'].unique().tolist()

# Get anime data
headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
}

variables = {
  "idIn": violence['anilist'].tolist(),
  "page": 1,
  "perPage": 50
}

response = requests.post(API_URL, json={'query': GET_MEDIA_QUERY, 'variables': variables}, headers=headers)
response_dict = json.loads(response.text)
anime_data = response_dict['data']['Page']['media']
for media in anime_data:
    media['title'] = media['title']['english']
    media['tags'] = [tag['name'] for tag in media['tags']]

anime = pd.DataFrame(anime_data)

# Descriptive Stats
violence_descriptive = violence.describe(include='all')
print(violence_descriptive)

anime_descriptive = anime.describe(include='all')
print(anime_descriptive)

merged = violence.merge(anime, left_on='anilist', right_on='id')
merged['progress'] = merged['episode'] / merged['episodes']

result = (
    merged.groupby('anilist', as_index=False)
    .apply(lambda x: pd.Series({
        'title': x['title'].iloc[0],
        'total_violence': x['total_violence'].sum(),
        'unique_episodes': x['episode'].nunique(),
        'duration_per_episode': x['duration'].iloc[0]
    }))
    .reset_index()
)
result['total_minutes_analyzed'] = result['unique_episodes'] * result['duration_per_episode']
result['violence_per_minute'] = result['total_violence'] / result['total_minutes_analyzed']

print(result['anilist'])

anime = pd.merge(
    anime, 
    result[['anilist', 'violence_per_minute']],
    left_on='id', 
    right_on='anilist', 
    how='left'
).drop(columns=['anilist'])

merged_descriptive = merged.describe(include='all')
print(merged_descriptive)

exploded_genres = merged.explode('genres')
exploded_tags = merged.explode('tags')

# ICC
icc = []
icc_per_type = []

# Process each show
for show in violence['show'].unique():
    show_data = violence.loc[violence['show'] == show]
    
    # Extract all unique reviewer pairs with shared episodes
    pairs = {}
    for episode in show_data['episode'].unique():
        reviewers = show_data[show_data['episode'] == episode]['reviewer'].unique()
        if len(reviewers) == 2:
            pair = tuple(sorted(reviewers))  # Ensure consistent order
            if pair not in pairs:
                pairs[pair] = []
            pairs[pair].append(episode)

    # Calculate ICC for each pair with >= 2 episodes
    for (r1, r2), episodes in pairs.items():
        if len(episodes) >= 2:
            # Filter data for this pair and episodes
            pair_data = show_data[
                (show_data['episode'].isin(episodes)) &
                (show_data['reviewer'].isin([r1, r2]))
            ]
            
            # Compute ICC for each metric
            for metric in violence_types:
                # Reshape data for ICC_PER_TYPE calculation (long format)
                long_df = pair_data[['episode', 'reviewer', metric]].dropna()
                if len(long_df) < 2:
                    continue  # Skip if insufficient data
                
                try:
                    if long_df[metric].var() == 0:
                        icc_value = 1.0  # All values identical
                    else:
                        icc_per_type_result = pg.intraclass_corr(
                            data=long_df,
                            targets='episode',
                            raters='reviewer',
                            ratings=metric
                        )
                        # Extract ICC(3,1) value (two-way mixed, absolute agreement)
                        icc_per_type_value = icc_per_type_result[icc_per_type_result['Type'] == 'ICC3']['ICC'].values[0]
                except Exception as e:
                    agreement = (
                        long_df.groupby('episode')[metric]
                        .apply(lambda x: x.nunique() == 1)  # Check if ratings match
                        .mean()  # Average agreement across episodes
                    )
                    icc_per_type_value = agreement
                
                icc_per_type.append({
                    'Show': show,
                    'Reviewer 1': r1,
                    'Reviewer 2': r2,
                    'Metric': metric,
                    'ICC': icc_per_type_value,
                    'Episodes Evaluated': len(episodes)
                })

            # Melt data
            melted = pair_data.melt(
                id_vars=['episode', 'reviewer'],
                value_vars=violence_types,
                var_name='metric',
                value_name='violence_count'
            )
            print(melted)

            # Create unique target ID: episode + metric
            melted['target'] = melted['episode'].astype(str) + '_' + melted['metric']

            if len(melted['target'].unique()) < 2:
                print("Not enough targets for ICC in: " + melted['target'])
                continue  # Skip if <2 unique targets
            
            try:
                # Check if all ratings are identical (zero variance)
                if melted['violence_count'].var() == 0:
                    # Perfect agreement: set ICC = 1 (or use percent agreement)
                    icc_value = 1.0
                else:
                    # Compute ICC only if variance is non-zero
                    icc_result = pg.intraclass_corr(
                        data=melted,
                        targets='target',
                        raters='reviewer',
                        ratings='violence_count'
                    )
                    # Extract ICC(3,1) value
                    icc_value = icc_result[icc_result['Type'] == 'ICC3']['ICC'].values[0]
            except Exception as e:
                # Fallback to percent agreement if ICC fails
                agreement = (melted.groupby('target')['violence_count']
                             .apply(lambda x: x.nunique() == 1)  # Check if all raters agree
                             .mean())  # Average agreement across targets
                icc_value = agreement  # Treat percent agreement as a proxy
            
            icc.append({
                'Show': show,
                'Reviewer 1': r1,
                'Reviewer 2': r2,
                'ICC': icc_value,
                'Episodes': len(episodes),
                'Metrics Used': len(melted['metric'].unique())
            })

# Convert results to DataFrame
icc_df = pd.DataFrame(icc)
icc_per_type_df = pd.DataFrame(icc_per_type)

# Display results
print(icc_df[['Show', 'Reviewer 1', 'Reviewer 2', 'ICC', 'Episodes', 'Metrics Used']].dropna())
print(icc_per_type_df[['Show', 'Reviewer 1', 'Reviewer 2', 'Metric', 'ICC']].dropna())

# Outlier IQR
q1 = violence['total_violence'].quantile(0.25)
q3 = violence['total_violence'].quantile(0.75)
iqr = q3 - q1
outliers = violence[(violence['total_violence'] < q1 - 1.5 * iqr) | (violence['total_violence'] > q3 + 1.5 * iqr)]
print(f"Outliers in Total Violence:\n{outliers[['show', 'episode', 'total_violence']]}")

# Means
print("Average Instances Per Category")

avg_by_show = merged.groupby('show')[violence_types].mean()
avg_by_genre = exploded_genres.groupby('genres')[violence_types].mean()
avg_by_tag = exploded_tags.groupby('tags')[violence_types].mean()

print(avg_by_show)
print(avg_by_genre)
print(avg_by_tag)

# Pearson correlation
def calculate_correlations(df, target_cols, reference_col):
    results = []
    for col in target_cols:
        r, p = sp.stats.pearsonr(df[col], df[reference_col])
        results.append({'Violence_Type': col, 'R': r, 'p-value': p})
    return pd.DataFrame(results)

corr_year_df = calculate_correlations(merged, violence_types, 'seasonYear')
corr_score_df = calculate_correlations(merged, violence_types, 'averageScore')
corr_favorites_df = calculate_correlations(merged, violence_types, 'favourites')
corr_popularity_df = calculate_correlations(merged, violence_types, 'popularity')
corr_progress_df = calculate_correlations(merged, violence_types, 'progress')

print(f"Pearson R Year vs Violence:\n{corr_year_df}")
print(f"Pearson R Score vs Violence:\n{corr_score_df}")
print(f"Pearson R Favorites vs Violence:\n{corr_favorites_df}")
print(f"Pearson R Popularity vs Violence:\n{corr_popularity_df}")
print(f"Pearson R Progress vs Violence:\n{corr_progress_df}")

violence_types_df = violence[violence_types]
corr_violence = violence_types_df.corr(method='pearson')

print(f"Correlation Matrix Violence Types:\n{corr_violence}")

# ANOVA
anova_by_rating = []
anova_by_genre = []
anova_by_tag = []

tukey_rating = []
tukey_genre = []
tukey_tag = []

for col in violence_types:
    # Group by rating
    rating_groups = [group[col].dropna() for _, group in merged.groupby('rating')]
    anova_rating = sp.stats.f_oneway(*rating_groups)

    # Tukey for rating
    rating_data = merged[['rating', col]].dropna()
    if len(rating_data['rating'].unique()) > 1:
        tukey = pairwise_tukeyhsd(endog=rating_data[col],
                                 groups=rating_data['rating'],
                                 alpha=0.05)
        # Convert results to dataframe
        tukey_df = pd.DataFrame(tukey._results_table.data[1:], 
                              columns=tukey._results_table.data[0])
        tukey_df['variable'] = col
        tukey_df['grouping'] = 'rating'
        tukey_rating.append(tukey_df)

    # Group by genre
    genre_groups = [group[col].dropna() for _, group in exploded_genres.groupby('genres')]
    anova_genre = sp.stats.f_oneway(*genre_groups)

    # Tukey for genre
    genre_data = exploded_genres[['genres', col]].dropna()
    if len(genre_data['genres'].unique()) > 1:
        tukey = pairwise_tukeyhsd(endog=genre_data[col],
                                 groups=genre_data['genres'],
                                 alpha=0.05)
        tukey_df = pd.DataFrame(tukey._results_table.data[1:], 
                              columns=tukey._results_table.data[0])
        tukey_df['variable'] = col
        tukey_df['grouping'] = 'genre'
        tukey_genre.append(tukey_df)

    # Group by tags
    tag_groups = [group[col].dropna() for _, group in exploded_tags.groupby('tags')]
    anova_tag = sp.stats.f_oneway(*tag_groups)

    # Tukey for tags
    tag_data = exploded_tags[['tags', col]].dropna()
    if len(tag_data['tags'].unique()) > 1:
        tukey = pairwise_tukeyhsd(endog=tag_data[col],
                                 groups=tag_data['tags'],
                                 alpha=0.05)
        tukey_df = pd.DataFrame(tukey._results_table.data[1:], 
                              columns=tukey._results_table.data[0])
        tukey_df['variable'] = col
        tukey_df['grouping'] = 'tag'
        tukey_tag.append(tukey_df)
    
    # Save results
    anova_by_rating.append({'Violence_Type': col,
                            'F-Statistic': anova_rating.statistic,
                            'p-value': anova_rating.pvalue})
    anova_by_genre.append({'Violence_Type': col, 
                           'F-Statistic': anova_genre.statistic, 
                           'p-value': anova_genre.pvalue})
    anova_by_tag.append({'Violence_Type': col, 
                         'F-Statistic': anova_tag.statistic, 
                         'p-value': anova_tag.pvalue})

anova_rating_df = pd.DataFrame(anova_by_rating)
anova_genre_df = pd.DataFrame(anova_by_genre)
anova_tag_df = pd.DataFrame(anova_by_tag)

print(f"ANOVA by Rating:\n{anova_rating_df.sort_values('p-value')}")
print(f"ANOVA by Genre:\n{anova_genre_df.sort_values('p-value')}")
print(f"ANOVA by Tag:\n{anova_tag_df.sort_values('p-value')}")

# Convert Tukey results
tukey_rating_df = pd.concat(tukey_rating, ignore_index=True)
tukey_genre_df = pd.concat(tukey_genre, ignore_index=True)
tukey_tag_df = pd.concat(tukey_tag, ignore_index=True)

# Display significant results only
significant_tukey_rating = tukey_rating_df[tukey_rating_df['reject']]
significant_tukey_genre = tukey_genre_df[tukey_genre_df['reject']]
significant_tukey_tag = tukey_tag_df[tukey_tag_df['reject']]

print("\nSignificant Tukey HSD Results:")
print(significant_tukey_rating)
print(significant_tukey_genre)
print(significant_tukey_tag)

# Write out
with pd.ExcelWriter("Stats.xlsx") as writer:
    violence.to_excel(writer, sheet_name='Violence')
    violence_descriptive.to_excel(writer, sheet_name='Violence Descriptive')

    anime.to_excel(writer, sheet_name='Anime')
    anime_descriptive.to_excel(writer, sheet_name='Anime Descriptive')

    merged.to_excel(writer, sheet_name='Merged')
    merged_descriptive.to_excel(writer, sheet_name='Merged Descriptive')

    icc_df.to_excel(writer, sheet_name='ICC')
    icc_per_type_df.to_excel(writer, sheet_name='ICC Per Type')

    avg_by_show.to_excel(writer, sheet_name='Show Means')
    avg_by_genre.to_excel(writer, sheet_name='Genre Means')
    avg_by_tag.to_excel(writer, sheet_name='Tag Means')

    outliers.to_excel(writer, sheet_name='Outliers')

    corr_year_df.to_excel(writer, sheet_name='Year Correlation')
    corr_score_df.to_excel(writer, sheet_name='Score Correlation')
    corr_favorites_df.to_excel(writer, sheet_name='Favorites Correlation')
    corr_popularity_df.to_excel(writer, sheet_name='Popularity Correlation')
    corr_violence.to_excel(writer, sheet_name='Violence Type Correlation')
    corr_progress_df.to_excel(writer, sheet_name='Progress Correlation')

    anova_rating_df.to_excel(writer, sheet_name='Rating ANOVA')
    anova_genre_df.to_excel(writer, sheet_name='Genre ANOVA')
    anova_tag_df.to_excel(writer, sheet_name='Tag ANOVA')

    tukey_rating_df.to_excel(writer, sheet_name='Rating Tukey')
    tukey_genre_df.to_excel(writer, sheet_name='Genre Tukey')
    tukey_tag_df.to_excel(writer, sheet_name='Tag Tukey')

    significant_tukey_rating.to_excel(writer, sheet_name='Significant Rating Tukey')
    significant_tukey_genre.to_excel(writer, sheet_name='Significant Genre Tukey')
    significant_tukey_tag.to_excel(writer, sheet_name='Significant Tag Tukey')
