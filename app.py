import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import MinMaxScaler

# 1. PAGE CONFIGURATION & SIDEBAR
st.set_page_config(page_title="Goodreads Analysis", page_icon="📚", layout="wide")

# --- SIDEBAR ---
with st.sidebar:
    st.title("📚 Goodreads Insights")
    st.markdown("A collaborative data analysis project exploring book popularity, market trends, and reader preferences.")
    st.markdown("---")
    st.markdown("**Built by:**")
    st.markdown("👩‍💻 Mahak & Aaditya")
    st.markdown("---")
    st.info("💡 **Tip:** Use the tabs on the main screen to navigate between Market Trends, Series Deep-Dives, and Industry Demographics.")

# Main Title
st.title("Market & Reader Analysis Dashboard")
st.markdown("---")

# Global Palette Settings
PRIMARY_COLOR = "#1D4C63" 
SECONDARY_COLOR = "#59c49d" 
MAIN_PALETTE = "viridis" 

# 2. LOAD & PREP DATA 
@st.cache_data
def load_and_prep_data():
    df = pd.read_csv("data/final_merge.csv") 
    
    scaler = MinMaxScaler()
    df[['numRatings_norm','bbeVotes_norm','likedPercent_norm']] = scaler.fit_transform(df[['numRatings','bbeVotes','likedPercent']])
    df['popularity_score'] = (0.7 * df['numRatings_norm'] + 0.2 * df['bbeVotes_norm'] + 0.1 * df['likedPercent_norm'])
    
    df['publishDate'] = pd.to_datetime(df['publishDate'], format='mixed', errors='coerce')
    df['year'] = df['publishDate'].dt.year
    df['Month'] = df['publishDate'].dt.month
    
    return df

df = load_and_prep_data()

# --- HERO SECTION & KPIs ---
st.markdown("### 📊 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Total Books Analyzed", value=f"{len(df):,}")
with col2:
    st.metric(label="Average Rating", value=f"{df['rating'].mean():.2f} ⭐")
with col3:
    st.metric(label="Total Reader Votes", value=f"{df['numRatings'].sum():,.0f}")
with col4:
    st.metric(label="Years Spanned", value=f"{int(df['year'].min())} - {int(df['year'].max())}")

st.markdown("<br>", unsafe_allow_html=True)

# 3. CREATE DASHBOARD TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Top Books & Attributes", 
    "📈 Market & Seasonality", 
    "🎭 Genre & Awards",
    "✍️ Industry & Demographics"
])

# ==========================================
# TAB 1: TOP BOOKS & ATTRIBUTES
# ==========================================
with tab1:
    st.header("Top Performers & Book Anatomy")
    
    # MK's Top 10 Books
    st.subheader("Top 10 Most Popular Books (Custom Score)")
    top10 = df.sort_values(by='popularity_score', ascending=False).head(10)
    fig1 = plt.figure(figsize=(10,6))
    sns.barplot(x='popularity_score', y='title', data=top10, hue='title', palette=MAIN_PALETTE, legend=False)
    plt.xlabel('Popularity Score')
    plt.ylabel('')
    plt.tight_layout()
    st.pyplot(fig1)
    
    # NEW EXPANDER FOR RAW DATA
    with st.expander("🔍 View Raw Data for Top 10 Books"):
        st.dataframe(
            top10[['title', 'author', 'rating', 'popularity_score', 'numofawards']], 
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Reader Preference by Book Length")
        grp = df.groupby(['length'])
        avgbylength = grp['likedPercent'].mean()
        fig_len = plt.figure(figsize=(8,5))
        sns.barplot(x=avgbylength.index, y=avgbylength.values, palette=MAIN_PALETTE)
        plt.xlabel("Book Length Category")
        plt.ylabel("Average Liked Percent (%)")
        for i, v in enumerate(avgbylength.values):
            plt.text(i, v + 0.15, f"{v:.2f}%", ha='center', fontsize=11)
        plt.ylim(90,96)
        sns.despine()
        st.pyplot(fig_len)
        
    with col2:
        st.subheader("Reader Preference by Character Count")
        filt = (df['numofchar'] != 0) & (df['numofchar'] <= 100)
        filtered = df.loc[filt, :].copy()
        filtered['char_bin'] = pd.cut(filtered['numofchar'], bins=[0,5,10,15,20])
        binavg = filtered.groupby(['char_bin'])['likedPercent'].mean().sort_values()
        fig_char = plt.figure(figsize=(8,5))
        sns.barplot(x=binavg.index.astype(str), y=binavg.values, palette=MAIN_PALETTE)
        plt.xlabel("Character Range")
        plt.ylabel("Average Liked Percent")
        plt.ylim(90,95)   
        for i,v in enumerate(binavg.values):
            plt.text(i, v+0.02, f"{v:.2f}", ha='center')
        sns.despine()
        st.pyplot(fig_char)

    st.markdown("---")

    # MK's Series Comparisons
    st.subheader("Series vs Standalone & Major Franchises")
    series_summary = df.groupby('isSeries')[['popularity_score', 'rating']].mean().reset_index()
    fig_series, axes = plt.subplots(1, 2, figsize=(12,5))
    axes[0].set_ylim(0.05, 0.1)
    axes[1].set_ylim(3.50, 4.10)
    
    sns.barplot(data=series_summary, x="isSeries", y="popularity_score", hue="isSeries", palette=[SECONDARY_COLOR, PRIMARY_COLOR], ax=axes[0], legend=False)
    axes[0].set_xticks([0,1])
    axes[0].set_xticklabels(["Standalone","Series"])
    axes[0].set_title("Average Popularity Score")
    
    sns.barplot(data=series_summary, x="isSeries", y="rating", hue="isSeries", palette=[SECONDARY_COLOR, PRIMARY_COLOR], legend=False, ax=axes[1])
    axes[1].set_xticks([0,1])
    axes[1].set_xticklabels(["Standalone","Series"])
    axes[1].set_title("Average Rating")
    plt.tight_layout()
    st.pyplot(fig_series)


# ==========================================
# TAB 2: MARKET & SEASONALITY
# ==========================================
with tab2:
    st.header("Macro Trends & Release Timing")
    
    # MK's Decades Performance
    st.subheader("Book Performance Across Decades (1950–2020)")
    df_year = df.copy()
    df_year['decade'] = (df_year['year'] // 10) * 10
    df_decade = df_year[(df_year["decade"] >= 1950) & (df_year["decade"] <= 2020)]
    decade_summary = df_decade.groupby("decade")[["rating", "likedPercent", "popularity_score"]].mean().reset_index()

    fig_dec, ax1 = plt.subplots(figsize=(14, 6))
    sns.lineplot(data=decade_summary, x="decade", y="rating", marker="o", color=SECONDARY_COLOR, ax=ax1)
    ax1.set_ylabel("Average Rating", color=SECONDARY_COLOR)
    ax2 = ax1.twinx()
    sns.lineplot(data=decade_summary, x="decade", y="popularity_score", marker="o", color=PRIMARY_COLOR, ax=ax2)
    ax2.set_ylabel("Popularity Score", color=PRIMARY_COLOR)
    ax1.set_xlabel("Decade")
    plt.grid(alpha=0.3)
    st.pyplot(fig_dec)
    
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        # Bestie's Seasonality
        st.subheader("Seasonal Popularity by Release Month")
        date_df = df.copy()
        date_df = date_df[(date_df['year'].notna()) & (date_df['year'] >= 2010)]
        seasonal_pop = date_df.groupby('Month')['numRatings'].mean()
        fig_season = plt.figure(figsize=(10,6))
        month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        ax = sns.barplot(y=month_labels, x=seasonal_pop.values, palette=MAIN_PALETTE)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x/1000)}K'))
        plt.xlabel("Average Reach")
        plt.ylabel("Month")
        sns.despine()
        st.pyplot(fig_season)

    with col2:
        # MK's Pricing Trend
        st.subheader("Year-wise Trend of Average Book Price")
        df_filt_year = df_year[(df_year['year']>=1970) & (df_year['year']<=2026)] 
        year_price = df_filt_year.groupby("year")['price'].mean().reset_index()
        fig_price = plt.figure(figsize=(10,6))
        sns.lineplot(data=year_price, x="year", y="price", color=PRIMARY_COLOR, linewidth=2.5)
        plt.xlabel("Year")
        plt.ylabel("Average Price")
        plt.ylim(1.7, 2.5)
        plt.grid(alpha=0.5)
        st.pyplot(fig_price)


# ==========================================
# TAB 3: GENRE & AWARDS
# ==========================================
with tab3:
    st.header("The Impact of Genres and Literary Awards")
    
    # Bestie's Award Impact
    st.subheader("Impact of Literary Awards on Book Reach")
    filtered_aw = df[df['numofawards'] <= 15]
    avgpopgrp = filtered_aw.groupby('numofawards')['numRatings'].mean()
    fig_aw = plt.figure(figsize=(10,6))
    ax = sns.lineplot(x=avgpopgrp.index, y=avgpopgrp.values, marker="o", linewidth=2.5, color=PRIMARY_COLOR)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x/1000)}K'))
    plt.xlabel("Number of Awards")
    plt.ylabel("Average Reach")
    plt.grid(True, linestyle="--", alpha=0.35)
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig_aw)
    
    st.markdown("---")

    df_genre = df.copy()
    df_genre['genres'] = df_genre['genres'].replace('[]', None).astype(str)
    df_genre['genres'] = df_genre['genres'].str.replace('[', '').str.replace(']', '').str.replace("'", "").str.strip()
    df_genre['genres'] = df_genre['genres'].str.split(',')
    genre_exploded = df_genre.explode('genres')
    genre_exploded['genres'] = genre_exploded['genres'].str.strip()
    genre_exploded = genre_exploded[genre_exploded['genres'] != '']
    
    col1, col2 = st.columns(2)
    with col1:
        # MK's Genre Satisfaction
        st.subheader("Performance vs Satisfaction by Genre")
        genre_summary = genre_exploded.groupby('genres').agg(genre_count=('title','count'), avg_popularity_score=('popularity_score','mean'), avg_likedPercent=('likedPercent','mean')).reset_index()
        genre_summary = genre_summary[genre_summary['genre_count'] >= 50]
        top_genres = genre_summary.sort_values(by='genre_count', ascending=False).head(15).sort_values(by='avg_popularity_score', ascending=True)
        
        fig_gen1, ax1 = plt.subplots(figsize=(10,8))
        ax2 = ax1.twiny()
        y_positions = range(len(top_genres))
        ax1.scatter(top_genres['avg_popularity_score'], y_positions, color=PRIMARY_COLOR, s=100)
        ax2.scatter(top_genres['avg_likedPercent'], y_positions, color=SECONDARY_COLOR, s=100)
        ax1.set_yticks(y_positions)
        ax1.set_yticklabels(top_genres['genres'])
        ax1.set_xlabel('Market Performance (Score)', color=PRIMARY_COLOR)
        ax2.set_xlabel('Reader Satisfaction (%)', color=SECONDARY_COLOR)
        ax1.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig_gen1)
        
    with col2:
        # Bestie's Genre Awards
        st.subheader("Genres Receiving the Most Literary Awards")
        genre_awards = genre_exploded.groupby('genres')['numofawards'].sum().sort_values(ascending=False).head(10)
        fig_gen2 = plt.figure(figsize=(9,8))
        sns.barplot(x=genre_awards.values, y=genre_awards.index, palette=MAIN_PALETTE)
        for i, v in enumerate(genre_awards.values):
            plt.text(v + 1, i, str(v), va='center', fontsize=11)
        plt.xlabel("Total Awards")
        plt.ylabel("")
        sns.despine()
        st.pyplot(fig_gen2)


# ==========================================
# TAB 4: INDUSTRY & DEMOGRAPHICS
# ==========================================
with tab4:
    st.header("Industry Players & Demographics")
    
    # MK's Authors
    st.subheader("Most Frequently Published Authors")
    df_auth = df.copy()
    df_auth['author'] = df_auth['author'].str.split(",")
    df_auth = df_auth.explode("author")
    df_auth["author"] = df_auth["author"].str.strip().str.replace(r"\s*\(.*?\)", "", regex=True)
    author_count = df_auth['author'].value_counts().reset_index()
    author_count.columns = ["author", "book_count"]
    valid_authors = ~author_count['author'].str.contains("more|anonymous", case=False, na=False)
    top_authors = author_count[valid_authors].head(15)
    
    fig_auth = plt.figure(figsize=(10,7))
    plt.hlines(y=top_authors["author"], xmin=0, xmax=top_authors["book_count"], color=PRIMARY_COLOR, alpha=0.5)
    plt.scatter(top_authors["book_count"], top_authors["author"], color=PRIMARY_COLOR, s=100)
    plt.xlabel("Number of Books")
    plt.gca().invert_yaxis()
    plt.grid(axis='y', alpha=0.5)
    st.pyplot(fig_auth)
    
    # NEW EXPANDER FOR AUTHOR DATA
    with st.expander("🔍 View Top 15 Authors Data"):
        st.dataframe(top_authors, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        # MK's Publishers
        st.subheader("Publisher Output vs Average Rating")
        publisher_ratings = df.groupby('publisher')["rating"].agg(["mean", "count"])
        publisher_ratings = publisher_ratings[publisher_ratings['count'] >= 25]
        fig_pub = plt.figure(figsize=(8,6))
        sns.scatterplot(data=publisher_ratings, x='count', y='mean', hue='mean', legend=False, palette=MAIN_PALETTE)
        plt.xlabel("Number of Books Published")
        plt.ylabel("Average Rating")
        plt.xscale("log")
        plt.grid(alpha=0.3)
        st.pyplot(fig_pub)
        
    with col2:
        # MK's Ratings Volume
        st.subheader("Volume of Ratings vs Average Rating")
        fig_vol = plt.figure(figsize=(8,6))
        sns.scatterplot(data=df, x="numRatings", y="rating", alpha=0.3, s=12, hue='rating', palette=MAIN_PALETTE, legend=False)
        plt.xscale("log")
        plt.xlabel("Number of Ratings (log scale)")
        plt.ylabel("Average Rating")
        plt.grid(alpha=0.3)
        st.pyplot(fig_vol)