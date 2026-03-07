import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from matplotlib.ticker import FuncFormatter
import re

# --- MACHINE LEARNING IMPORTS ---
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="The Data Library", page_icon="📚", layout="wide")

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

# --- MACHINE LEARNING CACHE ENGINES ---
@st.cache_resource
def train_recommender(_df):
    df_rec = _df.copy()
    df_rec['genres'] = df_rec['genres'].fillna('')
    df_rec['pages'] = pd.to_numeric(df_rec['pages'], errors='coerce').fillna(df_rec['pages'].median())
    df_rec['rating'] = pd.to_numeric(df_rec['rating'], errors='coerce').fillna(df_rec['rating'].median())
    df_rec['likedPercent'] = pd.to_numeric(df_rec['likedPercent'], errors='coerce').fillna(df_rec['likedPercent'].median())

    tfidf = TfidfVectorizer(stop_words='english', max_features=100)
    genre_matrix = tfidf.fit_transform(df_rec['genres'])

    scaler = MinMaxScaler()
    num_features = scaler.fit_transform(df_rec[['pages', 'rating', 'likedPercent']])

    book_dna = hstack([genre_matrix, num_features * 0.5])
    knn_model = NearestNeighbors(n_neighbors=11, algorithm='brute', metric='cosine')
    knn_model.fit(book_dna)
    
    return knn_model, book_dna, df_rec

@st.cache_resource
def train_predictor(_df):
    model_df = _df[['genres','pages','price','numRatings','numofchar']].copy()
    model_df = model_df.dropna()
    
    # Clean and explode genres
    model_df['genres'] = model_df['genres'].str.replace('[', '', regex=False).str.replace(']', '', regex=False).str.replace("'", '', regex=False)
    model_df['genres'] = model_df['genres'].str.split(',')
    model_df = model_df.explode('genres')
    model_df['genres'] = model_df['genres'].str.strip()
    
    # Create the Bestseller Target (Top 20%)
    threshold = model_df['numRatings'].quantile(0.80)
    model_df['Bestseller'] = (model_df['numRatings'] > threshold).astype(int)
    
    # Filter for Aaditya's 17 specific genres
    listofgenre = ['Fantasy','Fiction','Young Adult','Audiobook','Horror','Novels','Romance','Adult','Historical','Adventure','Action','Crime','Comedy','Vampires','War','Drama','Dragons']
    model_df = model_df[model_df['genres'].isin(listofgenre)]
    
    # --- 🧠 NEW: Calculate Bestseller Averages per Genre ---
    # We create a dictionary that holds the average pages and price for successful books in each genre
    bestseller_stats = model_df[model_df['Bestseller'] == 1].groupby('genres')[['pages', 'price']].mean().to_dict('index')
    
    # Prep for ML
    model_df = model_df[['pages','price','numofchar','genres','Bestseller']]
    model_df = pd.get_dummies(model_df, columns=['genres'], dtype=int)
    
    # Split Features and Target
    X = model_df.drop('Bestseller', axis=1)
    y = model_df['Bestseller']
    
    # Train the Random Forest
    model = RandomForestClassifier(n_estimators=500, random_state=42, max_depth=12, class_weight='balanced')
    model.fit(X, y)
    
    # Notice we are now returning 'bestseller_stats' too!
    return model, X.columns, listofgenre, threshold, bestseller_stats


# ==========================================
# 🧭 MAIN SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.title("📚 The Data Library")
    st.markdown("A collaborative workspace for book market analysis and predictive modeling.")
    st.markdown("---")
    
    st.subheader("🧭 Navigation")
    app_mode = st.radio("Select a Tool:", [
        "📊 Market Analysis Dashboard",
        "🤖 AI Book Recommender",
        "🎯 Hit Maker Predictor"
    ])
    
    st.markdown("---")
    st.markdown("**Built by:**\n👩‍💻 Mahak & Aaditya")


# ==========================================
# 🚀 APP ROUTE 1: THE DASHBOARD
# ==========================================
if app_mode == "📊 Market Analysis Dashboard":
    st.title("Market & Reader Analysis Dashboard")
    st.markdown("---")

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

    # CREATE DASHBOARD TABS
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Top Books & Attributes", 
        "📈 Market & Seasonality", 
        "🎭 Genre & Awards",
        "✍️ Industry & Demographics"
    ])

    with tab1:
        st.header("Top Performers & Book Anatomy")
        st.subheader("Top 10 Most Popular Books (Custom Score)")
        top10 = df.sort_values(by='popularity_score', ascending=False).head(10)
        fig1 = plt.figure(figsize=(10,6))
        sns.barplot(x='popularity_score', y='title', data=top10, hue='title', palette=MAIN_PALETTE, legend=False)
        plt.xlabel('Popularity Score')
        plt.ylabel('')
        plt.tight_layout()
        st.pyplot(fig1)
        
        with st.expander("🔍 View Raw Data for Top 10 Books"):
            st.dataframe(top10[['title', 'author', 'rating', 'popularity_score', 'numofawards']], use_container_width=True, hide_index=True)

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

    with tab2:
        st.header("Macro Trends & Release Timing")
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

    with tab3:
        st.header("The Impact of Genres and Literary Awards")
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

    with tab4:
        st.header("Industry Players & Demographics")
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
        
        with st.expander("🔍 View Top 15 Authors Data"):
            st.dataframe(top_authors, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
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
            st.subheader("Volume of Ratings vs Average Rating")
            fig_vol = plt.figure(figsize=(8,6))
            sns.scatterplot(data=df, x="numRatings", y="rating", alpha=0.3, s=12, hue='rating', palette=MAIN_PALETTE, legend=False)
            plt.xscale("log")
            plt.xlabel("Number of Ratings (log scale)")
            plt.ylabel("Average Rating")
            plt.grid(alpha=0.3)
            st.pyplot(fig_vol)


# ==========================================
# 🚀 APP ROUTE 2: THE RECOMMENDER
# ==========================================
elif app_mode == "🤖 AI Book Recommender":
    st.title("🤖 The AI Recommender Engine")
    st.markdown("Select a book you love, and our K-Nearest Neighbors algorithm will find its statistical twins based on genre DNA, page count and reader sentiment.")
    st.markdown("---")
    
    knn_model, book_dna, df_rec = train_recommender(df) 
    
    # 1. grabbing all unique titles
    raw_book_list = df_rec['title'].dropna().unique()
    
    # 2. filtering out non-english/weird character titles using regex
    clean_book_list = sorted([title for title in raw_book_list if re.match(r'^[\x00-\x7F]+$', title)])
    
    selected_book = st.selectbox(
        "Search for a book in the library:", 
        options=clean_book_list,
        index=None, 
        placeholder="Start typing a book title..."
    )
    
    # 3. streamlit returns None if nothing is selected yet
    if selected_book:
        if st.button("Find Matches 🔍"):
            with st.spinner('Calculating vector distances in the 103-dimensional universe... 🌌'):
                idx = df_rec[df_rec['title'] == selected_book].index[0]
                target_dna = book_dna.tocsr()[idx]
                
                distances, indices = knn_model.kneighbors(target_dna)
                
                st.subheader(f"Because you liked *{selected_book}*...")
                st.divider()
                
                # display the top matches (skipping index 0)
                for i in range(1, len(distances[0])):
                    match_idx = indices[0][i]
                    match_row = df_rec.iloc[match_idx]
                    
                    match_score = (1 - distances[0][i]) * 100
                    
                    img_url = match_row['coverImg'] if pd.notna(match_row['coverImg']) else "https://via.placeholder.com/150x200?text=No+Cover"
                    
                    col1, col2, col3 = st.columns([1, 3, 1])
                    
                    with col1:
                        st.image(img_url, use_container_width=True)
                        
                    with col2:
                        st.markdown(f"#### {i}. {match_row['title']}")
                        st.markdown(f"**Author:** {match_row['author']}")
                        st.caption(f"**DNA:** {str(match_row['genres'])[:80]}...") 
                        
                    with col3:
                        st.metric(label="🔥 Match Score", value=f"{match_score:.1f}%")
                        st.caption(f"⭐ Rating: {match_row['rating']}")
                        st.caption(f"📖 {int(match_row['pages'])} pages")
                        
                    st.divider()


# ==========================================
# 🚀 APP ROUTE 3: THE PREDICTOR (Aaditya's Space)
# ==========================================
# ==========================================
# 🚀 APP ROUTE 3: THE PREDICTOR (Aaditya's Space)
# ==========================================
elif app_mode == "🎯 Hit Maker Predictor":
    st.title("🎯 The Hit Maker: Bestseller Predictor")
    st.markdown("Play the role of a publisher. Enter the specs of a hypothetical book, and our Random Forest model will predict its market success based on the top 20% of historical Goodreads data.")
    st.markdown("---")
    
    # Load Aaditya's model AND the new target stats
    rf_model, expected_cols, valid_genres, target_threshold, bestseller_stats = train_predictor(df)
    
    # Create the Input UI
    col1, col2 = st.columns(2)
    with col1:
        selected_genre = st.selectbox("Select Primary Genre:", options=sorted(valid_genres))
        input_pages = st.number_input("Page Count:", min_value=50, max_value=2000, value=350)
    with col2:
        input_price = st.number_input("Expected Book Price ($):", min_value=0.00, max_value=100.00, value=14.99)
        input_char = st.number_input("Number of Named Characters:", min_value=0, max_value=100, value=5)
        
    if st.button("Predict Market Success 🚀"):
        with st.spinner("Consulting the Random Forest... 🌲"):
            # Setup input data
            input_data = pd.DataFrame(columns=expected_cols)
            input_data.loc[0] = 0 
            input_data.at[0, 'pages'] = input_pages
            input_data.at[0, 'price'] = input_price
            input_data.at[0, 'numofchar'] = input_char
            
            genre_col_name = f"genres_{selected_genre}"
            if genre_col_name in input_data.columns:
                input_data.at[0, genre_col_name] = 1
                
            # Predict
            prob = rf_model.predict_proba(input_data)[0][1] 
            
            st.markdown("---")
            st.subheader("🔮 Prediction Results")
            
            # --- 🏆 NEW: Granular Success Tiers ---
            if prob >= 0.70:
                st.success("### 🌟 CERTIFIED BLOCKBUSTER")
                st.markdown(f"Incredible! The model gives this a **{prob*100:.1f}%** probability of hitting the top 20% of global readership (over {int(target_threshold):,} ratings).")
            elif prob >= 0.50:
                st.info("### 📈 STRONG CONTENDER")
                st.markdown(f"Solid concept. There is a **{prob*100:.1f}%** probability this breaks into the top tier of books.")
            elif prob >= 0.35:
                st.warning("### 📊 MODERATE POTENTIAL")
                st.markdown(f"At **{prob*100:.1f}%**, this book has a fighting chance, but it might face tough competition. Let's look at the data to optimize it.")
            else:
                st.error("### 📉 NICHE MARKET APPEAL")
                st.markdown(f"With only a **{prob*100:.1f}%** chance, this is likely a passion project for a highly specific audience rather than a mainstream hit.")
            
            st.progress(float(prob))
            
            # --- 💡 NEW: Dynamic Data-Driven Suggestions ---
            st.markdown("#### 💡 Data-Driven Optimization")
            st.markdown(f"How does your book compare to actual **{selected_genre}** bestsellers?")
            
            target_p = bestseller_stats[selected_genre]['pages']
            target_price = bestseller_stats[selected_genre]['price']
            
            # Dynamic Page Analysis
            if input_pages < target_p - 40:
                st.write(f"📖 **Add More Depth:** Bestselling {selected_genre} books average **{int(target_p)} pages**. Yours is quite a bit shorter. Consider expanding the world-building or character arcs.")
            elif input_pages > target_p + 40:
                st.write(f"📖 **Trim the Fat:** Bestselling {selected_genre} books average **{int(target_p)} pages**. At {int(input_pages)} pages, you might be risking pacing issues. Look for places to tighten the narrative.")
            else:
                st.write(f"✅ **Perfect Length:** Your page count is right in the sweet spot for {selected_genre} bestsellers (average is {int(target_p)} pages).")
                
            # Dynamic Price Analysis
            if input_price > target_price + 2.00:
                st.write(f"💰 **Lower the Price:** Your price of \${input_price:.2f} is higher than the \${target_price:.2f} average for bestsellers in this genre. A slight discount might boost your reach.")
            elif input_price < target_price - 2.00:
                st.write(f"💰 **Increase the Price:** You might be under-valuing your work! Bestselling {selected_genre} books average \${target_price:.2f}. You could likely raise it without hurting sales.")
            else:
                st.write(f"✅ **Competitive Pricing:** \${input_price:.2f} is perfectly aligned with the \${target_price:.2f} average for {selected_genre} bestsellers.")