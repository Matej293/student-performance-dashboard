import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Student Performance Dashboard", 
    page_icon=":books:", 
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    try:
        with open('.streamlit/style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css()

@st.cache_data
def get_data_from_csv():
    try:
        if os.path.exists("student_habits_performance.csv"):
            df = pd.read_csv("student_habits_performance.csv")
            return df
        else:
            st.info("Downloading dataset from Kaggle...")
            
            # Import kagglehub only when needed
            import kagglehub
            path = kagglehub.dataset_download("jayaantanaath/student-habits-vs-academic-performance")
            
            csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
            if csv_files:
                csv_file = os.path.join(path, csv_files[0])
                df = pd.read_csv(csv_file)
                
                # Cache the CSV locally for future runs
                df.to_csv("student_habits_performance.csv", index=False)
                st.success(f"Dataset downloaded and cached successfully from: {path}")
                return df
            else:
                st.error("No CSV files found in the downloaded dataset")
                return pd.DataFrame()
                
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def calculate_correlation_matrix(df_filtered, numeric_cols):
    """Cache expensive correlation calculations"""
    return df_filtered[numeric_cols].corr()

@st.cache_data  
def process_sunburst_data(df_filtered):
    """Cache sunburst data processing"""
    df_copy = df_filtered.copy()
    df_copy['performance_cat'] = pd.cut(df_copy['exam_score'], 
                                       bins=[0, 60, 80, 100], 
                                       labels=['Below Average', 'Average', 'Excellent'])
    
    sunburst_data = []
    # Build hierarchical data structure
    for gender in df_copy['gender'].unique():
        for edu in df_copy['parental_education_level'].unique():
            for perf in df_copy['performance_cat'].dropna().unique():
                count = len(df_copy[
                    (df_copy['gender'] == gender) & 
                    (df_copy['parental_education_level'] == edu) & 
                    (df_copy['performance_cat'] == perf)
                ])
                if count > 0:
                    sunburst_data.append({
                        'ids': f"{gender}-{edu}-{perf}",
                        'labels': f"{perf}",
                        'parents': f"{gender}-{edu}",
                        'values': count
                    })

    for gender in df_copy['gender'].unique():
        for edu in df_copy['parental_education_level'].unique():
            count = len(df_copy[
                (df_copy['gender'] == gender) & 
                (df_copy['parental_education_level'] == edu)
            ])
            if count > 0:
                sunburst_data.append({
                    'ids': f"{gender}-{edu}",
                    'labels': f"{edu}",
                    'parents': gender,
                    'values': count
                })

    for gender in df_copy['gender'].unique():
        count = len(df_copy[df_copy['gender'] == gender])
        sunburst_data.append({
            'ids': gender,
            'labels': gender,
            'parents': "",
            'values': count
        })
    
    return pd.DataFrame(sunburst_data)

df = get_data_from_csv()

st.sidebar.header("Filters:")
gender = st.sidebar.multiselect(
    "Gender:",
    options=df["gender"].unique(),
    default=df["gender"].unique()
)

part_time_job = st.sidebar.multiselect(
    "Part time job:",
    options=df["part_time_job"].unique(),
    default=df["part_time_job"].unique(),
)

parental_education = st.sidebar.multiselect(
    "Parental education level:",
    options=[x for x in df["parental_education_level"].unique() if pd.notna(x)],
    default=[x for x in df["parental_education_level"].unique() if pd.notna(x)]
)

extracurricular = st.sidebar.multiselect(
    "Extracurricular participation:",
    options=df["extracurricular_participation"].unique(),
    default=df["extracurricular_participation"].unique()
)

df_selection = df[
    (df["gender"].isin(gender)) & 
    (df["part_time_job"].isin(part_time_job)) & 
    (df["parental_education_level"].isin(parental_education) | df["parental_education_level"].isna()) &
    (df["extracurricular_participation"].isin(extracurricular))
]

if df_selection.empty:
    st.warning("No data available based on the current filter settings!")
    st.stop()

st.title("🎓 Student Performance Dashboard")

col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

with col1:
    total_students = len(df_selection)
    st.metric(
        label="📊 Total Students", 
        value=f"{total_students:,}",
        help="Total number of students in the filtered dataset"
    )

with col2:
    average_exam_score = round(df_selection["exam_score"].mean(), 2)
    st.metric(
        label="📈 Average Exam Score", 
        value=f"{average_exam_score}%",
        delta=f"{round(average_exam_score - df['exam_score'].mean(), 2)}%" if len(df_selection) < len(df) else None,
        help="Average exam score for selected students"
    )

with col3:
    average_mental_health = round(df_selection["mental_health_rating"].mean(), 1)
    star_rating = "⭐" * min(int(round(average_mental_health, 0)), 5)
    st.metric(
        label="🧠 Mental Health Rating", 
        value=f"{average_mental_health} {star_rating}",
        help="Average mental health rating (1-10 scale)"
    )

with col4:
    average_study_hours = round(df_selection["study_hours_per_day"].mean(), 2)
    st.metric(
        label="📚 Avg Study Hours/Day", 
        value=f"{average_study_hours} hrs",
        help="Average daily study hours"
    )

with col5:
    average_attendance = round(df_selection["attendance_percentage"].mean(), 1)
    st.metric(
        label="🏫 Average Attendance", 
        value=f"{average_attendance}%",
        help="Average attendance percentage"
    )

with col6:
    average_sleep = round(df_selection['sleep_hours'].mean(), 1)
    st.metric(
        label="😴 Average Sleep", 
        value=f"{average_sleep} hrs",
        help="Average sleep hours per night"
    )

st.markdown("""---""")

scores_by_education = df_selection.groupby(by=["parental_education_level"])[["exam_score"]].mean().sort_values(by="exam_score")

performance_colors = [
    '#FF4444', '#FF6B44', '#FF8844', '#FFA544', '#FFC244', 
    '#E6D244', '#B8E644', '#8AE644', '#5CE644', '#2EE644'
]

LEGEND_STYLE = {
    'bgcolor': "rgba(63, 36, 117, 0.9)",
    'bordercolor': "#6379cf",
    'borderwidth': 2,
    'font': dict(color="#ebedf9", size=12, family="Inter")
}

fig_education_scores = px.bar(
    scores_by_education,
    x="exam_score",
    y=scores_by_education.index,
    orientation="h",
    title="<b>Average Exam Scores by Parental Education Level</b>",
    template="plotly_white",
)

fig_education_scores.update_traces(
    marker=dict(
        line=dict(width=2, color='rgba(255,255,255,0.4)'),
        opacity=0.8
    ),
    texttemplate='%{x:.1f}%',
    textposition='outside',
    textfont=dict(color="#ebedf9", size=12, family="Inter")
)
fig_education_scores.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#ebedf9"),
    title_font=dict(color="#ebedf9"),
    xaxis=dict(
        showgrid=False, 
        color="#ebedf9",
        title="Average Exam Score (%)"
    ),
    yaxis=dict(
        color="#ebedf9",
        title="Parental Education Level"
    ),
    margin=dict(l=20, r=20, t=50, b=20),
    showlegend=False
)

st.subheader("""Overview""")

fig_study_vs_scores = px.scatter(
    df_selection,
    x="study_hours_per_day",
    y="exam_score",
    size="attendance_percentage",
    color="gender",
    hover_data=["mental_health_rating", "sleep_hours"],
    title="<b>Study Hours vs Exam Scores</b>",
    template="plotly_white",
)
fig_study_vs_scores.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#ebedf9"),
    title_font=dict(color="#ebedf9"),
    xaxis=dict(showgrid=False, color="#ebedf9"),
    yaxis=dict(showgrid=False, color="#ebedf9"),
    margin=dict(l=20, r=100, t=50, b=20),
    legend=dict(
        **LEGEND_STYLE,
        orientation="v",
        x=1.08,
        y=1,
        xanchor="right",
        yanchor="top",
        title=dict(
            text="Gender",
            font=dict(color="#ebedf9", size=12, family="Inter")
        )
    )
)

left_column, right_column = st.columns(2)
left_column.plotly_chart(fig_study_vs_scores, use_container_width=True)
right_column.plotly_chart(fig_education_scores, use_container_width=True)

st.markdown("""---""")

fig_mental_health = px.histogram(
    df_selection,
    x="mental_health_rating",
    title="<b>Mental Health Rating Distribution</b>",
    template="plotly_white",
    nbins=10,
    color_discrete_sequence=['#2E86AB']
)

fig_mental_health.update_traces(
    marker=dict(
        color=[performance_colors[int(rating)-1] if rating <= 10 else performance_colors[-1] 
               for rating in sorted(df_selection['mental_health_rating'].unique())],
        line=dict(width=2, color='rgba(255,255,255,0.4)'),
        opacity=0.8
    ),
    texttemplate='%{y}',
    textposition='outside',
    textfont=dict(color="#ebedf9", size=12)
)
fig_mental_health.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#ebedf9"),
    title_font=dict(color="#ebedf9"),
    xaxis=dict(
        showgrid=False, 
        color="#ebedf9",
        title="Mental Health Rating (1=Poor, 10=Excellent)",
        dtick=1
    ),
    yaxis=dict(
        showgrid=False, 
        color="#ebedf9",
        title="Number of Students"
    ),
    margin=dict(l=20, r=20, t=50, b=20),
    showlegend=False,
    bargap=0.2
)

fig_diet_scores = px.box(
    df_selection,
    x="diet_quality",
    y="exam_score",
    title="<b>Exam Scores by Diet Quality</b>",
    color="diet_quality",
    template="plotly_white"
)
fig_diet_scores.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#ebedf9"),
    title_font=dict(color="#ebedf9"),
    xaxis=dict(showgrid=False, color="#ebedf9"),
    yaxis=dict(showgrid=False, color="#ebedf9"),
    margin=dict(l=20, r=120, t=50, b=20),
    legend=dict(
        **LEGEND_STYLE,
        orientation="v",
        x=1,
        y=1,
        xanchor="left",
        yanchor="top",
        title=dict(
            text="Diet Quality",
            font=dict(color="#ebedf9", size=12, family="Inter")
        )
    )
)

left_column2, right_column2 = st.columns(2)
left_column2.plotly_chart(fig_mental_health, use_container_width=True)
right_column2.plotly_chart(fig_diet_scores, use_container_width=True)

st.markdown("""---""")
st.subheader("Correlation Analysis")

numeric_columns = ["study_hours_per_day", "social_media_hours", "netflix_hours", 
                  "attendance_percentage", "sleep_hours", "exercise_frequency", 
                  "mental_health_rating", "exam_score"]

correlation_data = calculate_correlation_matrix(df_selection, numeric_columns)

fig_correlation = px.imshow(
    correlation_data,
    title="<b>Correlation Matrix of Key Variables</b>",
    color_continuous_scale=[
        [0.0, "#ebeef9"], [0.2, "#b1bce7"], [0.35, "#8a9adb"], [0.45, "#6379cf"], [0.5, "#3c57c3"],
        [0.55, "#30469c"], [0.65, "#243475"], [0.8, "#18234e"], [0.9, "#0c1127"], [1.0, "#18234e"]
    ],
    aspect="auto",
    template="plotly_white",
    zmin=-1,
    zmax=1
)

fig_correlation.update_traces(
    text=correlation_data.round(2).values,
    texttemplate="%{text}",
    textfont=dict(
        size=12,
        color="#ebedf9",
        family="Inter"
    ),
    hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
)
fig_correlation.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#ebedf9"),
    title_font=dict(color="#ebedf9"),
    margin=dict(l=20, r=120, t=80, b=20),
    showlegend=False,
    width=650,
    xaxis=dict(
        side="bottom",
        tickangle=-45,
        color="#ebedf9",
        domain=[0, 0.9]
    ),
    yaxis=dict(
        color="#ebedf9",
        domain=[0, 1]
    ),
    coloraxis_colorbar=dict(
        title=dict(
            text="Correlation Coefficient",
            font=dict(color="#ebedf9", size=12)
        ),
        tickfont=dict(color="#ebedf9", size=12),
        tickvals=[-1, -0.5, 0, 0.5, 1],
        ticktext=["-1.0 (Perfect Negative)", "-0.5 (Moderate Negative)", 
                 "0.0 (No Correlation)", "0.5 (Moderate Positive)", 
                 "1.0 (Perfect Positive)"],
        thickness=10,
        len=0.95,
        x=0.92,
        xanchor="left",
        y=0.5,
        yanchor="middle",
        bgcolor="rgba(63, 36, 117, 0.9)",
        bordercolor="#6379cf",
        borderwidth=2,
        ticks="outside",
    ),
)

st.plotly_chart(fig_correlation, use_container_width=True)

st.markdown("""---""")
st.subheader("Parallel Coordinates Analysis")
st.markdown("""
**How to read this chart:**
- Each vertical line represents a different variable
- Each colored line represents one student's profile across all variables
- **Color indicates exam score** (red = low, yellow = medium, blue = high)
""")

@st.cache_data
def prepare_parallel_data(df_filtered):
    """Prepare and cache parallel coordinates data"""
    df_parallel = df_filtered.copy()
    
    # Optimize groupings
    df_parallel['study_hours_grouped'] = df_parallel['study_hours_per_day'].round(0).astype(int)
    df_parallel['social_media_grouped'] = pd.cut(df_parallel['social_media_hours'], 
                                               bins=[0, 1, 2, 3, 4, 10], 
                                               labels=[1, 2, 3, 4, 5], 
                                               include_lowest=True).astype(int)
    df_parallel['sleep_hours_grouped'] = df_parallel['sleep_hours'].round(0).astype(int)
    
    return df_parallel

df_parallel = prepare_parallel_data(df_selection)

parallel_vars = ["study_hours_grouped", "social_media_grouped", "sleep_hours_grouped", 
                "attendance_percentage", "mental_health_rating", "exercise_frequency", "exam_score"]

fig_parallel = go.Figure(data=
    go.Parcoords(
        line=dict(color=df_parallel['exam_score'],
                 colorscale='RdYlBu_r',
                 showscale=True,
                 colorbar=dict(
                     title=dict(text="Exam Score (%)", font=dict(color="#ebedf9", size=12)),
                     tickfont=dict(color="#ebedf9", size=12),
                     bgcolor="rgba(63, 36, 117, 0.9)",
                     bordercolor="#6379cf",
                     borderwidth=2,
                     orientation="h",
                     x=0.5,
                     xanchor="center",
                     y=-0.12,
                     yanchor="top",
                     thickness=12,
                     len=0.6
                 )),
        dimensions=[
            dict(range=[df_parallel[var].min(), df_parallel[var].max()],
                 constraintrange=[df_parallel[var].min(), df_parallel[var].max()],
                 label=var.replace('_', ' ').replace('grouped', '').title(), 
                 values=df_parallel[var]) for var in parallel_vars
        ]
    )
)

fig_parallel.update_layout(
    title="<b>Parallel Coordinates Plot - Student Performance Patterns</b>",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#ebedf9"),
    title_font=dict(color="#ebedf9"),
    margin=dict(l=50, r=50, t=80, b=80)
)

st.plotly_chart(fig_parallel, use_container_width=True)

st.markdown("""---""")
st.subheader("Student Profile Comparison")
st.markdown("""
**What this shows:**
- **Radar charts compare average profiles** of low, medium, and high performers
- Each axis represents a different habit/factor
- **Larger areas** indicate stronger performance in those dimensions

*Note: All values are normalized to 0-1 scale for fair comparison*
""")

# Cache performance tier calculation for reuse
@st.cache_data
def calculate_performance_tiers(df_filtered):
    """Calculate performance tiers with caching"""
    return pd.cut(df_filtered['exam_score'], 
                  bins=[0, 50, 75, 100], 
                  labels=['Low (0-50)', 'Medium (50-75)', 'High (75-100)'])

# Calculate once and reuse
df_selection['performance_tier'] = calculate_performance_tiers(df_selection)

profile_vars = ["study_hours_per_day", "sleep_hours", "exercise_frequency", 
               "mental_health_rating", "attendance_percentage"]

col1, col2, col3 = st.columns(3)

def normalize_for_radar(df, columns):
    normalized = df.copy()
    for col in columns:
        max_val = df[col].max()
        min_val = df[col].min()
        normalized[col] = (df[col] - min_val) / (max_val - min_val) if max_val != min_val else 0.5
    return normalized

normalized_df = normalize_for_radar(df_selection, profile_vars)

def get_tier_colors():
    """Get simplified color scheme for performance tiers"""
    return {
        'Low (0-50)': ('#ff6b6b', 'rgba(255, 107, 107, 0.6)'),
        'Medium (50-75)': ('#4ecdc4', 'rgba(76, 205, 196, 0.6)'),
        'High (75-100)': ('#45b7d1', 'rgba(69, 183, 209, 0.6)')
    }

tier_colors = get_tier_colors()

for tier, col in zip(['Low (0-50)', 'Medium (50-75)', 'High (75-100)'], [col1, col2, col3]):
    if tier in df_selection['performance_tier'].values:
        tier_data = normalized_df[df_selection['performance_tier'] == tier][profile_vars].mean()
        
        fig_radar = go.Figure()
        
        line_color, fill_color = tier_colors[tier]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=list(tier_data.values) + [tier_data.values[0]],
            theta=list(tier_data.index) + [tier_data.index[0]],
            fill='toself',
            name=f'{tier} Performers',
            line_color=line_color,
            fillcolor=fill_color
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    color="#ebedf9",
                    tickfont=dict(size=12),
                    gridcolor="rgba(180, 190, 210, 0.3)",
                    linecolor="rgba(180, 190, 210, 0.5)"
                ),
                angularaxis=dict(
                    color="#ebedf9",
                    tickfont=dict(size=11),
                    gridcolor="rgba(180, 190, 210, 0.3)",
                    linecolor="rgba(180, 190, 210, 0.5)"
                ),
                bgcolor="rgba(0,0,0,0)"
            ),
            showlegend=False,
            title=f"<b>{tier} Performers</b>",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ebedf9"),
            title_font=dict(color="#ebedf9"),
            margin=dict(l=10, r=10, t=50, b=20),
            height=400
        )
        
        col.plotly_chart(fig_radar, use_container_width=True)

st.markdown("""---""")
st.subheader("Hierarchical Student Distribution")
st.markdown("""
**How to read this sunburst:**
- **Center**: Gender distribution
- **Middle ring**: Parental education level within each gender
- **Outer ring**: Performance level within each education-gender combination
- **Size** indicates number of students in each category
""")

df_sunburst = df_selection.copy()
df_sunburst['performance_cat'] = pd.cut(df_sunburst['exam_score'], 
                                      bins=[0, 60, 80, 100], 
                                      labels=['Below Average', 'Average', 'Excellent'])

sunburst_df = process_sunburst_data(df_selection)

fig_sunburst = go.Figure(go.Sunburst(
    ids=sunburst_df['ids'],
    labels=sunburst_df['labels'],
    parents=sunburst_df['parents'],
    values=sunburst_df['values'],
    branchvalues="total",
    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent}<extra></extra>',
    maxdepth=3,
))

fig_sunburst.update_layout(
    title="<b>Student Distribution: Gender → Education → Performance</b>",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#ebedf9"),
    title_font=dict(color="#ebedf9"),
    margin=dict(l=20, r=20, t=50, b=20)
)

st.plotly_chart(fig_sunburst, use_container_width=True)

st.markdown("""---""")
st.subheader("Distribution Analysis")
st.markdown("""
**Violin plots show the following:**
- **Shape shows distribution**: Wide = many students, narrow = few students at that level
- **Box inside**: Shows median, quartiles, and outliers
- **Multiple bumps**: Indicate subgroups within performance tiers

*Each violin represents the full distribution of values for that performance group*
""")

col1, col2 = st.columns(2)

with col1:
    fig_violin1 = go.Figure()
    
    for tier in df_selection['performance_tier'].dropna().unique():
        tier_data = df_selection[df_selection['performance_tier'] == tier]['study_hours_per_day']
        line_color, fill_color = tier_colors.get(tier, ('#6379cf', 'rgba(99, 121, 207, 0.6)'))
        
        fig_violin1.add_trace(go.Violin(
            y=tier_data,
            name=tier,
            box_visible=True,
            meanline_visible=True,
            fillcolor=fill_color,
            line_color=line_color
        ))
    
    fig_violin1.update_layout(
        title="<b>Study Hours Distribution by Performance</b>",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ebedf9"),
        title_font=dict(color="#ebedf9"),
        xaxis=dict(color="#ebedf9"),
        yaxis=dict(color="#ebedf9", title="Study Hours per Day"),
        margin=dict(l=20, r=20, t=50, b=60),
        showlegend=True,
        legend=dict(
            **LEGEND_STYLE,
            orientation="h",
            x=0.5,
            y=-0.1,
            xanchor="center",
            yanchor="top",
            title=dict(
                text="Performance Tier",
                font=dict(color="#ebedf9", size=12, family="Inter")
            )
        )
    )

with col2:
    fig_violin2 = go.Figure()
    
    for tier in df_selection['performance_tier'].dropna().unique():
        tier_data = df_selection[df_selection['performance_tier'] == tier]['mental_health_rating']
        line_color, fill_color = tier_colors.get(tier, ('#6379cf', 'rgba(99, 121, 207, 0.6)'))
        
        fig_violin2.add_trace(go.Violin(
            y=tier_data,
            name=tier,
            box_visible=True,
            meanline_visible=True,
            fillcolor=fill_color,
            line_color=line_color
        ))
    
    fig_violin2.update_layout(
        title="<b>Mental Health Distribution by Performance</b>",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ebedf9"),
        title_font=dict(color="#ebedf9"),
        xaxis=dict(color="#ebedf9"),
        yaxis=dict(color="#ebedf9", title="Mental Health Rating"),
        margin=dict(l=20, r=20, t=50, b=60),
        showlegend=True,
        legend=dict(
            **LEGEND_STYLE,
            orientation="h",
            x=0.5,
            y=-0.1,
            xanchor="center",
            yanchor="top",
            title=dict(
                text="Performance Tier",
                font=dict(color="#ebedf9", size=12, family="Inter")
            )
        )
    )

col1.plotly_chart(fig_violin1, use_container_width=True)
col2.plotly_chart(fig_violin2, use_container_width=True)

st.markdown("""---""")
st.subheader("3D Analysis of Key Habits")

fig_3d = px.scatter_3d(
    df_selection,
    x="study_hours_per_day",
    y="sleep_hours",
    z="social_media_hours",
    color="exam_score",
    size="attendance_percentage",
    hover_data=["mental_health_rating", "exercise_frequency"],
    title="<b>3D Analysis: Study Hours vs Sleep vs Social Media</b>",
    color_continuous_scale="RdYlBu_r",
    template="plotly_white"
)

fig_3d.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#ebedf9"),
    title_font=dict(color="#ebedf9"),
    margin=dict(l=20, r=80, t=50, b=20),
    coloraxis_colorbar=dict(
        title=dict(text="Exam Score", font=dict(color="#ebedf9", size=12)),
        tickfont=dict(color="#ebedf9", size=12),
        bgcolor="rgba(63, 36, 117, 0.9)",
        bordercolor="#6379cf",
        borderwidth=2,
        x=0.95,
        xanchor="left",
        y=0.85,
        yanchor="top",
        thickness=10,
        len=0.5
    ),
    scene=dict(
        xaxis=dict(
            color="#ebedf9", 
            title="Study Hours per Day",
            backgroundcolor="rgba(0,0,0,0)",
            gridcolor="rgba(180, 190, 210, 0.2)",
            showbackground=True,
        ),
        yaxis=dict(
            color="#ebedf9", 
            title="Sleep Hours",
            backgroundcolor="rgba(0,0,0,0)",
            gridcolor="rgba(180, 190, 210, 0.2)",
            showbackground=True,
        ),
        zaxis=dict(
            color="#ebedf9", 
            title="Social Media Hours",
            backgroundcolor="rgba(0,0,0,0)",
            gridcolor="rgba(180, 190, 210, 0.2)",
            showbackground=True,
        ),
        bgcolor="rgba(0,0,0,0)"
    )
)

st.plotly_chart(fig_3d, use_container_width=True)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
