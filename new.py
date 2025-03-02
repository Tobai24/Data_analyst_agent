import streamlit as st
from langchain.schema import Document
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import os
from langgraph.prebuilt import ToolNode
import time
import uuid
from typing import Callable
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sqlalchemy.sql import text
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import List, TypedDict, Dict, Any, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage,  SystemMessage
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, Integer, Float, String, MetaData, Table
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()


class Settings(BaseSettings):
    database_hostname: str
    database_port: str
    database_password: str
    database_name: str
    database_username: str

    openai_api_key: str
    langsmith_tracing: bool
    langsmith_endpoint: str
    langsmith_api_key: str
    langsmith_project: str
    tavily_api_key: str
    upload_folder: str
    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()

llm = ChatOpenAI(model="gpt-4o-mini") 

def infer_column_types(df):
    """Infer SQL column types from DataFrame"""
    type_map = {}
    
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            type_map[col] = Integer
        elif pd.api.types.is_float_dtype(df[col]):
            type_map[col] = Float
        else:
            type_map[col] = String(255)
    
    return type_map

def create_database_and_table(df, database_name=None, table_name=None):
    """Create a new database and table based on DataFrame schema"""
    
    if database_name is None:
        database_name = settings.database_name
        if database_name is None:  
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            database_name = f"analysis_db_{timestamp}"
    
    if table_name is None:
        table_name = "data_table"
    
    settings.database_name = database_name

    admin_engine = create_engine(
        f"postgresql+psycopg2://{settings.database_username}:{settings.database_password}@"
        f"{settings.database_hostname}:{settings.database_port}/postgres"
    )
    
    with admin_engine.connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")
        conn.execute(text(f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{database_name}'"))
        conn.execute(text(f"DROP DATABASE IF EXISTS {database_name}"))
        conn.execute(text(f"CREATE DATABASE {database_name}"))

    db_engine = create_engine(
        f"postgresql+psycopg2://{settings.database_username}:{settings.database_password}@"
        f"{settings.database_hostname}:{settings.database_port}/{database_name}"
    )
    
    metadata = MetaData()
    columns = [Column('id', Integer, primary_key=True)]
    schema_info = {'id': 'Integer (primary key)'}

    # Infer column types
    column_types = infer_column_types(df)
    for col_name, sql_type in column_types.items():
        safe_col_name = col_name.replace(' ', '_').lower()
        columns.append(Column(safe_col_name, sql_type))
        schema_info[safe_col_name] = sql_type.__name__ if isinstance(sql_type, type) else str(sql_type)

    table = Table(table_name, metadata, *columns)
    metadata.create_all(db_engine)

    df_to_insert = df.copy()
    df_to_insert.columns = [col.replace(' ', '_').lower() for col in df.columns]

    with db_engine.connect() as conn:
        batch_size = 1000
        for i in range(0, len(df_to_insert), batch_size):
            batch = df_to_insert.iloc[i:i+batch_size]
            batch.to_sql(table_name, conn, if_exists='append', index=False)

    return {
        'database_name': database_name,
        'table_name': table_name,
        'schema': schema_info,
        'engine': db_engine
    }
    
    
def get_db_connection(database_name=None):
    """Get a database connection to the specified database"""
    db_name = database_name or settings.database_name
    
    if not db_name:
        raise ValueError("No database name specified")
    
    engine = create_engine(
        f"postgresql+psycopg2://{settings.database_username}:{settings.database_password}@"
        f"{settings.database_hostname}:{settings.database_port}/{db_name}"
    )
    
    SessionLocal = sessionmaker(autoflush=False, bind=engine, autocommit=False)
    db = SessionLocal()
    
    return db, engine

class Analyst(BaseModel):
    name: str = Field(description="Name of the analyst.")
    affiliation: str = Field(description="Primary affiliation of the analyst.")
    role: str = Field(description="Role of the analyst in the context of data analysis.")
    data_analysis_focus: str = Field(description="Primary focus area in data analysis (e.g., Machine Learning, Statistical Analysis, Data Engineering).")
    description: str = Field(description="Description of the analyst's expertise, concerns, and analysis approach.")
    analysis_results: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Results of the analyst's work.")
    database_interactions: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Database queries and transformations performed by the analyst.")

    @property
    def persona(self) -> str:
        return (
            f"Name: {self.name}\n"
            f"Role: {self.role}\n"
            f"Affiliation: {self.affiliation}\n"
            f"Focus: {self.data_analysis_focus}\n"
            f"Description: {self.description}\n"
        )


class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(description="List of analysts contributing to the data analysis.")
    analysis_summary: str = Field(description="Summary of findings from all analysts.")

    class Config:
        # Explicitly define required fields
        schema_extra = {
            "required": ["analysts", "analysis_summary"]
        }
        
        
class GenerateAnalystsState(TypedDict):
    csv_path: str 
    analysis_goal: str  
    max_analysts: int
    user_query: Optional[str]
    human_analyst_feedback: Optional[str]
    database_name: Optional[str] 
    table_name: Optional[str]  
    analysts: Optional[List[Analyst]]
    data_df: Dict[str, Any] 
    ml_models: Dict[str, Any]
    visualizations: Dict[str, str] 
    database_schema: Optional[Dict[str, str]] 
    analysis_summary: Optional[str]
    
analyst_instructions = """You are responsible for creating a team of AI analyst personas that will collaborate to achieve specific data analysis goals. 
Each analyst should specialize in a distinct role and contribute effectively to the overall analysis process. Follow these instructions carefully:

---

### **Step 1: Understand the Analysis Goal **
- Review the **primary analysis goal** that the AI analysts need to accomplish:  
  **Goal:** {analyst_goal}  
- Determine the scope of the analysis, including key challenges and expected outcomes.
- Pick the top {max_analysts} analyst.

### **Step 2: Incorporate Human Feedback (If Provided)**
- If any **editorial feedback** or **domain-specific instructions** are provided, integrate them into the analysis plan:  
  **Feedback:** {human_analyst_feedback}  
- Adjust the analysts' approaches accordingly to align with expert guidance.

### **Step 3: Identify Core Analytical Themes**
- Break down the **analysis goal** into **key sub-goals** that require specialized expertise.  
- Determine the **most relevant aspects** of the dataset that must be examined.  
- Consider important data-related tasks such as:
  - Data cleaning and preprocessing  
  - Exploratory Data Analysis (EDA)  
  - Statistical insights and hypothesis testing  
  - Predictive modeling or machine learning  
  - Data visualization and reporting  

### **Step 4: Define a Team of Specialized AI Analysts**
Assign one **AI analyst persona** to each identified **sub-goal**.  
Each analyst should have **distinct expertise** while contributing to the overall objective.  
Ensure diversity in roles by selecting from the following:

#### ** 1️ Database Engineer**
   - Responsible for database creation, management, and query optimization.  
   - Ensures efficient data retrieval, indexing, and schema design.  
   - Works with SQL/PostgreSQL, NoSQL, and data warehouses.  

#### ** 2️ Data Engineer**
   - Focuses on data extraction, transformation, and loading (ETL).  
   - Cleans and preprocesses raw data for analysis.  
   - Handles large-scale data pipelines and ensures data integrity.  

#### **3️ Statistical Analyst**
   - Performs rigorous statistical testing and hypothesis validation.  
   - Uses techniques like regression analysis, ANOVA, and probability modeling.  
   - Ensures statistical soundness in insights and interpretations.  

#### **4️ Machine Learning Specialist**
   - Develops predictive models and applies machine learning techniques.  
   - Tunes hyperparameters and validates model performance.  
   - Works with libraries such as Scikit-learn, TensorFlow, or PyTorch.  

#### **5️ Data Visualization Expert**
   - Translates complex data into intuitive and interactive visualizations.  
   - Uses tools like Matplotlib, Seaborn, Power BI, and Tableau.  
   - Ensures insights are clear and accessible to non-technical stakeholders.  

#### **6️ Domain Expert (Optional)**
   - Provides industry-specific knowledge to contextualize findings.  
   - Ensures the analysis aligns with real-world applications in healthcare, finance, etc.  

### **Step 5: Define Each Analyst’s Approach**
For each assigned **AI analyst persona**, provide a **detailed description** of:  
  - Their **area of expertise**  
  - Their **approach to solving the assigned sub-goal**  
  - The **methods and tools** they will use  
  - The **expected outputs** from their analysis  

### **Step 6: Execute the Analysis as a Collaborative Multi-Agent System**
- Ensure that each AI analyst persona works **cohesively** towards the final analysis goal.  
- The outputs from one analyst may serve as inputs for another (e.g., the Data Engineer preps data for the Machine Learning Specialist).  
- Optimize the workflow for efficiency, accuracy, and interpretability.  

---

**Outcome:** A structured and effective multi-agent AI analysis team, with each persona assigned a well-defined role, working together to accomplish the specified goal.  
"""

def process_csv_and_create_db(state: GenerateAnalystsState):
    """Load CSV and create database"""
    csv_path = state['csv_path']
    
    try:
        df = pd.read_csv(csv_path)
        
        db_info = create_database_and_table(
            df, 
            database_name=state.get('database_name'),
            table_name=state.get('table_name')
        )
        
        state["data_df"] = {"raw": df}
        
        return {
            'data_df': df.to_dict(orient="records"),
            "database_name": db_info['database_name'],
            "table_name": db_info['table_name'],
            "database_schema": db_info['schema']
        }
    except Exception as e:
        return {
            "data_df": {"error": str(e)},
            "error_message": f"Failed to process CSV and create database: {str(e)}"
        }
        
def process_csv_and_create_db(state: GenerateAnalystsState):
    """Load CSV and create database"""
    csv_path = state['csv_path']
    
    try:
        df = pd.read_csv(csv_path)
        
        db_info = create_database_and_table(
            df, 
            database_name=state.get('database_name'),
            table_name=state.get('table_name')
        )
        
        # Store DataFrame as a serializable dict
        state["data_df"] = {"raw": df.to_dict(orient="records")}
        
        return {
            "data_df": state["data_df"],
            "database_name": db_info['database_name'],
            "table_name": db_info['table_name'],
            "database_schema": db_info['schema']
        }
    except Exception as e:
        return {
            "data_df": {"error": str(e)},
            "error_message": f"Failed to process CSV and create database: {str(e)}"
        }
    
import json
def create_analysts(state: GenerateAnalystsState):
    """Create analysts based on the dataset and analysis goals"""
    if "data_df" not in state or "raw" not in state["data_df"]:
        return {"error_message": "No raw data available to create analysts"}
    
    db_info = f"Database: {state['database_name']}, Table: {state['table_name']}"
    schema_info = "Schema: " + ", ".join([f"{k}: {v}" for k, v in state.get("database_schema", {}).items()])
    data_sample = "Sample data: " + str(state["data_df"]["raw"][:3])  # First 3 records
    
    goal = f"{db_info}\n{schema_info}\n{data_sample}\n\nAnalysis Goal: {state['analysis_goal']}"
    max_analysts = int(state.get("max_analysts", 4))
    human_analyst_feedback = state.get("human_analyst_feedback", "")

    structured_llm = llm.with_structured_output(Perspectives)
    
    system_message = analyst_instructions.format(
        analyst_goal=goal,
        human_analyst_feedback=human_analyst_feedback, 
        max_analysts=max_analysts
    )

    perspectives = structured_llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content="Generate the set of analysts that will create and interact with the database, analyze the data, clean it, visualize it, perform data engineering tasks, and build machine learning models.")
    ])
    
    return {"analysts": perspectives.analysts, "analysis_summary": perspectives.analysis_summary}

def perform_database_interactions(state: GenerateAnalystsState):
    """Have analysts interact with the database to perform various queries and transformations"""
    database_name = state['database_name']
    table_name = state['table_name']
    analysts = state['analysts']
    
    db, engine = get_db_connection(database_name)
    
    try:
        all_interactions = []
        
        # Find database-focused analysts
        db_analysts = [a for a in analysts if "Database" in a.data_analysis_focus or "Data Engineer" in a.role]
        
        for analyst in db_analysts:
            interactions = []
            
            stats_query = f"""
            SELECT 
                COUNT(*) as row_count,
                (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = '{table_name}') as column_count
            FROM {table_name}
            """
            stats_result = pd.read_sql(stats_query, engine)
            row_count = stats_result.iloc[0]['row_count']
            column_count = stats_result.iloc[0]['column_count']
            
            interactions.append({
                "query": stats_query,
                "description": "Basic table statistics",
                "result_summary": f"Table has {row_count} rows and {column_count} columns"
            })
            
            column_query = f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            AND data_type IN ('integer', 'numeric', 'real', 'double precision')
            """
            numeric_cols = pd.read_sql(column_query, engine)
            
            if not numeric_cols.empty:
                col_names = numeric_cols['column_name'].tolist()
                agg_cols = ", ".join([f"AVG({col}) as avg_{col}, MIN({col}) as min_{col}, MAX({col}) as max_{col}" 
                                     for col in col_names])
                
                agg_query = f"""
                SELECT {agg_cols}
                FROM {table_name}
                """
                
                agg_results = pd.read_sql(agg_query, engine)
                interactions.append({
                    "query": agg_query,
                    "description": "Aggregate statistics on numeric columns",
                    "result_summary": "Calculated AVG, MIN, MAX for all numeric columns"
                })
                
                if col_names: 
                    view_name = f"agg_{table_name}_view"
                    create_view_query = f"""
                    CREATE OR REPLACE VIEW {view_name} AS
                    SELECT {', '.join(col_names)}
                    FROM {table_name}
                    """

                    with engine.connect() as conn:
                        conn.execute(text(create_view_query))
                        conn.commit()

                    interactions.append({
                        "query": create_view_query,
                        "description": "Created aggregated view for ML analysis",
                        "result_summary": f"View {view_name} created with columns: {', '.join(col_names)}"
                    })
            
            analyst.database_interactions = interactions
            all_interactions.extend(interactions)
        
        return {
            "database_interactions": all_interactions
        }
        
    except Exception as e:
        return {
            "database_interactions": [],
            "error_message": f"Failed to perform database interactions: {str(e)}"
        }
    finally:
        db.close()
        
def data_cleaning_and_enrichment(state: GenerateAnalystsState):
    """Clean and enrich the data, updating the database with cleaned data"""
    if "data_df" not in state or not isinstance(state["data_df"], dict):
        return {
            "error_message": "Invalid data format: 'data_df' should be a dictionary."
        }
    
    # Reconstruct DataFrame from serialized format
    if "raw" in state["data_df"] and isinstance(state["data_df"]["raw"], list):
        raw_df = pd.DataFrame(state["data_df"]["raw"])
    else:
        return {
            "error_message": "Invalid data format: 'raw' key is missing or not a list."
        }
    
    # Clean the data
    clean_df = raw_df.copy()
    
    for col in clean_df.columns:
        if pd.api.types.is_numeric_dtype(clean_df[col]):
            clean_df[col] = clean_df[col].fillna(clean_df[col].mean())
        else:
            clean_df[col] = clean_df[col].fillna(clean_df[col].mode()[0] if not clean_df[col].mode().empty else "Unknown")
    
    clean_df = clean_df.drop_duplicates()
    
    # Enrich the data
    numeric_cols = clean_df.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numeric_cols) >= 2:
        col1, col2 = numeric_cols[0], numeric_cols[1]
        clean_df[f'sum_{col1}_{col2}'] = clean_df[col1] + clean_df[col2]
        
        if (clean_df[col2] != 0).all():
            clean_df[f'ratio_{col1}_{col2}'] = clean_df[col1] / clean_df[col2]
    
    # Save cleaned data to the database
    db, engine = get_db_connection(state['database_name'])
    try:
        cleaned_table_name = f"{state['table_name']}_cleaned"
        clean_df.to_sql(cleaned_table_name, engine, if_exists='replace', index=False)
        
        # Update state with serialized cleaned data
        state["data_df"]["cleaned"] = clean_df.to_dict(orient="records")
        state["cleaned_table_name"] = cleaned_table_name
        
        return {
            "data_df": state["data_df"],
            "cleaned_table_name": cleaned_table_name
        }
    except Exception as e:
        return {
            "data_df": state["data_df"],
            "error_message": f"Failed to save cleaned data to database: {str(e)}"
        }
    finally:
        db.close()
        
def data_visualization(state: GenerateAnalystsState):
    """Create enhanced, interactive visualizations based on user query"""
    if "data_df" not in state or "cleaned" not in state["data_df"]:
        return {"error_message": "No cleaned data available for visualization"}
    
    clean_df = pd.DataFrame(state["data_df"]["cleaned"])
    visualizations = {}
    
    viz_dir = os.path.join(settings.upload_folder, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    viz_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    numeric_cols = clean_df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = clean_df.select_dtypes(include=['object']).columns
    
    # Dynamic visualization based on user query
    user_query = state.get("user_query", "").lower()
    
    # 1. Interactive Scatter Matrix
    if len(numeric_cols) >= 2:
        fig = px.scatter_matrix(
            clean_df,
            dimensions=numeric_cols[:4],
            title="Scatter Matrix of Numeric Features",
            height=800
        )
        scatter_file = f"{viz_dir}/{viz_prefix}_scatter_matrix.html"
        fig.write_html(scatter_file)
        visualizations['scatter_matrix'] = scatter_file

    # 2. Distribution Plot with Statistics
    if numeric_cols and ("distribution" in user_query or not user_query):
        for col in numeric_cols[:3]:
            fig = px.histogram(
                clean_df,
                x=col,
                marginal="box",
                title=f"Distribution of {col}",
                hover_data=clean_df.columns,
                height=600
            )
            dist_file = f"{viz_dir}/{viz_prefix}_dist_{col}.html"
            fig.write_html(dist_file)
            visualizations[f'dist_{col}'] = dist_file

    # 3. Correlation Heatmap
    if len(numeric_cols) > 1 and ("correlation" in user_query or not user_query):
        corr = clean_df[numeric_cols].corr()
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Correlation Heatmap",
            height=600
        )
        corr_file = f"{viz_dir}/{viz_prefix}_correlation.html"
        fig.write_html(corr_file)
        visualizations['correlation'] = corr_file

    # 4. Categorical Analysis
    if categorical_cols and ("category" in user_query or not user_query):
        for col in categorical_cols[:2]:
            fig = px.bar(
                clean_df[col].value_counts().reset_index(),
                x=col,
                y="count",
                title=f"Distribution of {col}",
                height=600
            )
            bar_file = f"{viz_dir}/{viz_prefix}_bar_{col}.html"
            fig.write_html(bar_file)
            visualizations[f'bar_{col}'] = bar_file

    return {"visualizations": visualizations}

def build_ml_models(state: GenerateAnalystsState):
    """Build machine learning models based on the cleaned data"""
    # Check if data_df is a dictionary and contains the "cleaned" key
    if not isinstance(state["data_df"], dict) or "cleaned" not in state["data_df"]:
        return {
            "error_message": "Invalid data format: 'data_df' must be a dictionary with a 'cleaned' key."
        }
    
    clean_df = state["data_df"]["cleaned"]
    
    # Ensure clean_df is a DataFrame
    if not isinstance(clean_df, pd.DataFrame):
        return {
            "error_message": "Invalid data format: 'cleaned' key must contain a pandas DataFrame."
        }
    
    ml_models = {}
    
    # Identify numeric features
    numeric_features = clean_df.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numeric_features) > 1:
        X_cols = numeric_features[:-1]
        y_col = numeric_features[-1]
        
        X = clean_df[X_cols]
        y = clean_df[y_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Basic linear regression model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        ml_models['linear_regression'] = {
            'model': model,
            'train_score': train_score,
            'test_score': test_score,
            'features': list(X.columns),
            'target': y_col,
            'coefficients': {feature: coef for feature, coef in zip(X.columns, model.coef_)}
        }
        
        # Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        
        rf_train_score = rf_model.score(X_train_scaled, y_train)
        rf_test_score = rf_model.score(X_test_scaled, y_test)
        
        ml_models['random_forest'] = {
            'model': rf_model,
            'train_score': rf_train_score,
            'test_score': rf_test_score,
            'features': list(X.columns),
            'target': y_col,
            'feature_importance': {feature: importance for feature, importance in 
                                 zip(X.columns, rf_model.feature_importances_)}
        }
        
        # Save model results to the database
        db, engine = get_db_connection(state['database_name'])
        try:
            model_results = pd.DataFrame({
                'model_name': ['linear_regression', 'random_forest'],
                'train_score': [train_score, rf_train_score],
                'test_score': [test_score, rf_test_score],
                'target_variable': [y_col, y_col],
                'timestamp': [datetime.now(), datetime.now()]
            })
            
            model_results.to_sql('model_results', engine, if_exists='replace', index=False)
            
        except Exception as e:
            ml_models['error'] = str(e)
        finally:
            db.close()
    
    return {"ml_models": ml_models}


def generate_analysis_report(state: GenerateAnalystsState):
    """Generate a detailed, formatted analysis report"""
    analysts = state.get('analysts', [])
    data_df = state.get('data_df', {})
    visualizations = state.get('visualizations', {})
    ml_models = state.get('ml_models', {})
    database_name = state.get('database_name', 'unknown')
    user_query = state.get('user_query', 'General analysis')

    # Process data
    cleaned_df = pd.DataFrame(data_df.get('cleaned', [])) if 'cleaned' in data_df else None
    
    report = f"""
    # Comprehensive Data Analysis Report
    *Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
    *Analysis Focus: {user_query}*

    ## Dataset Overview
    - **Database**: {database_name}
    - **Rows**: {len(cleaned_df) if cleaned_df is not None else 'N/A'}
    - **Columns**: {', '.join(cleaned_df.columns) if cleaned_df is not None else 'N/A'}
    """
    
    # Data Statistics
    if cleaned_df is not None:
        numeric_cols = cleaned_df.select_dtypes(include=['float64', 'int64']).columns
        if numeric_cols.any():
            stats = cleaned_df[numeric_cols].describe().to_dict()
            report += "\n## Key Statistics\n"
            for col, stat_dict in stats.items():
                report += f"- **{col}**:\n"
                report += f"  - Mean: {stat_dict['mean']:.2f}\n"
                report += f"  - Min: {stat_dict['min']:.2f}\n"
                report += f"  - Max: {stat_dict['max']:.2f}\n"

    # Visualizations
    if visualizations:
        report += "\n## Visualizations\n"
        for viz_type, path in visualizations.items():
            report += f"- [{viz_type.replace('_', ' ').title()}]({path})\n"

    # ML Models
    if ml_models and isinstance(ml_models, dict) and 'error' not in ml_models:
        report += "\n## Machine Learning Insights\n"
        for model_name, info in ml_models.items():
            report += f"- **{model_name.replace('_', ' ').title()}**\n"
            report += f"  - Training Score: {info.get('train_score', 'N/A'):.3f}\n"
            report += f"  - Test Score: {info.get('test_score', 'N/A'):.3f}\n"
            report += f"  - Target: {info.get('target', 'N/A')}\n"

    # Analyst Contributions
    report += "\n## Analyst Contributions\n"
    for analyst in analysts:
        report += f"- **{analyst.name}** ({analyst.data_analysis_focus})\n"
        for key, value in (analyst.analysis_results or {}).items():
            report += f"  - {key}: {value}\n"

    return {"analysis_summary": report}


tools = [
    perform_database_interactions,
    data_cleaning_and_enrichment,
    data_visualization,
    build_ml_models,
    generate_analysis_report
]

llm_with_tools = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def decide_next_step(state: GenerateAnalystsState):
    """Use LLM with tool calling to decide the next step based on user query."""
    query = state.get("user_query", "Perform a general analysis of the data")
    
    system_prompt = """
    You are an intelligent assistant tasked with deciding the next step in a data analysis workflow based on a user query.
    Use the available tools to choose the appropriate action. If the query is unclear, default to generating a report.
    Query: {query}
    """
    
    messages = [
        SystemMessage(content=system_prompt.format(query=query)),
        HumanMessage(content=f"What should be the next step for this query: {query}?")
    ]
    
    response = llm_with_tools.invoke(messages)
    
    # Extract tool call from response
    if response.tool_calls:
        tool_name = response.tool_calls[0]["name"]
        return {"next_step": tool_name} 
    else:
        return {"next_step": "generate_analysis_report"}


def build_analysis_workflow():
    builder = StateGraph(GenerateAnalystsState)
    
    # Define nodes
    builder.add_node("process_csv_and_create_db", process_csv_and_create_db)
    builder.add_node("create_analysts", create_analysts)
    builder.add_node("decide_next_step", decide_next_step)
    builder.add_node("perform_database_interactions", perform_database_interactions)
    builder.add_node("data_cleaning_and_enrichment", data_cleaning_and_enrichment)
    builder.add_node("data_visualization", data_visualization)
    builder.add_node("build_ml_models", build_ml_models)
    builder.add_node("generate_analysis_report", generate_analysis_report)
    
    # Tool node to handle tool execution
    tool_node = ToolNode(tools)
    builder.add_node("tools", tool_node)
    
    # Define workflow edges
    builder.add_edge(START, "process_csv_and_create_db")
    builder.add_edge("process_csv_and_create_db", "create_analysts")
    builder.add_edge("create_analysts", "decide_next_step")
    
    # Conditional edges based on LLM decision
    builder.add_conditional_edges(
        "decide_next_step",
        lambda state: state.get("next_step", "generate_analysis_report"),
        {
            "perform_database_interactions": "perform_database_interactions",
            "data_cleaning_and_enrichment": "data_cleaning_and_enrichment",
            "data_visualization": "data_visualization",
            "build_ml_models": "build_ml_models",
            "generate_analysis_report": "generate_analysis_report"
        }
    )
    
    # After each tool, go to report generation
    for node in ["perform_database_interactions", "data_cleaning_and_enrichment", 
                 "data_visualization", "build_ml_models"]:
        builder.add_edge(node, "generate_analysis_report")
    
    builder.add_edge("generate_analysis_report", END)
    
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


def run_analysis(csv_file_path, analysis_goal, max_analysts=4, database_name=None, table_name=None):
    """Run the analysis workflow on a CSV file"""
    graph = build_analysis_workflow()

    # Initial state setup
    initial_state = {
    "csv_path": csv_file_path,  
    "analysis_goal": analysis_goal,
    "max_analysts": max_analysts,
    "database_name": database_name,
    "table_name": table_name,
    "data_df": {} 
    }

    thread = {"configurable": {"thread_id": "1"}}

    print("Initial state before running graph:", initial_state)

    # Run graph and stream results
    results = graph.invoke(initial_state, thread)

    print("Final results:", results)
    return results
 


def main():
    st.set_page_config(page_title="Interactive Data Analysis", layout="wide")
    st.title("📊 Interactive Multi-Agent Data Analysis System")
    
    st.sidebar.header("Data Analysis Control Panel")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    user_query = st.sidebar.text_input("Ask a question about your data", "")
    run_button = st.sidebar.button("Run Analysis")
    
    if uploaded_file and run_button:
        csv_file_path = os.path.join(settings.upload_folder, "uploaded_data.csv")
        data = pd.read_csv(uploaded_file)
        data.to_csv(csv_file_path, index=False)
        
        graph = build_analysis_workflow()
        initial_state = {
            "csv_path": csv_file_path,
            "analysis_goal": "Dynamic analysis based on user query",
            "user_query": user_query,
            "max_analysts": 4,
            "data_df": {}
        }
        
        thread = {"configurable": {"thread_id": "1"}}
        results = graph.invoke(initial_state, thread)
        
        st.subheader("Analysis Results")
        
        if "analysis_summary" in results:
            st.markdown(results["analysis_summary"])
        
        if "visualizations" in results:
            st.subheader("Visualizations")
            for viz_name, viz_path in results["visualizations"].items():
                if os.path.exists(viz_path):
                    with open(viz_path, 'r') as f:
                        st.components.v1.html(f.read(), height=600)
        
        # Enhanced question answering
        if user_query and "data_df" in results and "cleaned" in results["data_df"]:
            df = pd.DataFrame(results["data_df"]["cleaned"])
            st.subheader("Quick Answer")
            
            query_lower = user_query.lower()
            if "how many" in query_lower:
                st.write(f"Dataset contains {len(df)} rows")
            elif "average" in query_lower or "mean" in query_lower:
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                if numeric_cols.any():
                    col = numeric_cols[0]
                    mean_val = df[col].mean()
                    st.write(f"Average {col}: {mean_val:.2f}")
            else:
                st.write("Detailed analysis provided in the report above.")

if __name__ == "__main__":
    main()