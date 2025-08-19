import streamlit as st
from src import forecasting
from src import visualization
from src import policy_scenarios
from src import milestone1
from src import milestone2
from src import milestone3
from src import milestone4other
from src import milestone5
from src import milestone6
from src import milestone7
from src import milestone8
from src import milestone9
from src import milestone10
from src import milestone11
from src import milestone12
from src import milestone13

def create_feature_card(title, description, icon, page_key):
    """Create a feature card with styling"""
    card_html = f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.18);
        backdrop-filter: blur(10px);
        color: white;
        transition: transform 0.3s ease;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <div style="font-size: 2rem; margin-right: 1rem;">{icon}</div>
            <h3 style="margin: 0; font-weight: 600;">{title}</h3>
        </div>
        <p style="margin: 0; opacity: 0.9; line-height: 1.5;">{description}</p>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def main():
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.3rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    
    .stat-box {
        text-align: center;
        padding: 1.5rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 0.5rem;
        min-width: 150px;
        border: 2px solid #f0f0f0;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        color: #667eea;
        display: block;
    }
    
    .stat-label {
        color: #666;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    .section-header {
        color: #333;
        font-size: 2rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        text-align: center;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .cta-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 3rem 0;
        color: white;
    }
    
    .cta-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .cta-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

    page = st.sidebar.selectbox(
        "Navigate Dashboard",
        ["Home", "Forecasting", "Visualisations", "Policies", "Trade Risk Analysis", 
         "China Export Drop Percentage", "Drought Shock Simulation", "Food Security Analysis",
         "Youth Unemployment Projection", "Export Sector Ageing Risk", "Global Trade Network Analysis",
         "Trade Relationship Mutual Benefit Analysis", "Trade Partner Suggestions", 
         "Economic Prediction for 2030", "2030 Economic Scenario Projections", 
         "Resilience Projections", "Trade Network Optimization"]
    )

    if page == "Home":
        st.markdown('<h1 class="main-header">Economic Intelligence Hub</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Advanced Analytics & Strategic Insights for Global Economic Decision Making</p>', unsafe_allow_html=True)
        
        st.markdown('<div class="stats-container">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="stat-box">
                <span class="stat-number">16</span>
                <div class="stat-label">Analysis Tools</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="stat-box">
                <span class="stat-number">2030</span>
                <div class="stat-label">Forecast Horizon</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="stat-box">
                <span class="stat-number">Global</span>
                <div class="stat-label">Trade Coverage</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="stat-box">
                <span class="stat-number">Real-time</span>
                <div class="stat-label">Data Processing</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<h2 class="section-header">Core Analytics Suite</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            create_feature_card(
                "Economic Forecasting", 
                "Advanced GDP growth projections, poverty rate analysis, and comprehensive 2030 economic scenarios with machine learning models.",
                "📈", "forecasting"
            )
            create_feature_card(
                "Trade Risk Intelligence", 
                "Vulnerability assessment for trade dependencies, partner collapse scenarios, and strategic risk mitigation planning.",
                "⚠️", "trade_risk"
            )
            create_feature_card(
                "Policy Impact Modeling", 
                "Simulate economic policies, trade agreements, and regulatory changes to forecast their economic impact.",
                "🏛️", "policies"
            )
        
        with col2:
            create_feature_card(
                "Interactive Visualizations", 
                "Dynamic charts, network graphs, and geospatial analysis for comprehensive data exploration and insights.",
                "📊", "visualizations"
            )
            create_feature_card(
                "Global Trade Networks", 
                "Analyze international trade relationships, identify optimal partners, and optimize trade network efficiency.",
                "🌐", "trade_network"
            )
            create_feature_card(
                "Crisis Simulation", 
                "Model drought impacts, export sector risks, and economic shocks to build resilient strategies.",
                "🛡️", "simulation"
            )
        
        st.markdown('<h2 class="section-header">Specialized Analysis Tools</h2>', unsafe_allow_html=True)
        
        with st.expander("**Global Trade & Security Analysis**", expanded=False):
            col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            - **China Export Analysis**  
            Analyze impact of Chinese export fluctuations on global markets  

            - **Food Security Modeling**  
            Assess food security risks and supply chain vulnerabilities  
            """)

        with col2:
            st.markdown("""
            - **Agricultural Risk Assessment**  
            Simulate drought impacts and agricultural sector resilience  

            - **Mutual Benefit Analysis**  
            Evaluate trade relationships for mutual economic benefits  
            """)

        with col3:
            st.markdown("""
            - **Trade Partner Optimization**  
            AI-powered suggestions for optimal trade partnerships  

            - **Network Optimization**  
            Optimize trade networks for maximum efficiency and resilience  
            """)

        with st.expander("**Demographic & Employment Insights**", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                - **Youth Unemployment Forecasting**  
                Project youth employment trends and identify intervention points  
                """)
            with col2:
                st.markdown("""
                - **Sector Ageing Risk Analysis**  
                Assess risks from demographic changes in export sectors  
                """)

        with st.expander("**Future Scenario Planning**", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                - **2030 Economic Projections**  
                Comprehensive economic scenario modeling for the next decade  
                """)
            with col2:
                st.markdown("""
                - **Resilience Planning**  
                Build economic resilience strategies against multiple risk factors  
                """)
        
        st.markdown("""
        <div class="cta-section">
            <div class="cta-title">Ready to Explore Economic Intelligence?</div>
            <div class="cta-subtitle">Start with our most popular tools or dive deep into specialized analysis</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("###  Recommended Starting Points")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📈 **Start with Forecasting**", use_container_width=True):
                st.info("Navigate to 'Forecasting' in the sidebar to explore GDP growth, poverty rates, and trade resilience projections!")
        
        with col2:
            if st.button("⚠️ **Assess Trade Risks**", use_container_width=True):
                st.info("Check out 'Trade Risk Analysis' to identify countries most vulnerable to trade partner collapse!")
        
        with col3:
            if st.button("📊 **Explore Data Visualizations**", use_container_width=True):
                st.info("Visit 'Visualisations' for interactive charts and comprehensive data exploration!")
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem;">
            <p><strong>Economic Intelligence Hub</strong> - Empowering data-driven economic decisions</p>
            <p> Use the sidebar navigation to explore all available analysis tools and insights</p>
        </div>
        """, unsafe_allow_html=True)

    elif page == "Forecasting":
        forecasting.main()
    elif page == "Visualisations":
        visualization.main()
    elif page == "Policies":
        policy_scenarios.main()
    elif page == "Trade Risk Analysis":
        milestone1.main()
    elif page == "China Export Drop Percentage":
        milestone2.main()
    elif page == "Drought Shock Simulation":
        milestone3.main()
    elif page == "Food Security Analysis":
        milestone4other.main()
    elif page == "Youth Unemployment Projection":
        milestone5.main()
    elif page == "Export Sector Ageing Risk":
        milestone6.main()
    elif page == "Global Trade Network Analysis":
        milestone7.main()
    elif page == "Trade Relationship Mutual Benefit Analysis":
        milestone8.main()
    elif page == "Trade Partner Suggestions":
        milestone9.main()
    elif page == "Economic Prediction for 2030":
        milestone10.main()
    elif page == "2030 Economic Scenario Projections":
        milestone11.main()
    elif page == "Resilience Projections":
        milestone12.main()
    elif page == "Trade Network Optimization":
        milestone13.main()

if __name__ == "__main__":
    main()
