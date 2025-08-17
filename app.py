import streamlit as st
from src import forecasting
from src import visualization
from src import policy_scenarios
from src import milestone1
from src import milestone2
from src import milestone3
from src import milestone4
from src import milestone5
from src import milestone6
from src import milestone7
from src import milestone8
from src import milestone9
from src import milestone10
from src import milestone11
from src import milestone12
from src import milestone13
def main():

    page = st.sidebar.selectbox(
        "Choose a page",
        ["Home", "Forecasting","Visualisations","Policies","Trade Risk Analysis", "China Export Drop Percentage",
         "Drought Shock Simulation","Food Security Analysis","Youth Unemployment Projection","Export Sector Ageing Risk",
         "Global Trade Network Analysis","Trade Relationship Mutual Benefit Analysis","Trade Partner Suggestions","Economic Prediction for 2030",
         "2030 Economic Scenario Projections","Resilience Projections","Trade Network Optimization"]
    )
    
    if page == "Home":
        st.subheader("Welcome to the Economic Data Analysis Dashboard")
        st.write("""
        This dashboard provides comprehensive analysis of economic data including:
        
        - **Forecasting**: Economic projections and scenario modeling for 2030
        - **Trade Risk Analysis**: Trade dependency vulnerability assessment
        
        Navigate using the sidebar to explore different sections.
        """)
        
        st.info("Start with the **Forecasting** section to explore GDP growth, poverty rates, and trade resilience projections!")
        st.info("Check out **Trade Risk Analysis** to identify countries most vulnerable to trade partner collapse!")


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
        milestone4.main()
        
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