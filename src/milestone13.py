import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import random
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r"datasets/integrated_tren_dataset_with_indexes.csv")
        return df
    except:
        st.error("Could not load the dataset. Please check the file path.")
        return None

@st.cache_data
def get_country_coordinates():
    coords = {
        'China': (35.8617, 104.1954), 'United States': (37.0902, -95.7129),
        'Japan': (36.2048, 138.2529), 'Germany': (51.1657, 10.4515),
        'India': (20.5937, 78.9629), 'United Kingdom': (55.3781, -3.4360),
        'France': (46.6034, 2.2137), 'Italy': (41.8719, 12.5674),
        'Brazil': (-14.2350, -51.9253), 'Canada': (56.1304, -106.3468),
        'Russia': (61.5240, 105.3188), 'South Korea': (35.9078, 127.7669),
        'Australia': (-25.2744, 133.7751), 'Spain': (40.4637, -3.7492),
        'Mexico': (23.6345, -102.5528), 'Indonesia': (-0.7893, 113.9213),
        'Netherlands': (52.1326, 5.2913), 'Saudi Arabia': (23.8859, 45.0792),
        'Turkey': (38.9637, 35.2433), 'Taiwan': (23.6978, 120.9605),
        'Switzerland': (46.8182, 8.2275), 'Belgium': (50.5039, 4.4699),
        'Argentina': (-38.4161, -63.6167), 'Ireland': (53.4129, -8.2439),
        'Poland': (51.9194, 19.1451), 'Thailand': (15.8700, 100.9925),
        'Nigeria': (9.0820, 8.6753), 'Egypt': (26.0975, 30.0444),
        'South Africa': (-30.5595, 22.9375), 'Malaysia': (4.2105, 101.9758),
        'Philippines': (12.8797, 121.7740), 'Vietnam': (14.0583, 108.2772),
        'Norway': (60.4720, 8.4689), 'Singapore': (1.3521, 103.8198),
        'United Arab Emirates': (23.4241, 53.8478), 'Czech Republic': (49.8175, 15.4730),
        'Finland': (61.9241, 25.7482), 'Portugal': (39.3999, -8.2245)
    }
    return coords

class FastTradeOptimizer:
    def __init__(self, df, budget, max_distance):
        self.df = df
        self.budget = budget
        self.max_distance = max_distance
        self.coords = get_country_coordinates()
        self.setup_optimization_data()
    
    def setup_optimization_data(self):
        latest_year = self.df['Year'].max()
        recent_data = self.df[self.df['Year'] >= latest_year - 1]
        
        self.countries = []
        self.country_data = {}
        
        for country in self.coords.keys():
            country_df = recent_data[recent_data['Country'] == country]
            if not country_df.empty:
                data = country_df.iloc[-1]
                self.countries.append(country)
                self.country_data[country] = {
                    'gdp': data.get('GDP (current US$)', 1e12),
                    'trade_pct': data.get('Trade (% of GDP)', 50),
                    'resilience': data.get('Resilience_Score', 0.5),
                    'trade_dep': data.get('Trade_Dependency_Index', 0.5),
                    'exports': data.get('Exports of goods and services (% of GDP)', 20),
                    'imports': data.get('Imports of goods and services (% of GDP)', 20)
                }
        
        self.countries = self.countries[:20]
        self.n_countries = len(self.countries)
        self.generate_potential_links()
    
    def generate_potential_links(self):
        self.potential_links = []
        self.existing_links = set()
        
        for i in range(self.n_countries):
            for j in range(i + 1, self.n_countries):
                country1, country2 = self.countries[i], self.countries[j]
                
                distance = geodesic(self.coords[country1], self.coords[country2]).kilometers
                
                if distance <= self.max_distance:
                    cost = self.calculate_link_cost(country1, country2, distance)
                    benefit = self.calculate_link_benefit(country1, country2)
                    
                    if not self.has_geopolitical_constraint(country1, country2):
                        
                        is_existing = self.is_existing_link(country1, country2)
                        
                        link_data = {
                            'id': len(self.potential_links),
                            'country1': country1,
                            'country2': country2,
                            'distance': distance,
                            'cost': cost,
                            'benefit': benefit,
                            'existing': is_existing
                        }
                        
                        self.potential_links.append(link_data)
                        
                        if is_existing:
                            self.existing_links.add((country1, country2))
    
    def calculate_link_cost(self, country1, country2, distance):
        base_cost = distance / 500
        
        infra_factor = 2 - (self.country_data[country1]['resilience'] + 
                           self.country_data[country2]['resilience']) / 2
        
        return max(1, base_cost * infra_factor)
    
    def calculate_link_benefit(self, country1, country2):
        data1 = self.country_data[country1]
        data2 = self.country_data[country2]
        
        gdp_factor = (data1['gdp'] + data2['gdp']) / 2e12
        
        trade_complementarity = abs(data1['exports'] - data2['imports']) + abs(data2['exports'] - data1['imports'])
        
        vulnerability = (data1['trade_dep'] * (1 - data1['resilience']) + 
                        data2['trade_dep'] * (1 - data2['resilience'])) / 2
        
        return gdp_factor * trade_complementarity * vulnerability * 10
    
    def is_existing_link(self, country1, country2):
        data1 = self.country_data[country1]
        data2 = self.country_data[country2]
        
        threshold = 40
        return data1['trade_pct'] > threshold and data2['trade_pct'] > threshold
    
    def has_geopolitical_constraint(self, country1, country2):
        restricted = [
            ('China', 'Taiwan'), ('Russia', 'Ukraine'), 
            ('India', 'Pakistan'), ('Israel', 'Iran')
        ]
        
        pair = tuple(sorted([country1, country2]))
        return any(set(pair) == set(r) for r in restricted)
    
    def calculate_network_resilience(self, selected_links):
        network = nx.Graph()
        
        for country in self.countries:
            network.add_node(country)
        
        for link in self.potential_links:
            if link['existing'] or link['id'] in selected_links:
                network.add_edge(link['country1'], link['country2'], 
                               weight=link['benefit'])
        
        max_loss = 0
        country_losses = {}
        
        for failed_country in self.countries:
            total_loss = self.calculate_failure_impact(network, failed_country)
            country_losses[failed_country] = total_loss
            max_loss = max(max_loss, total_loss)
        
        return max_loss, country_losses
    
    def calculate_failure_impact(self, network, failed_country):
        temp_network = network.copy()
        if failed_country in temp_network:
            temp_network.remove_node(failed_country)
        
        total_loss = 0
        
        failed_gdp = self.country_data[failed_country]['gdp']
        total_loss += failed_gdp * 0.3
        
        for country in self.countries:
            if country != failed_country and country in temp_network:
                original_degree = network.degree(country) if country in network else 0
                remaining_degree = temp_network.degree(country)
                
                connectivity_loss = max(0, (original_degree - remaining_degree) / max(1, original_degree))
                
                gdp = self.country_data[country]['gdp']
                trade_dep = self.country_data[country]['trade_dep']
                resilience = self.country_data[country]['resilience']
                
                impact_factor = connectivity_loss * trade_dep * (1 - resilience) * 0.15
                loss = gdp * impact_factor
                total_loss += loss
        
        return total_loss
    
    def optimize_greedy(self):
        available_links = [link for link in self.potential_links if not link['existing']]
        available_links.sort(key=lambda x: x['benefit'] / x['cost'], reverse=True)
        
        selected = []
        total_cost = 0
        
        for link in available_links:
            if total_cost + link['cost'] <= self.budget:
                selected.append(link['id'])
                total_cost += link['cost']
                
                if len(selected) >= 10:
                    break
        
        max_loss, country_losses = self.calculate_network_resilience(selected)
        
        selected_links_info = [link for link in available_links if link['id'] in selected]
        
        return {
            'selected_links': selected,
            'selected_links_info': selected_links_info,
            'total_cost': total_cost,
            'max_loss': max_loss,
            'country_losses': country_losses
        }
    
    def optimize_random_search(self, iterations=100):
        available_links = [link for link in self.potential_links if not link['existing']]
        
        best_solution = None
        best_max_loss = float('inf')
        
        for _ in range(iterations):
            random.shuffle(available_links)
            selected = []
            total_cost = 0
            
            for link in available_links:
                if total_cost + link['cost'] <= self.budget:
                    selected.append(link['id'])
                    total_cost += link['cost']
                    
                    if len(selected) >= 8:
                        break
            
            if selected:
                max_loss, country_losses = self.calculate_network_resilience(selected)
                
                if max_loss < best_max_loss:
                    best_max_loss = max_loss
                    selected_links_info = [link for link in available_links if link['id'] in selected]
                    
                    best_solution = {
                        'selected_links': selected,
                        'selected_links_info': selected_links_info,
                        'total_cost': total_cost,
                        'max_loss': max_loss,
                        'country_losses': country_losses
                    }
        
        return best_solution

def main():
    st.title("Trade Network Optimization")
    st.subheader("Minimize Maximum GDP Loss Under Single-Point Failures")
    
    df = load_data()
    if df is None:
        return
    
    st.subheader("Optimization Parameters")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        budget = st.slider("Budget Limit", 50, 300, 100)
    with col2:
        max_distance = st.slider("Max Distance (km)", 5000, 20000, 12000)
    with col3:
        algorithm = st.selectbox("Algorithm", ["Greedy", "Random Search"])
    
    if st.button("Run Optimization", type="primary"):
        with st.spinner("Setting up optimization problem..."):
            optimizer = FastTradeOptimizer(df, budget, max_distance)
        
        st.subheader("Problem Setup")
        setup_col1, setup_col2, setup_col3 = st.columns(3)
        with setup_col1:
            st.metric("Countries", len(optimizer.countries))
        with setup_col2:
            existing_count = sum(1 for link in optimizer.potential_links if link['existing'])
            st.metric("Existing Links", existing_count)
        with setup_col3:
            new_count = len(optimizer.potential_links) - existing_count
            st.metric("Potential New Links", new_count)
        
        baseline_max_loss, baseline_losses = optimizer.calculate_network_resilience([])
        
        with st.spinner(f"Running {algorithm.lower()} optimization..."):
            if algorithm == "Greedy":
                solution = optimizer.optimize_greedy()
            else:
                solution = optimizer.optimize_random_search(200)
        
        if solution:
            st.subheader("Optimization Results")
            
            improvement = ((baseline_max_loss - solution['max_loss']) / baseline_max_loss * 100)
            
            result_col1, result_col2, result_col3, result_col4 = st.columns(4)
            with result_col1:
                st.metric("Max Loss Reduction", f"{improvement:.1f}%")
            with result_col2:
                st.metric("Baseline Max Loss", f"${baseline_max_loss/1e12:.2f}T")
            with result_col3:
                st.metric("Optimized Max Loss", f"${solution['max_loss']/1e12:.2f}T")
            with result_col4:
                st.metric("Budget Used", f"{solution['total_cost']:.1f}/{budget}")
            
            if solution['selected_links_info']:
                st.subheader("Selected Links")
                
                selected_df = pd.DataFrame([{
                    'From': link['country1'],
                    'To': link['country2'],
                    'Distance (km)': f"{link['distance']:.0f}",
                    'Cost': f"{link['cost']:.1f}",
                    'Benefit': f"{link['benefit']:.2f}",
                    'Benefit/Cost': f"{link['benefit']/link['cost']:.2f}"
                } for link in solution['selected_links_info']])
                
                st.dataframe(selected_df, use_container_width=True)
                
                st.subheader("Country Failure Impact Analysis")
                
                comparison_data = []
                for country in optimizer.countries:
                    baseline_loss = baseline_losses.get(country, 0)
                    optimized_loss = solution['country_losses'].get(country, 0)
                    reduction = ((baseline_loss - optimized_loss) / baseline_loss * 100) if baseline_loss > 0 else 0
                    
                    comparison_data.append({
                        'Country': country,
                        'Baseline Loss': baseline_loss / 1e12,
                        'Optimized Loss': optimized_loss / 1e12,
                        'Reduction (%)': reduction
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df = comparison_df.sort_values('Reduction (%)', ascending=False)
                
                fig = px.bar(comparison_df.head(12), 
                           x='Country', 
                           y=['Baseline Loss', 'Optimized Loss'],
                           title="GDP Loss Comparison (Trillions USD)",
                           barmode='group')
                fig.update_layout(height=500, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Network Visualization")
                
                nodes_data = []
                for country in optimizer.countries:
                    if country in optimizer.coords:
                        lat, lon = optimizer.coords[country]
                        gdp = optimizer.country_data[country]['gdp']
                        nodes_data.append({
                            'Country': country,
                            'Latitude': lat,
                            'Longitude': lon,
                            'GDP_Trillions': gdp/1e12
                        })
                
                if nodes_data:
                    nodes_df = pd.DataFrame(nodes_data)
                    
                    fig = px.scatter_geo(nodes_df,
                                       lat='Latitude',
                                       lon='Longitude',
                                       size='GDP_Trillions',
                                       hover_name='Country',
                                       title="Optimized Trade Network",
                                       projection='natural earth',
                                       size_max=20)
                    
                    for link in solution['selected_links_info']:
                        if link['country1'] in optimizer.coords and link['country2'] in optimizer.coords:
                            lat1, lon1 = optimizer.coords[link['country1']]
                            lat2, lon2 = optimizer.coords[link['country2']]
                            
                            fig.add_trace(go.Scattergeo(
                                lon=[lon1, lon2],
                                lat=[lat1, lat2],
                                mode='lines',
                                line=dict(width=2, color='red'),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Sensitivity Analysis")
            
            st.write(f"**Current Settings:** Budget={budget}, Max Distance={max_distance}km")
            st.write(f"**Results:** {len(solution['selected_links_info'])} new links, {improvement:.1f}% improvement")
            
            sensitivity_data = []
            test_budgets = [budget * 0.7, budget, budget * 1.3, budget * 1.6]
            
            for test_budget in test_budgets:
                test_optimizer = FastTradeOptimizer(df, test_budget, max_distance)
                if algorithm == "Greedy":
                    test_solution = test_optimizer.optimize_greedy()
                else:
                    test_solution = test_optimizer.optimize_random_search(50)
                
                if test_solution:
                    test_improvement = ((baseline_max_loss - test_solution['max_loss']) / baseline_max_loss * 100)
                    sensitivity_data.append({
                        'Budget': test_budget,
                        'Links Selected': len(test_solution['selected_links_info']),
                        'Improvement (%)': test_improvement,
                        'Cost Used': test_solution['total_cost']
                    })
            
            
    
    st.subheader("Mathematical Formulation")
    st.latex(r'''
    \min \quad Z = \max_{k} \sum_{i \neq k} L_i(k)
    ''')
    st.latex(r'''
    \text{subject to:} \quad \sum_{(i,j)} c_{ij} x_{ij} \leq B
    ''')
    st.latex(r'''
    d_{ij} \leq D_{\max}, \quad x_{ij} \in \{0,1\}
    ''')
    
    st.write("Where $L_i(k)$ is the GDP loss of country $i$ when country $k$ fails, $x_{ij}$ are binary link decisions, $c_{ij}$ are link costs, and $B$ is the budget.")

if __name__ == "__main__":
    main()