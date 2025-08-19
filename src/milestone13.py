import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import linprog
import random
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def load_sample_data():
    countries = [
        'China', 'United States', 'Japan', 'Germany', 'India', 'United Kingdom',
        'France', 'Italy', 'Brazil', 'Canada', 'Russia', 'South Korea',
        'Australia', 'Spain', 'Mexico', 'Indonesia', 'Netherlands', 'Saudi Arabia',
        'Turkey', 'Taiwan', 'Switzerland', 'Belgium', 'Argentina', 'Ireland',
        'Poland', 'Thailand', 'Nigeria', 'Egypt', 'South Africa', 'Malaysia'
    ]
    
    data = []
    years = [2022, 2023]
    
    for year in years:
        for country in countries:
            data.append({
                'Country': country,
                'Year': year,
                'GDP (current US$)': np.random.uniform(0.5e12, 25e12),
                'Trade (% of GDP)': np.random.uniform(20, 80),
                'Resilience_Score': np.random.uniform(0.3, 0.9),
                'Trade_Dependency_Index': np.random.uniform(0.2, 0.8),
                'Exports of goods and services (% of GDP)': np.random.uniform(15, 45),
                'Imports of goods and services (% of GDP)': np.random.uniform(15, 45)
            })
    
    return pd.DataFrame(data)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r"datasets/processed/integrated_tren_dataset.csv")
        return df
    except:
        st.warning("Dataset file not found. Using generated sample data.")
        return load_sample_data()

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

class EnhancedTradeOptimizer:
    def __init__(self, df, budget, max_distance, min_connections=2, capacity_factor=0.1):
        self.df = df
        self.budget = budget
        self.max_distance = max_distance
        self.min_connections = min_connections
        self.capacity_factor = capacity_factor
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
                    capacity = self.calculate_link_capacity(country1, country2)  # NEW: Added capacity constraint
                    
                    if not self.has_geopolitical_constraint(country1, country2):
                        is_existing = self.is_existing_link(country1, country2)
                        
                        link_data = {
                            'id': len(self.potential_links),
                            'country1': country1,
                            'country2': country2,
                            'distance': distance,
                            'cost': cost,
                            'benefit': benefit,
                            'capacity': capacity,  # NEW: Store capacity
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
 
    def calculate_link_capacity(self, country1, country2):
        gdp_min = min(self.country_data[country1]['gdp'], 
                      self.country_data[country2]['gdp'])
        infra_avg = (self.country_data[country1]['resilience'] + 
                     self.country_data[country2]['resilience']) / 2
        return gdp_min * infra_avg * self.capacity_factor
    
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
    
    def check_minimum_connectivity(self, network):
        for country in self.countries:
            if country in network and network.degree(country) < self.min_connections:
                return False
        return True
    
    def check_network_connectivity(self, network):
        if len(network.nodes()) <= 2:
            return True
            
        for node in list(network.nodes()):
            temp_network = network.copy()
            temp_network.remove_node(node)
            
            if len(temp_network.nodes()) > 0 and not nx.is_connected(temp_network):
                return False
        return True
    
    def calculate_network_resilience(self, selected_links):
        network = nx.Graph()
        
        for country in self.countries:
            network.add_node(country)
        
        total_capacity = 0  # NEW: Track total network capacity
        for link in self.potential_links:
            if link['existing'] or link['id'] in selected_links:
                network.add_edge(link['country1'], link['country2'], 
                               weight=link['benefit'],
                               capacity=link['capacity'])  # NEW: Store capacity in network
                total_capacity += link['capacity']
        
        max_loss = 0
        country_losses = {}
        connectivity_penalty = 0
        if not self.check_minimum_connectivity(network):
            connectivity_penalty += 1e12  # Large penalty for insufficient connectivity
        if not self.check_network_connectivity(network):
            connectivity_penalty += 1e12  # Large penalty for network disconnection
        
        for failed_country in self.countries:
            total_loss = self.calculate_failure_impact(network, failed_country)
            country_losses[failed_country] = total_loss
            max_loss = max(max_loss, total_loss)
        
        return max_loss + connectivity_penalty, country_losses, total_capacity
    
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
        
        selected = self.ensure_connectivity_constraints(selected, available_links)
        
        max_loss, country_losses, total_capacity = self.calculate_network_resilience(selected)
        
        selected_links_info = [link for link in available_links if link['id'] in selected]
        
        return {
            'selected_links': selected,
            'selected_links_info': selected_links_info,
            'total_cost': sum(link['cost'] for link in selected_links_info),
            'max_loss': max_loss,
            'country_losses': country_losses,
            'total_capacity': total_capacity
        }
    
    def optimize_random_search(self, iterations=100):
        available_links = [link for link in self.potential_links if not link['existing']]
        
        best_solution = None
        best_max_loss = float('inf')
        
        for iteration in range(iterations):
            random.shuffle(available_links)
            selected = []
            total_cost = 0
            
            # Random selection within budget
            for link in available_links:
                if total_cost + link['cost'] <= self.budget:
                    selected.append(link['id'])
                    total_cost += link['cost']
                    
                    if len(selected) >= 8:
                        break
            
            if selected:
                if iteration % 10 == 0:  # Every 10th iteration, try to fix constraints
                    selected = self.ensure_connectivity_constraints(selected, available_links)
                
                max_loss, country_losses, total_capacity = self.calculate_network_resilience(selected)
                
                if max_loss < best_max_loss:
                    best_max_loss = max_loss
                    selected_links_info = [link for link in available_links if link['id'] in selected]
                    
                    best_solution = {
                        'selected_links': selected,
                        'selected_links_info': selected_links_info,
                        'total_cost': sum(link['cost'] for link in selected_links_info),
                        'max_loss': max_loss,
                        'country_losses': country_losses,
                        'total_capacity': total_capacity
                    }
        
        return best_solution
    def ensure_connectivity_constraints(self, selected, available_links):
        network = nx.Graph()
        for country in self.countries:
            network.add_node(country)
        for link in self.potential_links:
            if link['existing']:
                network.add_edge(link['country1'], link['country2'])
        for link_id in selected:
            link = self.potential_links[link_id]
            network.add_edge(link['country1'], link['country2'])
        
        attempts = 0
        max_attempts = 10
        current_cost = sum(self.potential_links[link_id]['cost'] for link_id in selected)
        
        while (not self.check_minimum_connectivity(network) and 
               attempts < max_attempts and 
               current_cost < self.budget * 0.9): 
            under_connected = [country for country in self.countries 
                             if network.degree(country) < self.min_connections]
            
            if under_connected:
                for country in under_connected:
                    best_link = None
                    best_score = -1
                    
                    for link in available_links:
                        if (link['id'] not in selected and 
                            (link['country1'] == country or link['country2'] == country) and
                            current_cost + link['cost'] <= self.budget):
                            
                            score = link['benefit'] / link['cost']
                            if score > best_score:
                                best_link = link
                                best_score = score
                    
                    if best_link:
                        selected.append(best_link['id'])
                        current_cost += best_link['cost']
                        network.add_edge(best_link['country1'], best_link['country2'])
                        break
            
            attempts += 1
        
        return selected

def main():
    st.title("Enhanced Trade Network Optimization")
    st.subheader("Minimize Maximum GDP Loss Under Single-Point Failures")
    
    df = load_data()
    if df is None:
        return
    st.sidebar.header("Optimization Parameters")
    
    budget = st.sidebar.slider("Budget Limit", 50, 300, 100, help="Maximum budget for new trade links")
    max_distance = st.sidebar.slider("Max Distance (km)", 5000, 20000, 12000, help="Maximum distance for trade links")
    min_connections = st.sidebar.slider("Min Connections per Country", 1, 4, 2, help="Minimum trade connections per country")
    capacity_factor = st.sidebar.slider("Capacity Factor", 0.05, 0.2, 0.1, help="Link capacity as fraction of smaller country's GDP")
    algorithm = st.sidebar.selectbox("Algorithm", ["Greedy ", "Random Search"])
    
    # Advanced settings
    with st.sidebar.expander(" Advanced Settings"):
        if "Random Search" in algorithm:
            iterations = st.slider("Random Search Iterations", 50, 300, 200)
        else:
            iterations = 100
    
    if st.button(" Run Optimization", type="primary"):
        with st.spinner("Setting up optimization problem..."):
            optimizer = EnhancedTradeOptimizer(df, budget, max_distance, min_connections, capacity_factor)
        
        st.subheader(" Problem Setup")
        setup_col1, setup_col2, setup_col3, setup_col4 = st.columns(4)
        with setup_col1:
            st.metric("Countries", len(optimizer.countries))
        with setup_col2:
            existing_count = sum(1 for link in optimizer.potential_links if link['existing'])
            st.metric("Existing Links", existing_count)
        with setup_col3:
            new_count = len(optimizer.potential_links) - existing_count
            st.metric("Potential New Links", new_count)
        with setup_col4:
            total_capacity = sum(link['capacity'] for link in optimizer.potential_links)
            st.metric("Total Network Capacity", f"${total_capacity/1e12:.1f}T")
        
        baseline_max_loss, baseline_losses, baseline_capacity = optimizer.calculate_network_resilience([])
        
        with st.spinner(f"Running {algorithm.lower()}..."):
            if "Greedy" in algorithm:
                solution = optimizer.optimize_greedy()
            else:
                solution = optimizer.optimize_random_search(iterations)
        
        if solution:
            st.subheader(" Optimization Results")
            
            improvement = ((baseline_max_loss - solution['max_loss']) / baseline_max_loss * 100)
            capacity_increase = ((solution['total_capacity'] - baseline_capacity) / baseline_capacity * 100) if baseline_capacity > 0 else 0
            
            result_col1, result_col2, result_col3, result_col4, result_col5 = st.columns(5)
            with result_col1:
                st.metric("Max Loss Reduction", f"{improvement:.1f}%", delta=f"{improvement:.1f}%")
            with result_col2:
                st.metric("Baseline Max Loss", f"${baseline_max_loss/1e12:.2f}T")
            with result_col3:
                st.metric("Optimized Max Loss", f"${solution['max_loss']/1e12:.2f}T")
            with result_col4:
                st.metric("Budget Used", f"{solution['total_cost']:.1f}/{budget}")
            with result_col5:
                st.metric("Capacity Increase", f"{capacity_increase:.1f}%", delta=f"{capacity_increase:.1f}%")
            
            network = nx.Graph()
            for country in optimizer.countries:
                network.add_node(country)
            for link in optimizer.potential_links:
                if link['existing'] or link['id'] in solution['selected_links']:
                    network.add_edge(link['country1'], link['country2'])
            
            constraint_col1, constraint_col2, constraint_col3 = st.columns(3)
            with constraint_col1:
                budget_ok = solution['total_cost'] <= budget
                st.metric("Budget Constraint", " Satisfied" if budget_ok else " Violated")
            with constraint_col2:
                connectivity_ok = optimizer.check_minimum_connectivity(network)
                st.metric("Min Connectivity", " Satisfied" if connectivity_ok else " Partial")
            with constraint_col3:
                network_ok = optimizer.check_network_connectivity(network)
                st.metric("Network Robustness", " Robust" if network_ok else " Fragile")
            
            if solution['selected_links_info']:
                st.subheader(" Selected Links")
                
                selected_df = pd.DataFrame([{
                    'From': link['country1'],
                    'To': link['country2'],
                    'Distance (km)': f"{link['distance']:.0f}",
                    'Cost': f"{link['cost']:.1f}",
                    'Benefit': f"{link['benefit']:.2f}",
                    'Capacity ($B)': f"{link['capacity']/1e9:.1f}",
                    'Benefit/Cost': f"{link['benefit']/link['cost']:.2f}"
                } for link in solution['selected_links_info']])
                
                st.dataframe(selected_df, use_container_width=True)
                
                st.subheader(" Country Failure Impact Analysis")
                
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
                           title="GDP Loss Comparison (Trillions USD) - Top 12 Countries",
                           barmode='group')
                fig.update_layout(height=500, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader(" Network Visualization")
                
                nodes_data = []
                for country in optimizer.countries:
                    if country in optimizer.coords:
                        lat, lon = optimizer.coords[country]
                        gdp = optimizer.country_data[country]['gdp']
                        degree = network.degree(country) if country in network else 0
                        nodes_data.append({
                            'Country': country,
                            'Latitude': lat,
                            'Longitude': lon,
                            'GDP_Trillions': gdp/1e12,
                            'Connections': degree
                        })
                
                if nodes_data:
                    nodes_df = pd.DataFrame(nodes_data)
                    
                    fig = px.scatter_geo(nodes_df,
                                       lat='Latitude',
                                       lon='Longitude',
                                       size='GDP_Trillions',
                                       color='Connections',
                                       hover_name='Country',
                                       title="Optimized Trade Network (Red=New Links, Blue=Existing)",
                                       projection='natural earth',
                                       size_max=20,
                                       color_continuous_scale='Viridis')
                    
                    for link in optimizer.potential_links:
                        if link['existing'] and link['country1'] in optimizer.coords and link['country2'] in optimizer.coords:
                            lat1, lon1 = optimizer.coords[link['country1']]
                            lat2, lon2 = optimizer.coords[link['country2']]
                            
                            fig.add_trace(go.Scattergeo(
                                lon=[lon1, lon2],
                                lat=[lat1, lat2],
                                mode='lines',
                                line=dict(width=1, color='blue', dash='dot'),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                    
                    for link in solution['selected_links_info']:
                        if link['country1'] in optimizer.coords and link['country2'] in optimizer.coords:
                            lat1, lon1 = optimizer.coords[link['country1']]
                            lat2, lon2 = optimizer.coords[link['country2']]
                            
                            fig.add_trace(go.Scattergeo(
                                lon=[lon1, lon2],
                                lat=[lat1, lat2],
                                mode='lines',
                                line=dict(width=3, color='red'),
                                showlegend=False,
                                hoverinfo='text',
                                text=f"{link['country1']} ↔ {link['country2']}<br>Cost: {link['cost']:.1f}<br>Capacity: ${link['capacity']/1e9:.1f}B"
                            ))
                    
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
            
            st.subheader(" Sensitivity Analysis")
            
            st.write(f"**Current Settings:** Budget={budget}, Max Distance={max_distance}km, Min Connections={min_connections}")
            st.write(f"**Results:** {len(solution['selected_links_info'])} new links, {improvement:.1f}% improvement")
            
            sensitivity_data = []
            test_budgets = [int(budget * 0.7), budget, int(budget * 1.3)]
            
            for test_budget in test_budgets:
                test_optimizer = EnhancedTradeOptimizer(df, test_budget, max_distance, min_connections, capacity_factor)
                if "Greedy" in algorithm:
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
            
            if sensitivity_data:
                sens_df = pd.DataFrame(sensitivity_data)
                fig = px.line(sens_df, x='Budget', y='Improvement (%)', 
                             title='Budget vs Improvement Sensitivity',
                             markers=True)
                st.plotly_chart(fig, use_container_width=True)
        st.subheader(" Mathematical Formulation")
    st.latex(r'''
    \min \quad Z = \max_{k} \sum_{i \neq k} L_i(k)
    ''')
    st.latex(r'''
    \text{subject to:} \quad \sum_{(i,j)} c_{ij} x_{ij} \leq B
    ''')
    st.latex(r'''
    d_{ij} \leq D_{\max}, \quad \text{degree}(i) \geq M_{\min}, \quad \text{capacity}_{ij} \leq C_{ij}
    ''')
    st.latex(r'''
    x_{ij} \in \{0,1\}
    ''')
    
    st.write("""
    **Enhanced Constraints:**
    - **Budget**: $\sum c_{ij} x_{ij} \leq B$ (Total cost within budget)  
    - **Distance**: $d_{ij} \leq D_{max}$ (Geographic feasibility)
    - **Connectivity**: $degree(i) \geq M_{min}$ (Minimum connections per country)
    - **Capacity**: $capacity_{ij} \leq C_{ij}$ (Infrastructure-based capacity limits)""")
if __name__ == "__main__":
    main()
