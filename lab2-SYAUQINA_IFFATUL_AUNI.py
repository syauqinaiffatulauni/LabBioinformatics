import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Assuming the retrieve functions and helper functions are defined in the same script or imported from another module
# from ppi_functions import retrieve_ppi_biogrid, retrieve_ppi_string, generate_network, get_centralities

# Dummy functions for example (replace with actual functions as defined previously)
def retrieve_ppi_biogrid(target_protein):
    # Example dummy data
    return pd.DataFrame({
        "ProteinA": ["BRCA1", "BRCA1", "BRCA1"],
        "ProteinB": ["TP53", "EGFR", "MYC"]
    })

def retrieve_ppi_string(target_protein):
    # Example dummy data
    return pd.DataFrame({
        "ProteinA": ["BRCA1", "BRCA1", "BRCA1"],
        "ProteinB": ["MDM2", "AKT1", "ATM"]
    })

def generate_network(dataframe):
    return nx.from_pandas_edgelist(dataframe, source="ProteinA", target="ProteinB")

def get_centralities(network_graph):
    degree_cent = nx.degree_centrality(network_graph)
    betweenness_cent = nx.betweenness_centrality(network_graph)
    closeness_cent = nx.closeness_centrality(network_graph)
    eigenvector_cent = nx.eigenvector_centrality(network_graph)
    pagerank_cent = nx.pagerank(network_graph)

    centrality_measures = {
        "Degree Centrality": degree_cent,
        "Betweenness Centrality": betweenness_cent,
        "Closeness Centrality": closeness_cent,
        "Eigenvector Centrality": eigenvector_cent,
        "PageRank Centrality": pagerank_cent
    }

    return centrality_measures

# Streamlit app layout
st.title("Protein-Protein Interaction (PPI) Viewer")
st.write("Retrieve and display Protein-Protein Interaction (PPI) data along with centrality measures.")

# Sidebar input for protein ID and database selection
protein_id = st.sidebar.text_input("Enter Protein ID:", value="BRCA1")
database = st.sidebar.selectbox("Choose PPI Database:", ["BioGRID", "STRING"])

# Button to retrieve PPI data
if st.sidebar.button("Retrieve PPI Data"):
    # Retrieve data based on selected database
    if database == "BioGRID":
        ppi_data = retrieve_ppi_biogrid(protein_id)
    elif database == "STRING":
        ppi_data = retrieve_ppi_string(protein_id)
    else:
        ppi_data = None

    # Check if data retrieval was successful
    if ppi_data is not None and not ppi_data.empty:
        # Generate network graph
        network_graph = generate_network(ppi_data)

        # Set up columns for displaying results
        col1, col2 = st.columns(2)

        # Column 1: PPI Data Information
        with col1:
            st.subheader("PPI Data Information")

            # Display PPI data in a dataframe
            st.dataframe(ppi_data)

            # Display details of the PPI data
            st.write(f"**Number of Nodes:** {network_graph.number_of_nodes()}")
            st.write(f"**Number of Edges:** {network_graph.number_of_edges()}")

            # Visualize the network
            st.write("**Network Visualization**")
            fig, ax = plt.subplots()
            pos = nx.spring_layout(network_graph)
            nx.draw_networkx(network_graph, pos, ax=ax, node_size=50, font_size=8)
            st.pyplot(fig)

        # Column 2: Centrality Measures
        with col2:
            st.subheader("Centrality Measures")

            # Calculate centralities
            centralities = get_centralities(network_graph)

            # Display centrality measures in a table
            centrality_df = pd.DataFrame(centralities).T
            st.dataframe(centrality_df)

            # Interpret centrality measures
            st.write("### Interpretation of Centrality Measures")
            st.write("""
                - **Degree Centrality**: Nodes with higher degree centrality are often key hubs in the network.
                - **Betweenness Centrality**: Nodes with high betweenness centrality serve as bridges in the network.
                - **Closeness Centrality**: Nodes with high closeness centrality are closer to other nodes on average.
                - **Eigenvector Centrality**: Nodes with high eigenvector centrality are influential, connected to other influential nodes.
                - **PageRank Centrality**: Nodes with high PageRank are important and tend to have more influential connections.
            """)
    else:
        st.error("No data retrieved. Please check the protein ID or try a different database.")
