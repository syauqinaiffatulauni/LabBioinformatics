import streamlit as st
from Bio import Entrez, SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Set the Entrez email for API access
Entrez.email = "syauqinaiffatulauni@graduate.utm.my"

# Function to retrieve protein data
def retrieve_data(uniprot_id, email):
    Entrez.email = email
    try:
        # Fetch the record from the NCBI protein database
        with Entrez.efetch(db="protein", id=uniprot_id, rettype="gb", retmode="text") as handle:
            record = SeqIO.read(handle, "genbank")
        return record
    except Exception as e:
        st.error(f"Error retrieving data for {uniprot_id}: {e}")
        return None

# Function to perform basic analysis on the protein sequence
def get_basic_analysis(sequence):
    analysis = ProteinAnalysis(sequence)
    
    # Calculate various properties
    results = {
        "Sequence Length": len(sequence),
        "Amino Acid Composition": analysis.count_amino_acids(),
        "Molecular Weight": analysis.molecular_weight(),
        "Isoelectric Point": analysis.isoelectric_point()
    }
    
    return results

# Streamlit UI setup
st.title('Lab 1 - SYAUQINA IFFATUL AUNI')

# Get user input
protein_id = st.text_input('Enter Uniprot ID')
retrieve = st.button('Retrieve')

# Retrieve and display data when the button is clicked
if retrieve:
    if protein_id != "":
        # Retrieve protein data
        record = retrieve_data(protein_id, "syauqinaiffatulauni@graduate.utm.my")
        
        if record:
            # Display protein data
            col1, col2 = st.columns(2)

        with col1:
            st.header("Retrieved Protein:")
            st.write(f"ID: {record.id}")
            st.write(f"Description: {record.description}")
            st.write(f"Sequence: {record.seq}")

            # Perform and display basic analysis
        with col2:
            analysis_result = get_basic_analysis(str(record.seq))
            st.header("\nBasic Protein Analysis:")
            for key, value in analysis_result.items():
                st.write(f"{key}: {value}")
    else:
        st.warning('Please enter Uniprot ID')
