"""
Drug Interaction Graph Visualizer
=================================
Creates interactive PyVis network visualizations of drug interactions.
Embeddable in Streamlit via HTML component.

Recruiter Talking Point:
"I added interactive graph visualization so users can see their medication
network visually. Nodes are drugs, edges are interactions color-coded by
severity. This makes complex polypharmacy scenarios immediately understandable."
"""

from pyvis.network import Network
from typing import Optional
import tempfile
import os


# Severity color mapping
SEVERITY_COLORS = {
    "major": "#e74c3c",      # Red
    "moderate": "#f39c12",   # Orange/Yellow
    "minor": "#27ae60",      # Green
    "unknown": "#95a5a6"     # Gray
}

SEVERITY_WIDTH = {
    "major": 4,
    "moderate": 2,
    "minor": 1,
    "unknown": 1
}


class DrugInteractionGraph:
    """
    Creates interactive network visualizations of drug interactions.
    """
    
    def __init__(
        self,
        height: str = "500px",
        width: str = "100%",
        bgcolor: str = "#0e1117",  # Dark theme to match Streamlit
        font_color: str = "#ffffff"
    ):
        self.height = height
        self.width = width
        self.bgcolor = bgcolor
        self.font_color = font_color
    
    def create_network(
        self,
        drugs: list[str],
        interactions: list[dict],
        title: str = "Drug Interaction Network"
    ) -> Network:
        """
        Create a PyVis network from drugs and interactions.
        
        Args:
            drugs: List of drug names
            interactions: List of dicts with keys: drug1, drug2, severity, description
        
        Returns:
            PyVis Network object
        """
        # Initialize network
        net = Network(
            height=self.height,
            width=self.width,
            bgcolor=self.bgcolor,
            font_color=self.font_color,
            directed=False,
            notebook=False
        )
        
        # Physics settings for nice layout
        net.barnes_hut(
            gravity=-3000,
            central_gravity=0.3,
            spring_length=200,
            spring_strength=0.05,
            damping=0.09
        )
        
        # Track which drugs have interactions
        drugs_with_interactions = set()
        for interaction in interactions:
            drugs_with_interactions.add(interaction["drug1"])
            drugs_with_interactions.add(interaction["drug2"])
        
        # Add drug nodes
        for drug in drugs:
            # Larger node if it has interactions
            has_interactions = drug in drugs_with_interactions
            size = 30 if has_interactions else 20
            color = "#3498db" if has_interactions else "#7f8c8d"  # Blue vs gray
            
            net.add_node(
                drug,
                label=drug,
                title=f"{drug}\nClick to see interactions",
                size=size,
                color=color,
                font={"size": 14, "color": self.font_color}
            )
        
        # Add interaction edges
        for interaction in interactions:
            severity = interaction.get("severity", "unknown")
            description = interaction.get("description", "No description")
            
            # Truncate description for tooltip
            desc_short = description[:150] + "..." if len(description) > 150 else description
            
            edge_color = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["unknown"])
            edge_width = SEVERITY_WIDTH.get(severity, 1)
            
            # Create hover tooltip
            tooltip = f"""
            <b>{interaction['drug1']} ↔ {interaction['drug2']}</b><br>
            <b>Severity:</b> {severity.upper()}<br>
            <b>Details:</b> {desc_short}
            """
            
            net.add_edge(
                interaction["drug1"],
                interaction["drug2"],
                title=tooltip,
                color=edge_color,
                width=edge_width
            )
        
        return net
    
    def generate_html(
        self,
        drugs: list[str],
        interactions: list[dict],
        title: str = "Drug Interaction Network"
    ) -> str:
        """
        Generate HTML string for embedding in Streamlit.
        
        Returns:
            HTML string containing the interactive graph
        """
        net = self.create_network(drugs, interactions, title)
        
        # Generate HTML to a temp file, then read it
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            temp_path = f.name
        
        net.save_graph(temp_path)
        
        with open(temp_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Clean up temp file
        os.unlink(temp_path)
        
        return html_content
    
    def save_graph(
        self,
        drugs: list[str],
        interactions: list[dict],
        output_path: str,
        title: str = "Drug Interaction Network"
    ):
        """
        Save graph to an HTML file.
        """
        net = self.create_network(drugs, interactions, title)
        net.save_graph(output_path)


def create_simple_graph(
    drugs: list[str],
    interactions: list[dict],
    height: str = "500px"
) -> str:
    """
    Convenience function to quickly create a graph HTML.
    
    Args:
        drugs: List of drug names
        interactions: List of dicts with drug1, drug2, severity, description
        height: Graph height
    
    Returns:
        HTML string for Streamlit embedding
    """
    visualizer = DrugInteractionGraph(height=height)
    return visualizer.generate_html(drugs, interactions)


# ==============================================================
# STANDALONE TEST
# ==============================================================

if __name__ == "__main__":
    # Test with sample data
    test_drugs = ["Lisinopril", "Metformin", "Metoprolol", "Omeprazole", "Amlodipine"]
    
    test_interactions = [
        {
            "drug1": "Lisinopril",
            "drug2": "Metformin",
            "severity": "moderate",
            "description": "Lisinopril may increase the hypoglycemic effect of Metformin."
        },
        {
            "drug1": "Lisinopril",
            "drug2": "Metoprolol",
            "severity": "major",
            "description": "Both drugs lower blood pressure, which may cause excessive hypotension."
        },
        {
            "drug1": "Metoprolol",
            "drug2": "Amlodipine",
            "severity": "moderate",
            "description": "Combined use may increase risk of bradycardia and hypotension."
        },
        {
            "drug1": "Omeprazole",
            "drug2": "Metformin",
            "severity": "minor",
            "description": "Omeprazole may slightly affect Metformin absorption."
        }
    ]
    
    # Generate and save
    visualizer = DrugInteractionGraph()
    visualizer.save_graph(
        test_drugs,
        test_interactions,
        "test_graph.html",
        "Test Drug Interaction Network"
    )
    
    print("✅ Graph saved to test_graph.html")
    print("   Open in browser to view interactive visualization")