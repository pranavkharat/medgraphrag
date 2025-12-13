"""
DrugBank CSV Parser and Neo4j Ingestion Pipeline
=================================================
Modified version that works with Kaggle CSV datasets instead of XML.

Supported datasets:
1. Drug-Drug Interactions (Kaggle): drug_interaction.csv, drug_information.csv
2. Any CSV with columns: drug1, drug2, mechanism/description

Setup:
1. Download from Kaggle (see instructions below)
2. Place CSV files in data/raw/
3. Run: python -m src.ingestion.csv_parser
"""

import pandas as pd
from pathlib import Path
import logging
from neo4j import GraphDatabase
from tqdm import tqdm
import os
from dotenv import load_dotenv
import re

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CSVDrugParser:
    """
    Parses drug interaction data from CSV files.
    Works with multiple CSV formats from Kaggle.
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
    
    def load_drug_interactions_dataset(self):
        """
        Load the 'Drug-Drug Interactions' dataset from Kaggle.
        Expected files:
        - drug_interaction.csv (columns: drug1, drug2, type/mechanism)
        - drug_information.csv (columns: drugbank_id, name, smiles, etc.)
        
        Returns: (drugs_df, interactions_df)
        """
        # Try to find interaction file
        interaction_files = [
            "drug_interaction.csv",
            "drug_interactions.csv", 
            "DDI_data.csv",
            "interactions.csv"
        ]
        
        interactions_df = None
        for filename in interaction_files:
            filepath = self.data_dir / filename
            if filepath.exists():
                logger.info(f"Found interaction file: {filename}")
                interactions_df = pd.read_csv(filepath)
                break
        
        if interactions_df is None:
            raise FileNotFoundError(
                f"No interaction file found in {self.data_dir}. "
                f"Expected one of: {interaction_files}"
            )
        
        # Try to find drug information file
        drug_files = [
            "drug_information.csv",
            "drug_information_1258.csv",
            "drugs.csv",
            "drug_info.csv"
        ]
        
        drugs_df = None
        for filename in drug_files:
            filepath = self.data_dir / filename
            if filepath.exists():
                logger.info(f"Found drug info file: {filename}")
                drugs_df = pd.read_csv(filepath)
                break
        
        # Standardize column names
        interactions_df = self._standardize_interaction_columns(interactions_df)
        if drugs_df is not None:
            drugs_df = self._standardize_drug_columns(drugs_df)
        
        logger.info(f"Loaded {len(interactions_df)} interactions")
        if drugs_df is not None:
            logger.info(f"Loaded {len(drugs_df)} drugs")
        
        return drugs_df, interactions_df
    
    def _standardize_interaction_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map various column names to standard format."""
        column_mapping = {
            # Drug 1 name variations
            'Drug 1': 'drug1_name',        # <-- Your file's format
            'Drug_A': 'drug1_name',
            'drug_1': 'drug1_name',
            'Drug1': 'drug1_name',
            'Drug1_name': 'drug1_name',
            'drugA': 'drug1_name',
            'drug1': 'drug1_name',
            
            # Drug 2 name variations
            'Drug 2': 'drug2_name',        # <-- Your file's format
            'Drug_B': 'drug2_name',
            'drug_2': 'drug2_name',
            'Drug2': 'drug2_name',
            'Drug2_name': 'drug2_name',
            'drugB': 'drug2_name',
            'drug2': 'drug2_name',
            
            # Drug 1 ID variations
            'Drug1_id': 'drug1_id',
            'drugbank_id_1': 'drug1_id',
            
            # Drug 2 ID variations  
            'Drug2_id': 'drug2_id',
            'drugbank_id_2': 'drug2_id',
            
            # Interaction description variations
            'Interaction Description': 'description',  # <-- Your file's format
            'Interaction': 'description',
            'interaction': 'description',
            'mechanism': 'description',
            'Mechanism': 'description',
            'type': 'mechanism_type',
            'Type': 'mechanism_type',
            'interaction_type': 'mechanism_type',
            
            # Action variations
            'Action': 'action',
            'action': 'action',
            'effect': 'action'
        }
        
        df = df.rename(columns=column_mapping)
        return df
    
    def _standardize_drug_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map drug info column names to standard format."""
        column_mapping = {
            'DrugBank ID': 'drugbank_id',
            'drugbank_id': 'drugbank_id',
            'DB_ID': 'drugbank_id',
            'id': 'drugbank_id',
            
            'Name': 'name',
            'name': 'name',
            'drug_name': 'name',
            'Drug_Name': 'name',
            
            'SMILES': 'smiles',
            'smiles': 'smiles',
            
            'Description': 'description',
            'description': 'description'
        }
        
        df = df.rename(columns=column_mapping)
        return df
    
    def classify_severity(self, description: str, action: str = None) -> str:
        """Classify interaction severity based on text."""
        if pd.isna(description):
            return "unknown"
        
        desc_lower = str(description).lower()
        action_lower = str(action).lower() if action else ""
        
        # Major severity indicators
        major_keywords = [
            "contraindicated", "avoid", "serious", "severe", "dangerous",
            "life-threatening", "fatal", "death", "bleeding", "hemorrhage",
            "cardiac arrest", "seizure", "respiratory depression",
            "serotonin syndrome", "neuroleptic malignant"
        ]
        
        # Moderate severity indicators
        moderate_keywords = [
            "monitor", "caution", "adjust dose", "may increase", "may decrease",
            "enhanced", "reduced", "clinical significance", "risk of",
            "therapeutic efficacy", "serum concentration"
        ]
        
        for kw in major_keywords:
            if kw in desc_lower:
                return "major"
        
        for kw in moderate_keywords:
            if kw in desc_lower or kw in action_lower:
                return "moderate"
        
        return "minor"


class Neo4jCSVLoader:
    """Loads parsed CSV data into Neo4j."""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Connected to Neo4j at {uri}")
    
    def close(self):
        self.driver.close()
    
    def clear_database(self):
        """Clear all existing data (use with caution!)."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Cleared existing database")
    
    def create_constraints(self):
        """Create database constraints and indexes."""
        constraints = [
            "CREATE CONSTRAINT drug_name IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE",
            "CREATE INDEX drug_name_index IF NOT EXISTS FOR (d:Drug) ON (d.name)",
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.debug(f"Constraint may exist: {e}")
        
        # Create full-text index for drug name search
        try:
            with self.driver.session() as session:
                session.run("""
                    CREATE FULLTEXT INDEX drug_search IF NOT EXISTS
                    FOR (d:Drug) ON EACH [d.name, d.synonyms]
                """)
        except Exception as e:
            logger.debug(f"Full-text index may exist: {e}")
        
        logger.info("Created constraints and indexes")
    
    def load_drugs_from_interactions(self, interactions_df: pd.DataFrame):
        """
        Extract unique drugs from interaction data and create nodes.
        Use this when you don't have a separate drug info file.
        """
        # Get all unique drug names
        drug1_names = interactions_df['drug1_name'].dropna().unique()
        drug2_names = interactions_df['drug2_name'].dropna().unique()
        all_drugs = set(drug1_names) | set(drug2_names)
        
        logger.info(f"Creating {len(all_drugs)} drug nodes...")
        
        with self.driver.session() as session:
            for drug_name in tqdm(all_drugs, desc="Creating drugs"):
                if not drug_name or pd.isna(drug_name):
                    continue
                    
                # Clean the drug name
                clean_name = str(drug_name).strip()
                
                session.run("""
                    MERGE (d:Drug {name: $name})
                    SET d.name_lower = toLower($name)
                """, name=clean_name)
        
        logger.info(f"Created {len(all_drugs)} drug nodes")
    
    def load_drugs_from_info(self, drugs_df: pd.DataFrame):
        """Load drugs from a separate drug information file."""
        logger.info(f"Loading {len(drugs_df)} drugs...")
        
        with self.driver.session() as session:
            for _, row in tqdm(drugs_df.iterrows(), total=len(drugs_df), desc="Loading drugs"):
                name = row.get('name')
                if not name or pd.isna(name):
                    continue
                
                drugbank_id = row.get('drugbank_id', '')
                description = row.get('description', '')
                smiles = row.get('smiles', '')
                
                session.run("""
                    MERGE (d:Drug {name: $name})
                    SET d.drugbank_id = $drugbank_id,
                        d.description = $description,
                        d.smiles = $smiles,
                        d.name_lower = toLower($name)
                """, 
                    name=str(name).strip(),
                    drugbank_id=str(drugbank_id) if not pd.isna(drugbank_id) else "",
                    description=str(description)[:2000] if not pd.isna(description) else "",
                    smiles=str(smiles) if not pd.isna(smiles) else ""
                )
        
        logger.info("Drug loading complete")
    
    def load_interactions(self, interactions_df: pd.DataFrame, parser: CSVDrugParser):
        """Load drug-drug interactions."""
        logger.info(f"Loading {len(interactions_df)} interactions...")
        
        success_count = 0
        error_count = 0
        
        with self.driver.session() as session:
            for _, row in tqdm(interactions_df.iterrows(), total=len(interactions_df), desc="Loading interactions"):
                try:
                    drug1 = row.get('drug1_name')
                    drug2 = row.get('drug2_name')
                    
                    if not drug1 or not drug2 or pd.isna(drug1) or pd.isna(drug2):
                        continue
                    
                    # Get description/mechanism
                    description = row.get('description', '')
                    if pd.isna(description):
                        description = row.get('mechanism_type', '')
                    
                    # Get action (increase/decrease)
                    action = row.get('action', '')
                    
                    # Classify severity
                    severity = parser.classify_severity(description, action)
                    
                    # Create interaction relationship
                    session.run("""
                        MATCH (d1:Drug {name: $drug1})
                        MATCH (d2:Drug {name: $drug2})
                        MERGE (d1)-[i:INTERACTS_WITH]->(d2)
                        SET i.description = $description,
                            i.severity = $severity,
                            i.action = $action,
                            i.source = 'DrugBank'
                    """,
                        drug1=str(drug1).strip(),
                        drug2=str(drug2).strip(),
                        description=str(description)[:500] if not pd.isna(description) else "",
                        severity=severity,
                        action=str(action) if not pd.isna(action) else ""
                    )
                    success_count += 1
                    
                except Exception as e:
                    error_count += 1
                    if error_count < 5:
                        logger.warning(f"Error loading interaction: {e}")
        
        logger.info(f"Loaded {success_count} interactions ({error_count} errors)")
    
    def get_stats(self) -> dict:
        """Get graph statistics."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Drug) WITH count(d) as drugs
                MATCH ()-[i:INTERACTS_WITH]->() WITH drugs, count(i) as interactions
                RETURN drugs, interactions
            """)
            record = result.single()
            return {
                "drugs": record["drugs"],
                "interactions": record["interactions"]
            }


def download_instructions():
    """Print instructions for downloading data from Kaggle."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           HOW TO DOWNLOAD DATA FROM KAGGLE                       ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Option 1: Web Download (Easiest)                                ║
║  ─────────────────────────────────                               ║
║  1. Go to: https://www.kaggle.com/datasets/mghobashy/            ║
║            drug-drug-interactions                                 ║
║  2. Click "Download" button (you need a free Kaggle account)     ║
║  3. Unzip the downloaded file                                    ║
║  4. Move CSV files to: medgraphrag/data/raw/                     ║
║                                                                  ║
║  Option 2: Kaggle CLI                                            ║
║  ────────────────────                                            ║
║  pip install kaggle                                              ║
║  kaggle datasets download -d mghobashy/drug-drug-interactions    ║
║  unzip drug-drug-interactions.zip -d data/raw/                   ║
║                                                                  ║
║  Expected files after download:                                  ║
║  • data/raw/drug_interaction.csv                                 ║
║  • data/raw/drug_information_1258.csv (optional)                 ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)


def main():
    """Main entry point for CSV data ingestion."""
    # Print download instructions
    download_instructions()
    
    # Configuration
    data_dir = os.getenv("DATA_DIR")
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    # Check if data files exist
    data_path = Path(data_dir)
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
        logger.error(f"Created {data_dir} folder. Please add CSV files and run again.")
        return
    
    csv_files = list(data_path.glob("*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {data_dir}. Please download data first.")
        download_instructions()
        return
    
    logger.info(f"Found CSV files: {[f.name for f in csv_files]}")
    
    # Parse CSV files
    parser = CSVDrugParser(data_dir)
    
    try:
        drugs_df, interactions_df = parser.load_drug_interactions_dataset()
    except FileNotFoundError as e:
        logger.error(str(e))
        download_instructions()
        return
    
    # Load into Neo4j
    loader = Neo4jCSVLoader(neo4j_uri, neo4j_user, neo4j_password)
    
    try:
        # Create constraints first
        loader.create_constraints()
        
        # Load drugs
        if drugs_df is not None and not drugs_df.empty:
            loader.load_drugs_from_info(drugs_df)
        else:
            # Extract drugs from interaction data
            loader.load_drugs_from_interactions(interactions_df)
        
        # Load interactions
        loader.load_interactions(interactions_df, parser)
        
        # Print stats
        stats = loader.get_stats()
        print(f"\n{'='*50}")
        print("✅ DATA LOADING COMPLETE!")
        print(f"{'='*50}")
        print(f"📊 Drugs loaded:        {stats['drugs']:,}")
        print(f"🔗 Interactions loaded: {stats['interactions']:,}")
        print(f"{'='*50}")
        print("\nNext steps:")
        print("1. Open Neo4j Browser: http://localhost:7474")
        print("2. Run: MATCH (d:Drug) RETURN d LIMIT 10")
        print("3. Test the app: streamlit run app/main.py")
        
    finally:
        loader.close()


if __name__ == "__main__":
    main() 