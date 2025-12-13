"""
MedGraphRAG - Streamlit Web Interface v5
=========================================
NEW IN V5:
- 👤 Patient Profile Management
- Personalized risk assessment based on age, kidney/liver function
- Synthetic patient database (750 patients)
- Add new patients on the fly

Run: streamlit run app/main.py
"""

import streamlit as st
import streamlit.components.v1 as components
import os
import sys
import time
import json
from datetime import date, datetime
from dotenv import load_dotenv
from neo4j import GraphDatabase

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="MedGraphRAG - Drug Interactions",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DATABASE & ENGINE CONNECTIONS
# =============================================================================

@st.cache_resource
def get_neo4j_driver():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password123")
    return GraphDatabase.driver(uri, auth=(user, password))


@st.cache_resource
def get_polypharmacy_analyzer():
    from src.analysis.polypharmacy_analyzer import PolypharmacyAnalyzer
    return PolypharmacyAnalyzer(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password123"),
        llm_model="llama3.2:3b",
        ollama_base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )


@st.cache_resource
def get_text_to_cypher_engine():
    from src.retrieval.text_to_cypher import TextToCypherEngine
    return TextToCypherEngine(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password123"),
        llm_model="llama3.2:3b",
        ollama_base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )


def run_query(query: str, params: dict = None):
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(query, params or {})
        return [dict(record) for record in result]


# =============================================================================
# PATIENT DATA MANAGEMENT
# =============================================================================

PATIENTS_FILE = "data/synthetic/patients.json"


@st.cache_data
def load_patients_cached():
    """Load patients from JSON file (cached)."""
    if os.path.exists(PATIENTS_FILE):
        with open(PATIENTS_FILE, 'r') as f:
            data = json.load(f)
        return data
    return []


def load_patients():
    """Load patients from JSON (uncached for fresh reads)."""
    if os.path.exists(PATIENTS_FILE):
        with open(PATIENTS_FILE, 'r') as f:
            return json.load(f)
    return []


def save_patients(patients: list):
    """Save patients to JSON file."""
    os.makedirs(os.path.dirname(PATIENTS_FILE), exist_ok=True)
    with open(PATIENTS_FILE, 'w') as f:
        json.dump(patients, f, indent=2, default=str)
    st.cache_data.clear()  # Clear cache to reload


def get_patient_by_id(patient_id: str):
    """Get a single patient by ID."""
    from src.models.patient import PatientProfile
    patients = load_patients()
    for p in patients:
        if p["patient_id"] == patient_id:
            return PatientProfile.from_dict(p)
    return None


def add_new_patient(patient_data: dict):
    """Add a new patient to the database."""
    patients = load_patients()
    patients.append(patient_data)
    save_patients(patients)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_database_stats():
    drug_count = run_query("MATCH (d:Drug) RETURN count(d) as count")[0]['count']
    interaction_count = run_query("MATCH ()-[i:INTERACTS_WITH]->() RETURN count(i) as count")[0]['count']
    return drug_count, interaction_count


def get_severity_color(severity: str):
    return {'major': '🔴', 'moderate': '🟡', 'minor': '🟢'}.get(severity, '⚪')


def get_risk_emoji(risk_level: str):
    return {'Critical': '🚨', 'High': '🔴', 'Moderate': '🟡', 'Low': '🟢'}.get(risk_level, '❓')


def search_drug_interactions(drug_name: str):
    query = """
    MATCH (d:Drug)-[i:INTERACTS_WITH]-(other:Drug)
    WHERE toLower(d.name) CONTAINS toLower($drug_name)
    RETURN d.name as drug, other.name as interacts_with,
           i.severity as severity, i.description as description
    ORDER BY CASE i.severity WHEN 'major' THEN 1 WHEN 'moderate' THEN 2 ELSE 3 END
    LIMIT 50
    """
    return run_query(query, {"drug_name": drug_name})


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/pill.png", width=80)
    st.title("MedGraphRAG")
    st.caption("AI-Powered Personalized Drug Safety")
    
    st.divider()
    
    # Database stats
    try:
        drug_count, interaction_count = get_database_stats()
        patients_data = load_patients_cached()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Drugs", f"{drug_count:,}")
        with col2:
            st.metric("Patients", f"{len(patients_data):,}")
        
        st.metric("Interactions", f"{interaction_count:,}")
    except Exception as e:
        st.error(f"Error: {e}")
    
    st.divider()
    
    st.subheader("Severity Legend")
    st.write("🔴 Major / 🟡 Moderate / 🟢 Minor")
    
    st.divider()
    
    st.subheader("Risk Levels")
    st.write("🚨 Critical — Immediate review needed")
    st.write("🔴 High — Consult pharmacist")
    st.write("🟡 Moderate — Monitor closely")
    st.write("🟢 Low — Standard precautions")


# =============================================================================
# MAIN CONTENT
# =============================================================================

st.title("💊 MedGraphRAG")
st.markdown("**Personalized AI Drug Interaction Analysis** — Patient-specific risk assessment")

# Tab navigation - 6 TABS
tab_patient, tab_poly, tab_nl, tab1, tab2, tab3 = st.tabs([
    "👤 Patient Analysis",
    "📋 Quick Check",
    "🤖 Natural Language", 
    "🔍 Search Drug", 
    "⚡ Two Drugs", 
    "📊 Browse"
])

# =============================================================================
# TAB 0: PATIENT ANALYSIS (THE MAIN FEATURE!)
# =============================================================================
with tab_patient:
    st.subheader("👤 Patient-Centered Medication Analysis")
    st.markdown("Select an existing patient or add a new one for personalized risk assessment.")
    
    # Patient selection or creation
    patient_mode = st.radio(
        "Choose option:",
        ["Select Existing Patient", "Add New Patient"],
        horizontal=True
    )
    
    selected_patient = None
    
    if patient_mode == "Select Existing Patient":
        patients_data = load_patients()
        
        if not patients_data:
            st.warning("No patients in database. Generate synthetic data first or add a new patient.")
            st.code("python -m src.data.synthetic_patients", language="bash")
        else:
            # Search/filter patients
            search_col, filter_col = st.columns([2, 1])
            
            with search_col:
                search_term = st.text_input("🔍 Search by name or ID:", placeholder="e.g., Johnson or PAT000001")
            
            with filter_col:
                risk_filter = st.selectbox("Filter by risk:", ["All", "High Risk Only", "Elderly (65+)", "Kidney Impairment"])
            
            # Filter patients
            filtered = patients_data
            if search_term:
                search_lower = search_term.lower()
                filtered = [p for p in filtered if 
                    search_lower in p.get("first_name", "").lower() or
                    search_lower in p.get("last_name", "").lower() or
                    search_lower in p.get("patient_id", "").lower()
                ]
            
            if risk_filter == "High Risk Only":
                filtered = [p for p in filtered if len(p.get("medications", [])) >= 5]
            elif risk_filter == "Elderly (65+)":
                today = date.today()
                filtered = [p for p in filtered if 
                    (today.year - date.fromisoformat(p["date_of_birth"]).year) >= 65
                ]
            elif risk_filter == "Kidney Impairment":
                filtered = [p for p in filtered if p.get("renal_function") in ["moderate", "severe", "failure"]]
            
            # Display patient selector
            if filtered:
                # Create display options
                options = []
                for p in filtered[:100]:  # Limit to 100 for performance
                    dob = date.fromisoformat(p["date_of_birth"])
                    age = date.today().year - dob.year
                    med_count = len(p.get("medications", []))
                    options.append(f"{p['patient_id']} — {p['first_name']} {p['last_name']} ({age}yo, {med_count} meds)")
                
                selected_option = st.selectbox(
                    f"Select patient ({len(filtered)} found):",
                    options
                )
                
                if selected_option:
                    selected_id = selected_option.split(" — ")[0]
                    selected_patient = get_patient_by_id(selected_id)
            else:
                st.info("No patients match your search criteria.")
    
    else:  # Add New Patient
        st.markdown("### Add New Patient")
        
        with st.form("new_patient_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                first_name = st.text_input("First Name*", placeholder="John")
                last_name = st.text_input("Last Name*", placeholder="Smith")
                dob = st.date_input("Date of Birth*", value=date(1950, 1, 1), min_value=date(1920, 1, 1))
            
            with col2:
                sex = st.selectbox("Sex*", ["male", "female", "other"])
                weight = st.number_input("Weight (kg)*", min_value=30.0, max_value=200.0, value=70.0)
                height = st.number_input("Height (cm)*", min_value=100.0, max_value=220.0, value=170.0)
            
            with col3:
                egfr = st.number_input("eGFR (mL/min)", min_value=5.0, max_value=130.0, value=90.0)
                renal = st.selectbox("Kidney Function", ["normal", "mild", "moderate", "severe", "failure"])
                hepatic = st.selectbox("Liver Function", ["normal", "mild", "moderate", "severe"])
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                allergies_input = st.text_input("Allergies (comma-separated)", placeholder="Penicillin, Sulfa")
                conditions_input = st.text_input("Medical Conditions (comma-separated)", placeholder="Hypertension, Diabetes")
            
            with col2:
                medications_input = st.text_area(
                    "Current Medications (comma-separated)*",
                    placeholder="Lisinopril, Metformin, Metoprolol",
                    height=100
                )
            
            pregnancy = st.selectbox("Pregnancy Status", [
                "not_applicable", "not_pregnant", "pregnant_trimester_1", 
                "pregnant_trimester_2", "pregnant_trimester_3", "lactating"
            ])
            
            submitted = st.form_submit_button("➕ Add Patient & Analyze", type="primary")
            
            if submitted and first_name and last_name and medications_input:
                from src.models.patient import PatientProfile, Sex, RenalFunction, HepaticFunction, PregnancyStatus
                
                # Generate new patient ID
                patients_data = load_patients()
                new_id = f"PAT{len(patients_data) + 1:06d}"
                
                # Create patient profile
                new_patient = PatientProfile(
                    patient_id=new_id,
                    first_name=first_name,
                    last_name=last_name,
                    date_of_birth=dob,
                    sex=Sex(sex),
                    weight_kg=weight,
                    height_cm=height,
                    renal_function=RenalFunction(renal),
                    egfr=egfr,
                    hepatic_function=HepaticFunction(hepatic),
                    pregnancy_status=PregnancyStatus(pregnancy),
                    allergies=[a.strip() for a in allergies_input.split(",") if a.strip()],
                    conditions=[c.strip() for c in conditions_input.split(",") if c.strip()],
                    medications=[m.strip() for m in medications_input.split(",") if m.strip()]
                )
                
                # Save patient
                add_new_patient(new_patient.to_dict())
                st.success(f"✅ Patient {new_id} added successfully!")
                selected_patient = new_patient
    
    # =============================================================================
    # PATIENT ANALYSIS DISPLAY
    # =============================================================================
    
    if selected_patient:
        st.divider()
        
        # Patient Info Card
        st.markdown(f"## 👤 {selected_patient.full_name}")
        
        info_cols = st.columns(4)
        with info_cols[0]:
            st.metric("Age", f"{selected_patient.age} years")
        with info_cols[1]:
            st.metric("BMI", f"{selected_patient.bmi}")
        with info_cols[2]:
            st.metric("eGFR", f"{selected_patient.egfr}")
        with info_cols[3]:
            st.metric("Medications", len(selected_patient.medications))
        
        # Risk factors display
        risk_factors = selected_patient.get_risk_factors()
        if risk_factors:
            with st.expander("⚠️ Patient Risk Factors", expanded=True):
                for rf in risk_factors:
                    st.write(f"• {rf}")
        
        # Patient details
        with st.expander("📋 Patient Details"):
            detail_cols = st.columns(2)
            with detail_cols[0]:
                st.write(f"**ID:** {selected_patient.patient_id}")
                st.write(f"**Sex:** {selected_patient.sex.value}")
                st.write(f"**Weight:** {selected_patient.weight_kg} kg")
                st.write(f"**Height:** {selected_patient.height_cm} cm")
                st.write(f"**Kidney:** {selected_patient.renal_function.value}")
                st.write(f"**Liver:** {selected_patient.hepatic_function.value}")
            with detail_cols[1]:
                st.write(f"**Allergies:** {', '.join(selected_patient.allergies) or 'None'}")
                st.write(f"**Conditions:** {', '.join(selected_patient.conditions) or 'None'}")
                st.write(f"**Pregnancy Status:** {selected_patient.pregnancy_status.value}")
        
        # Medications list
        st.markdown("### 💊 Current Medications")
        med_cols = st.columns(min(4, len(selected_patient.medications) or 1))
        for i, med in enumerate(selected_patient.medications):
            with med_cols[i % 4]:
                st.info(f"**{med}**")
        
        # ANALYZE BUTTON
        st.divider()
        
        col1, col2 = st.columns([1, 3])
        with col1:
            analyze_btn = st.button("🔬 Analyze Medications", type="primary", use_container_width=True)
        with col2:
            explain_toggle = st.checkbox("Generate detailed explanations (slower)", value=True)
        
        if analyze_btn:
            try:
                analyzer = get_polypharmacy_analyzer()
                
                with st.status("Analyzing medications for this patient...", expanded=True) as status:
                    st.write(f"🔄 Checking {len(selected_patient.medications)} medications...")
                    st.write(f"👤 Applying patient-specific risk factors...")
                    start_time = time.time()
                    
                    report = analyzer.analyze_for_patient(selected_patient, generate_explanations=explain_toggle)
                    
                    elapsed = time.time() - start_time
                    status.update(label=f"✅ Analysis complete ({elapsed:.1f}s)", state="complete")
                
                # ==== RESULTS ====
                
                # Risk Score Banner
                risk_emoji = get_risk_emoji(report.risk_level)
                
                if report.risk_level == "Critical":
                    st.error(f"""
                    {risk_emoji} **CRITICAL RISK** — Adjusted Score: {report.adjusted_risk_score}/100
                    
                    Immediate pharmacist/physician review recommended.
                    """)
                elif report.risk_level == "High":
                    st.error(f"{risk_emoji} **HIGH RISK** — Adjusted Score: {report.adjusted_risk_score}/100")
                elif report.risk_level == "Moderate":
                    st.warning(f"{risk_emoji} **MODERATE RISK** — Adjusted Score: {report.adjusted_risk_score}/100")
                else:
                    st.success(f"{risk_emoji} **LOW RISK** — Adjusted Score: {report.adjusted_risk_score}/100")
                
                # Comparison: Base vs Adjusted
                st.caption(f"Base score: {report.risk_score:.0f} → Adjusted for patient factors: {report.adjusted_risk_score}")
                
                # Allergy Alerts (CRITICAL)
                if report.allergy_alerts:
                    st.error("### 🚨 ALLERGY ALERTS")
                    for alert in report.allergy_alerts:
                        st.write(alert)
                
                # Summary
                if report.summary:
                    st.info(f"**💬 Personalized Summary:**\n\n{report.summary}")
                
                # Personalized Warnings
                if report.personalized_warnings:
                    with st.expander("⚠️ Personalized Warnings", expanded=True):
                        for warning in report.personalized_warnings:
                            st.write(warning)
                
                # Stats Row
                stat_cols = st.columns(4)
                with stat_cols[0]:
                    st.metric("Total Interactions", report.total_interactions)
                with stat_cols[1]:
                    st.metric("🔴 Major", report.major_count)
                with stat_cols[2]:
                    st.metric("🟡 Moderate", report.moderate_count)
                with stat_cols[3]:
                    st.metric("🟢 Minor", report.minor_count)
                
                # Graph Visualization
                if report.interactions:
                    st.subheader("🕸️ Interaction Network")
                    
                    from src.visualization.graph_builder import create_simple_graph
                    
                    drug_names = [d.name for d in report.drugs_analyzed if d.found_in_db]
                    interactions_data = [
                        {"drug1": i.drug1, "drug2": i.drug2, "severity": i.severity, "description": i.description}
                        for i in report.interactions
                    ]
                    
                    graph_html = create_simple_graph(drug_names, interactions_data, height="400px")
                    components.html(graph_html, height=420, scrolling=True)
                
                # Interaction Details
                if report.interactions:
                    st.subheader("📋 Interaction Details")
                    
                    sorted_interactions = sorted(
                        report.interactions,
                        key=lambda x: {"major": 0, "moderate": 1, "minor": 2}.get(x.severity, 3)
                    )
                    
                    for interaction in sorted_interactions:
                        icon = get_severity_color(interaction.severity)
                        with st.expander(
                            f"{icon} {interaction.drug1} ↔ {interaction.drug2} — {interaction.severity.upper()}",
                            expanded=(interaction.severity == "major")
                        ):
                            st.write(f"**Technical:** {interaction.description}")
                            if hasattr(interaction, 'layman_explanation') and interaction.layman_explanation:
                                st.write(f"**In simple terms:** {interaction.layman_explanation}")
                
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
                st.caption("Make sure Ollama is running: `ollama serve`")

# =============================================================================
# TAB 1: QUICK CHECK (No patient context)
# =============================================================================
with tab_poly:
    st.subheader("📋 Quick Medication Check")
    st.markdown("Fast interaction check without patient profile.")
    
    meds_input = st.text_area(
        "Enter medications (comma-separated):",
        placeholder="Lisinopril, Metformin, Metoprolol, Omeprazole",
        height=80
    )
    
    if st.button("🔍 Quick Check", type="primary"):
        if meds_input:
            meds_list = [m.strip() for m in meds_input.split(",") if m.strip()]
            if len(meds_list) >= 2:
                analyzer = get_polypharmacy_analyzer()
                report = analyzer.analyze(meds_list, generate_explanations=False)
                
                risk_emoji = get_risk_emoji(report.risk_level)
                st.metric("Risk Score", f"{report.risk_score:.0f}/100", delta=report.risk_level)
                
                cols = st.columns(3)
                with cols[0]:
                    st.metric("🔴 Major", report.major_count)
                with cols[1]:
                    st.metric("🟡 Moderate", report.moderate_count)
                with cols[2]:
                    st.metric("🟢 Minor", report.minor_count)
            else:
                st.warning("Enter at least 2 medications.")

# =============================================================================
# TAB 2: NATURAL LANGUAGE (ENHANCED WITH SMART ROUTING)
# =============================================================================
with tab_nl:
    st.subheader("🤖 Ask Medication Questions")
    st.markdown("Ask any question about medications — interactions, side effects, dosing, conditions, and more!")
    
    # Example questions organized by category
    with st.expander("💡 Example Questions", expanded=False):
        example_cols = st.columns(3)
        
        example_categories = {
            "Drug Interactions": [
                "Does Lisinopril interact with Metoprolol?",
                "What drugs interact with Metformin?",
            ],
            "Condition Safety": [
                "What should a diabetic patient avoid?",
                "Which drugs are risky for kidney disease?",
            ],
            "Drug Info & Timing": [
                "What is Metformin used for?",
                "When should I take Lisinopril?",
                "What are the side effects of Metoprolol?",
            ],
        }
        
        for i, (category, questions) in enumerate(example_categories.items()):
            with example_cols[i]:
                st.markdown(f"**{category}**")
                for q in questions:
                    if st.button(q, key=f"ex_{q[:20]}", use_container_width=True):
                        st.session_state.nl_question = q
    
    # Question input
    question = st.text_area(
        "Your question:",
        value=st.session_state.get('nl_question', ''),
        placeholder="e.g., What medications should someone with diabetes avoid?",
        height=80,
        key="nl_input_main"
    )
    
    # Clear the session state
    if 'nl_question' in st.session_state:
        del st.session_state.nl_question
    
    # Patient context option
    col1, col2 = st.columns([1, 2])
    with col1:
        ask_btn = st.button("🚀 Ask", type="primary", use_container_width=True)
    with col2:
        use_patient_context = st.checkbox(
            "Use selected patient's context (if available)", 
            value=False,
            help="If you've selected a patient in the Patient Analysis tab, their info will be used for personalized answers"
        )
    
    if ask_btn and question:
        try:
            # Get the smart router
            from src.retrieval.smart_query_router import SmartQueryRouter, EXAMPLE_QUESTIONS
            
            cypher_engine = get_text_to_cypher_engine()
            router = SmartQueryRouter(
                text_to_cypher_engine=cypher_engine,
                llm_model="llama3.2:3b",
                ollama_base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434")
            )
            
            # Get patient context if requested
            patient_context = None
            if use_patient_context and 'selected_patient_id' in st.session_state:
                patient_context = get_patient_by_id(st.session_state.selected_patient_id)
            
            with st.spinner("🔍 Analyzing your question..."):
                result = router.route_query(question, patient=patient_context)
            
            # Display query classification
            query_type_labels = {
                "drug_interaction": "💊 Drug Interaction Query",
                "drug_info": "ℹ️ Drug Information",
                "condition_warning": "⚠️ Condition-Based Warning",
                "timing_admin": "⏰ Timing & Administration",
                "side_effects": "🩺 Side Effects",
                "alternatives": "🔄 Alternatives",
                "patient_specific": "👤 Patient-Specific",
                "general_medical": "📋 General Medical",
            }
            
            source_labels = {
                "graph": "📊 Knowledge Graph (Neo4j)",
                "llm": "🤖 AI Medical Knowledge",
                "hybrid": "🔀 Patient Context + AI",
                "system": "⚙️ System",
            }
            
            # Header with classification
            st.markdown(f"### {query_type_labels.get(result.query_type.value, '❓ Query')}")
            st.caption(f"Source: {source_labels.get(result.source, result.source)} | Confidence: {result.confidence}")
            
            # Show Cypher query if used
            if result.cypher_query:
                with st.expander("📝 Generated Cypher Query", expanded=False):
                    st.code(result.cypher_query, language="cypher")
            
            # Main answer
            st.markdown("---")
            st.markdown(result.answer)
            
            # Disclaimer
            if result.disclaimer:
                st.markdown("---")
                st.caption("⚠️ **Disclaimer:** This information is for educational purposes only. Always consult a healthcare professional for medical advice.")
        
        except ImportError:
            st.error("Smart Query Router not found. Please ensure `src/retrieval/smart_query_router.py` exists.")
            st.code("# Create the file from the artifact provided", language="bash")
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.caption("Make sure Ollama is running: `ollama serve`")

# =============================================================================
# TAB 3: SEARCH DRUG
# =============================================================================
with tab1:
    st.subheader("🔍 Search Drug Interactions")
    
    drug_name = st.text_input("Drug name:", placeholder="Lisinopril")
    
    if st.button("Search", key="search_single"):
        if drug_name:
            results = search_drug_interactions(drug_name)
            if results:
                st.success(f"Found {len(results)} interactions")
                for r in results[:20]:
                    icon = get_severity_color(r['severity'])
                    with st.expander(f"{icon} {r['drug']} ↔ {r['interacts_with']}"):
                        st.write(r['description'] or "No description")
            else:
                st.warning("No interactions found.")

# =============================================================================
# TAB 4: TWO DRUGS
# =============================================================================
with tab2:
    st.subheader("⚡ Check Two Drugs")
    
    col1, col2 = st.columns(2)
    with col1:
        d1 = st.text_input("First drug:", placeholder="Lisinopril")
    with col2:
        d2 = st.text_input("Second drug:", placeholder="Metoprolol")
    
    if st.button("Check", key="check_two"):
        if d1 and d2:
            query = """
            MATCH (d1:Drug)-[i:INTERACTS_WITH]-(d2:Drug)
            WHERE toLower(d1.name) CONTAINS toLower($d1)
              AND toLower(d2.name) CONTAINS toLower($d2)
            RETURN d1.name, d2.name, i.severity, i.description LIMIT 1
            """
            results = run_query(query, {"d1": d1, "d2": d2})
            if results:
                r = results[0]
                icon = get_severity_color(r['i.severity'])
                st.markdown(f"### {icon} {r['i.severity'].upper()}")
                st.write(r['i.description'])
            else:
                st.success("✅ No known interaction found.")

# =============================================================================
# TAB 5: BROWSE
# =============================================================================
with tab3:
    st.subheader("📊 Database Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Sample Drugs:**")
        samples = run_query("MATCH (d:Drug) RETURN d.name LIMIT 15")
        for s in samples:
            st.write(f"• {s['d.name']}")
    
    with col2:
        st.markdown("**Top by Interactions:**")
        top = run_query("""
            MATCH (d:Drug)-[i:INTERACTS_WITH]-()
            RETURN d.name, count(i) as c ORDER BY c DESC LIMIT 10
        """)
        for t in top:
            st.write(f"• {t['d.name']}: {t['c']}")

# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    <p>🎓 <strong>Northeastern University</strong> - Generative AI Capstone</p>
    <p>Built with Neo4j, LangChain, Ollama, PyVis, Streamlit</p>
    <p>⚠️ Educational purposes only. Consult healthcare professionals.</p>
</div>
""", unsafe_allow_html=True)