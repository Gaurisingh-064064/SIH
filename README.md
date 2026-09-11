README.md


🕵️ NyayaNet
AI-Powered Criminal Network Analysis System
Smart India Hackathon 2026 --- Problem Statement 26189

NyayaNet is an AI-assisted investigative intelligence platform designed
to help authorized investigators transform fragmented evidence into
structured entities, candidate relationships, suspicious activity
signals, and interactive criminal network insights.

The platform combines Natural Language Processing, Machine Learning,
Graph Analytics, Evidence Integrity Verification, and Interactive
Visualization to support faster and more explainable investigations.

✨ Overview
Modern investigations often involve information distributed across
multiple sources such as FIRs, police reports, call detail records,
financial transactions, surveillance reports, social-media intelligence,
and criminal-history records.

Analyzing these sources manually makes it difficult to identify:

Hidden relationships between individuals

Cross-source connections

Repeated suspicious activities

Important or influential entities in a network

Time-based patterns and event sequences

Potential investigative leads

NyayaNet addresses this problem by creating an investigation-centric
intelligence workspace where multiple evidence sources can be ingested,
analyzed, connected, and visualized.

⚠️ NyayaNet is designed as an investigative decision-support and
lead-generation prototype. Its outputs do not establish guilt or
identify someone as a criminal.

🎯 Key Features
🔐 Secure Investigator Workspace
Authorized investigator authentication

Investigation-specific workspaces

Create and manage investigations

Ongoing and closed investigation states

Ownership checks for investigation access

Investigation-scoped analysis and graph data

📂 Multi-Source Intelligence Ingestion
NyayaNet supports intelligence from multiple sources:

Source Supported Intelligence

📄 FIR / Police Complaint FIR narratives, complaint text,
witness statements

🛡️ Police Reports Investigation notes, reports and
case records

📞 Call Detail Records Calls, frequency, duration and
communication patterns

💰 Financial Transactions Transaction activity and financial
links

📹 Surveillance Reports Observations, movement and location
information

🌐 Social Media Intelligence Posts, messages, mentions and
interactions

PDF Support
Investigators can upload PDF evidence. The system extracts readable text
from supported PDFs and uses the extracted content for intelligence
analysis.

🧠 AI & Analytics Pipeline
                    INVESTIGATION
                         │
                         ▼
               MULTI-SOURCE EVIDENCE
                         │
                         ▼
                 PDF / TEXT INGESTION
                         │
                         ▼
                EVIDENCE HASHING
                         │
                         ▼
              NLP ENTITY EXTRACTION
                         │
       ┌─────────────────┼─────────────────┐
       ▼                 ▼                 ▼
    PERSON            LOCATION          ORGANIZATION
    PHONE             VEHICLE           EMAIL / BANK
       │                 │                 │
       └─────────────────┼─────────────────┘
                         ▼
            ENTITY NORMALIZATION
                         │
                         ▼
          CANDIDATE RELATIONSHIP
               GENERATION
                         │
                         ▼
          MACHINE LEARNING SCORING
                         │
                         ▼
        SUSPICIOUS ACTIVITY ANALYSIS
                         │
                         ▼
             NETWORK GRAPH ANALYTICS
                         │
       ┌─────────────────┼─────────────────┐
       ▼                 ▼                 ▼
   RELATIONSHIPS     INFLUENCE       INVESTIGATIVE
                    ANALYSIS            INSIGHTS
🔍 Entity Extraction
NyayaNet extracts and organizes intelligence entities from evidence.

Supported Entity Types
👤 Persons

🏢 Organizations

📍 Locations

📞 Phone Numbers

🚗 Vehicles

📧 Email Addresses

🏦 Bank Identifiers

The backend uses spaCy NLP, regex-based extraction, and conservative
fallback heuristics.

🔗 Candidate Relationship Analysis
When entities appear across evidence, NyayaNet generates candidate
relationships for investigator review.

The relationship model evaluates observable activity features such as:

Phone call frequency

Total call duration

Transaction count

Aggregate transaction value

Meeting count

Evidence and activity co-occurrence

Source diversity

Shared phone signals

Shared vehicle signals

Shared organization signals

Shared location signals

The result is an investigative confidence score, not a declaration
of guilt.

🚨 Suspicious Activity Detection
NyayaNet identifies unusual or suspicious combinations of observable
activities.

Examples include:

Repeated communication between connected entities

Frequent transactions between linked individuals

High aggregate transaction activity

Repeated meetings

Cross-source activity correlation

Unusual combinations of communication, movement and financial
activity

These signals are presented as investigative leads requiring human
review.

⏱️ Temporal Pattern Analysis
Temporal analysis helps investigators understand when and in what
sequence events occur.

NyayaNet can support analysis of:

Event timing

Activity frequency

Repeated behavior

Event sequences

Communication before or after transactions

Time correlation between activities

Patterns occurring across multiple intelligence sources

Example
11:20 PM  → Person A calls Person B
11:45 PM  → Financial transaction recorded
12:10 AM  → Vehicle movement observed
Next Day  → Both entities appear in a surveillance report
When events from different sources occur in meaningful sequences or
close time windows, they can form a candidate temporal pattern for
investigator review.

🕸️ Criminal Network Graph
NyayaNet transforms extracted intelligence into an interactive network
representation.

Graph Capabilities
Visual entity nodes

Relationship links

Candidate relationship confidence

Person search

Connected network inspection

Influential entity analysis

Evidence-backed relationship explanations

Network Analytics
The platform uses NetworkX for structural graph analysis, including:

Degree Centrality

Betweenness Centrality

PageRank

These metrics indicate structural importance within the observed
network. They are not criminality scores.

🔒 Evidence Integrity & Blockchain Verification
NyayaNet includes an evidence-integrity layer to make changes
detectable.

Current Prototype Features
SHA-256 evidence hashing

Hash-linked blockchain records

Investigation-specific evidence records

Blockchain chain verification

Evidence modification detection

Evidence
   │
   ▼
SHA-256 Hash
   │
   ▼
Private Blockchain Record
   │
   ▼
Previous Block Hash
   │
   ▼
Integrity Verification
Important
The current blockchain implementation is a prototype integrity
mechanism. A production law-enforcement deployment would require
stronger evidence-preservation infrastructure such as append-only or
WORM storage, independent audit copies, and appropriate operational
controls.

🏗️ System Architecture
┌──────────────────────────────────────────────┐
│              React + Vite Frontend           │
│                                              │
│  Dashboard • Sources • Analysis • Graph      │
└───────────────────────┬──────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────┐
│                 FastAPI Backend              │
│                                              │
│ Authentication • Evidence • Analysis APIs    │
└───────────────────────┬──────────────────────┘
                        │
       ┌────────────────┼─────────────────┐
       ▼                ▼                 ▼
┌─────────────┐  ┌──────────────┐  ┌─────────────┐
│ Supabase    │  │ NLP Engine   │  │ ML Models   │
│ Auth / DB   │  │ spaCy/Regex  │  │ sklearn     │
└─────────────┘  └──────────────┘  └─────────────┘
       │                │                 │
       └────────────────┼─────────────────┘
                        ▼
                 NetworkX Analytics
                        │
                        ▼
             Interactive Network Graph
🛠️ Technology Stack
Frontend
React 19

Vite

React Force Graph

React Force Graph 2D

PDF.js

Supabase JavaScript Client

Backend
Python

FastAPI

Uvicorn

Pydantic

AI / Machine Learning
spaCy

scikit-learn

pandas

NumPy

joblib

Graph Analytics
NetworkX

Database & Authentication
Supabase

PostgreSQL

Row Level Security architecture

📁 Project Structure
SIH/
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   ├── styles.css
│   │   └── lib/
│   │       └── supabase.js
│   ├── package.json
│   └── .env
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── blockchain.py
│   │   ├── nlp.py
│   │   ├── security.py
│   │   ├── config.py
│   │   └── supabase_client.py
│   │
│   ├── scripts/
│   ├── data/
│   ├── requirements.txt
│   └── .env
│
├── ml/
│   ├── relationship_model.joblib
│   ├── suspicious_pattern_model.joblib
│   ├── train_relationship_model.py
│   ├── train_best_relationship_model.py
│   └── evaluation utilities
│
├── db/
│
├── docs/
│   └── IMPLEMENTATION_PLAN.md
│
└── README.md
🚀 Getting Started
Prerequisites
Make sure you have:

Python 3.10+

Node.js 18+

npm

A Supabase project

spaCy English model

⚙️ Backend Setup
1. Navigate to the backend
cd backend
2. Create a virtual environment
python -m venv .venv
3. Activate the environment
macOS / Linux

source .venv/bin/activate
Windows

.venv\Scripts\activate
4. Install dependencies
pip install -r requirements.txt
5. Install the spaCy model
python -m spacy download en_core_web_sm
6. Configure environment variables
Create backend/.env using the provided example.

Example:

SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_publishable_key
SUPABASE_SERVICE_ROLE_KEY=your_server_only_service_role_key
🔒 Never expose the Supabase service-role key in the frontend.

7. Start the backend
python -m uvicorn app.main:app --reload --port 8080
Backend:

http://127.0.0.1:8080
Health Check:

http://127.0.0.1:8080/health
API Documentation:

http://127.0.0.1:8080/docs
💻 Frontend Setup
1. Navigate to the frontend
cd frontend
2. Install dependencies
npm install
3. Configure environment variables
Create frontend/.env:

VITE_API_URL=http://localhost:8080
4. Start the frontend
npm run dev
Open:

http://localhost:5173
🤖 Machine Learning
NyayaNet includes a trained relationship analysis model for scoring
candidate relationships.

The project also includes utilities for:

Generating realistic synthetic datasets

Training relationship models

Evaluating model performance

Comparing relationship models

Checking datasets

Training
From the project root:

cd backend
python scripts/train_relationship.py
⚠️ Validation metrics generated using the included synthetic dataset
must not be presented as real-world law-enforcement performance.

🧪 Synthetic Dataset
The demonstration dataset is designed for prototype development and
testing.

It includes synthetic:

Indian identities

NCR-focused locations

Phone information

Vehicle information

Organization information

Masked bank identifiers

FIR-related metadata

Communication activity

Transaction activity

Meeting activity

Candidate relationship references

The dataset is synthetic and intended for demonstration purposes.

🔌 Key API Capabilities
Capability Endpoint

Health Check GET /health
List Investigations GET /api/investigations
Create Investigation POST /api/investigations
Close Investigation POST /api/investigations/{id}/close
Investigation Workspace GET /api/investigations/{id}/workspace
Get Sources GET /api/investigations/{id}/sources
Upload Source PDF POST /api/investigations/{id}/sources/upload
Save Sources PUT /api/investigations/{id}/sources
Analyze Sources POST /api/investigations/{id}/analyze-sources
Get Analysis GET /api/investigations/{id}/analysis
Extract NLP Entities POST /api/nlp/extract
Get Persons GET /api/investigations/{id}/persons
Get Relationships GET /api/investigations/{id}/relationships
Get Graph GET /api/investigations/{id}/graph
Search Person GET /api/investigations/{id}/persons/search
Inspect Network GET /api/investigations/{id}/network/{person_id}
Verify Blockchain GET /api/investigations/{id}/blockchain/verify

🎬 Demo Workflow
A typical demonstration flow is:

1️⃣ Investigator Authentication
The authorized investigator logs into the NyayaNet workspace.

2️⃣ Create Investigation
Create a new investigation with:

Investigation title

Case description

3️⃣ Add Intelligence Sources
Upload or provide intelligence from one or more available sources.

4️⃣ Evidence Integrity
The evidence content is hashed and recorded for integrity verification.

5️⃣ Run AI Analysis
NyayaNet processes the available intelligence and extracts entities.

6️⃣ Review Relationships
Candidate relationships are generated and scored.

7️⃣ Review Suspicious Signals
Potentially unusual activity patterns are surfaced for investigator
review.

8️⃣ Explore the Network
Use the interactive graph to inspect:

People

Connections

Relationship confidence

Network influence

9️⃣ Verify Evidence Integrity
Run blockchain verification to check the recorded evidence chain.

🔐 Security Principles
NyayaNet follows several important security principles:

Authentication before investigation access

Authorization checks for protected investigations

Investigation ownership verification

Server-side service credentials

Evidence hashing

Hash-linked integrity records

Investigation-scoped data

Explainable analytical outputs

⚖️ Responsible AI & Limitations
NyayaNet is an investigative assistance system, not an autonomous
decision-making system.

Important Limitations
AI outputs require investigator review

Candidate relationship scores do not prove relationships

Network centrality does not indicate criminality

Suspicious activity signals are investigative leads

Free-form entity extraction can contain errors

NLP quality depends on language and source quality

English extraction is currently the strongest baseline

Hindi and Punjabi are preserved with language metadata, while
dedicated Indic-language NLP models remain a future improvement

The included ML dataset is synthetic

🔮 Future Enhancements
Advanced multilingual Indic-language NLP

OCR for scanned PDFs

Stronger entity resolution and deduplication

Real-time intelligence ingestion

Advanced temporal analytics

Improved anomaly detection

Geographic intelligence and map visualization

Role-based investigation collaboration

Production-grade immutable evidence storage

Enhanced explainability for ML outputs

🏆 Smart India Hackathon
Smart India Hackathon 2026

Problem Statement
PS 26189 --- AI-Powered Criminal Network Analysis System

NyayaNet is developed as a prototype to demonstrate how AI, NLP, Machine
Learning, evidence integrity mechanisms, and graph analytics can support
authorized investigators in discovering meaningful connections from
fragmented intelligence.

📌 Disclaimer
This project is a prototype developed for educational and hackathon
purposes.

It is not intended to:

Replace investigators

Determine guilt

Automatically classify individuals as criminals

Make legal or enforcement decisions without human review

All analytical outputs should be treated as investigative leads and
reviewed using appropriate legal, procedural, and ethical safeguards.

::: {align="center"}

🕵️ NyayaNet
Turning fragmented intelligence into explainable investigative
insights.

Built for Smart India Hackathon 2026 🇮🇳
:::