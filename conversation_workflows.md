# Conversation Workflow Documentation

This document contains detailed conversation workflow diagrams for the Voice Agent System. It maps the end-to-end paths, user actions, decision points, clarification loops, and exit conditions for the three primary business scenarios: **Shipment Tracking**, **Pickup Scheduling**, and **Delivery Delay Inquiry**.

---

## 1. Shipment Tracking Workflow

This diagram demonstrates the flow when a user requests the status of a package.

```mermaid
graph TD
    %% Define Styles
    classDef start_end fill:#f9f,stroke:#333,stroke-width:2px;
    classDef decision fill:#ff9,stroke:#333,stroke-width:2px;
    classDef action fill:#bbf,stroke:#333,stroke-width:1px;
    
    A([Start: Cold Boot / Initiate Session]) --> B[Greeting: 'How may I assist you today?']
    B --> C{User Input / Request}
    
    %% Intent Detection
    C -->|Intent: Track Shipment| D{Has Tracking ID SHxxx?}
    
    %% Direct Success Path
    D -->|Yes: Valid SHxxx ID| E[Database Lookup: Fetch Shipment Details]
    E --> F{Shipment Found?}
    F -->|Yes| G[Generate Localized Template Response bypassing LLM]
    G --> H[Convert Response to Speech and Play Audio]
    H --> I[Transition to WAITING_FOR_NEXT_TURN]
    
    %% Empty or Invalid Tracking ID
    D -->|No ID / Partial ID| J[Set Stage: WAITING_FOR_TRACKING_ID]
    J --> K[Prompt User: 'Please provide your shipment ID']
    K --> L{New User Input}
    
    %% Clarification Loop & Deflection
    L -->|Off-Topic / Gibberish| M[Unexpected Input Deflection Interception]
    M --> N[Play Deflection Prompt: Stay on topic]
    N --> J
    
    L -->|Valid SHxxx ID Provided| E
    
    L -->|No ID Provided Again| O[Prompt user to check number and try again]
    O --> J
    
    %% Database Lookup Fallback
    F -->|No: Not Found in DB| P[Play: Shipment ID not found, please check and try again]
    P --> J
    
    %% Session End
    I --> Q{User Says Goodbye / Thank You?}
    Q -->|Yes| R([Exit: Play Goodbye Template & Close Session])
    Q -->|No| C
    
    class A,R start_end;
    class D,F,L,Q decision;
    class G,M,N,P,K action;
```

---

## 2. Pickup Scheduling Workflow (Slot-Filling)

This diagram documents the multi-turn slot-filling process to schedule a courier collection.

```mermaid
graph TD
    classDef start_end fill:#f9f,stroke:#333,stroke-width:2px;
    classDef decision fill:#ff9,stroke:#333,stroke-width:2px;
    classDef action fill:#bbf,stroke:#333,stroke-width:1px;

    A([Start: Session Initiated]) --> B[Greeting / User Choice]
    B -->|Intent: Schedule Pickup| C[Set Stage: PICKUP_WAITING_FOR_LOCATION]
    
    %% Slot 1: Location
    C --> D[Prompt User: 'Please provide the pickup location.']
    D --> E{User Input: Location}
    E -->|Gibberish / Off-Topic| F[Unexpected Input Deflection]
    F --> D
    E -->|Valid Location| G[Store Location in JSON slots]
    
    %% Slot 2: Date
    G --> H[Set Stage: PICKUP_WAITING_FOR_DATE]
    H --> I[Prompt User: 'Please provide the delivery date.']
    I --> J{User Input: Date}
    J -->|Gibberish / Off-Topic| K[Unexpected Input Deflection]
    K --> I
    J -->|Valid Date| L[Store Date in JSON slots]
    
    %% Slot 3: Weight
    L --> M[Set Stage: PICKUP_WAITING_FOR_WEIGHT]
    M --> N[Prompt User: 'Please provide the package weight.']
    N --> O{User Input: Weight}
    O -->|Gibberish / Off-Topic| P[Unexpected Input Deflection]
    P --> N
    O -->|Valid Weight| Q[Store Weight in JSON slots]
    
    %% Success Completion
    Q --> R[Call DB: Create New Shipment Row]
    R --> S[Generate Random Tracking ID SHxxx]
    S --> T[Build Dynamic Scheduled Template Response]
    T --> U[Play Success Audio: Includes new tracking ID]
    U --> V[Reset Stage to GREETING / Ready for next turn]
    
    V --> W([Exit / Wait for next turn])
    
    class A,W start_end;
    class E,J,O decision;
    class F,K,P,R,T action;
```

---

## 3. Delivery Delay Inquiry Workflow

This diagram represents the query flow when a user reports or asks about a late package.

```mermaid
graph TD
    classDef start_end fill:#f9f,stroke:#333,stroke-width:2px;
    classDef decision fill:#ff9,stroke:#333,stroke-width:2px;
    classDef action fill:#bbf,stroke:#333,stroke-width:1px;

    A([Start: Session Initiated]) --> B[Greeting / User Choice]
    B -->|Intent: Delayed Shipment| C{Has Tracking ID SHxxx?}
    
    %% Request ID if missing
    C -->|No ID Provided| D[Set Stage: WAITING_FOR_DELAY_ID]
    D --> E[Prompt User: 'Please provide your shipment ID for the delay inquiry']
    E --> F{New User Input}
    
    F -->|Gibberish / Off-Topic| G[Unexpected Input Deflection]
    G --> E
    
    F -->|Valid ID Provided| H[Database Lookup: Fetch details]
    C -->|Yes: ID Provided| H
    
    %% Check Delay Statuses
    H --> I{Shipment Found?}
    I -->|No| J[Prompt User: 'Shipment ID not found, please check and try again']
    J --> D
    
    I -->|Yes| K{Check Shipment Status}
    
    K -->|Status: Delivered| L[Explain package has already been delivered]
    K -->|Status: Cancelled| M[Explain package was cancelled]
    K -->|Status: In Transit / Pending| N[Generate Delayed Template Response bypassing LLM]
    
    L --> O[Play Response Audio & transition stage to GREETING]
    M --> O
    N --> O
    
    O --> P([Exit / Wait for next turn])
    
    class A,P start_end;
    class C,F,I,K decision;
    class G,J,L,M,N action;
```

---

## 4. Timeout Inactivity Workflow (Two-Stage Idle Rule)

This diagram documents the background automated logic checking inactivity and closing sessions safely.

```mermaid
graph TD
    classDef start_end fill:#f9f,stroke:#333,stroke-width:2px;
    classDef decision fill:#ff9,stroke:#333,stroke-width:2px;
    classDef action fill:#bbf,stroke:#333,stroke-width:1px;

    A([Start: User finishes a Turn]) --> B[Record last_interaction_time = time.time]
    B --> C[Loop: Periodically calculate elapsed = current_time - last_interaction_time]
    C --> D{Is elapsed > 30 seconds?}
    
    %% Stage 1 Warning
    D -->|Yes| E{Warning already triggered?}
    E -->|No| F[Flag: warning_triggered = True]
    F --> G[Speak Prompt: 'Are you still there? Please let me know...']
    G --> C
    
    E -->|Yes| H{Is elapsed > 60 seconds?}
    D -->|No| C
    
    %% Stage 2 Exit
    H -->|Yes| I[Print Terminal: Inactivity limit reached]
    I --> J[HTTP POST to /end_session endpoint]
    J --> K[Play Localized Goodbye response matching Sticky Language]
    K --> L[Release sound locks, exit client process safely]
    L --> M([Exit: Session Closed])
    
    H -->|No| C
    
    class A,M start_end;
    class D,E,H decision;
    class G,J,K action;
```
