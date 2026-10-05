// The shapes the API sends and receives. They mirror the Pydantic models in app/schemas.
// TypeScript checks every use of these fields while the interface is built, so a field
// that is renamed on the server shows up here as a build error, not as a blank screen.

export type RiskLevel = "low" | "medium" | "high" | "critical";

export interface User {
  id: string;
  name: string;
  email: string | null;
  email_verified: boolean;
  created_at: string;
  has_password: boolean;
  has_google: boolean;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user: User;
}

export interface AuthConfig {
  auth_mode: "jwt" | "dev";
  registration_open: boolean;
  signup_code_required: boolean;
  google_client_id: string | null;
  min_password_chars: number;
}

export interface Health {
  status: string;
  version: string;
  database: string;
  llm_provider: string;
  llm_model: string;
  embedding_provider: string;
}

export interface DocumentInfo {
  id: string;
  filename: string;
  document_type: string;
  uploaded_at: string;
  page_count: number;
  status: "processing" | "ready" | "failed";
  chunk_count: number;
  error: string | null;
  duplicate: boolean;
  integrity_level: "none" | "low" | "medium" | "high" | null;
  integrity_score: number | null;
}

export interface Citation {
  document: string;
  page: number;
  document_id: string | null;
  quote: string | null;
}

export interface Finding {
  type: "fact" | "contradiction" | "missing_evidence" | "risk" | "calculation" | "recommendation";
  statement: string;
  severity: string | null;
  confidence: number | null;
  citations: Citation[];
}

export interface Contradiction {
  id: string;
  type: string;
  severity: string;
  description: string;
  key: string;
  value_a: number | null;
  value_b: number | null;
  unit_a: string | null;
  unit_b: string | null;
  text_a: string | null;
  text_b: string | null;
  source_a: string;
  source_b: string;
  difference: number | null;
  percentage_change: number | null;
  citations: Citation[];
}

export interface IntegrityIndicator {
  type: string;
  reason: string;
  page: number | null;
  region: string | null;
}

export interface DocumentIntegrity {
  id: string;
  document: string;
  document_id: string | null;
  level: string;
  score: number;
  indicators: IntegrityIndicator[];
}

export interface FollowUpAction {
  id: string;
  kind: string;
  server: string;
  target: string;
  title: string;
  body: string | null;
  status: "proposed" | "not_approved" | "done" | "failed" | "unknown";
  reference: string | null;
  url: string | null;
  detail: string | null;
}

export interface Calculation {
  id: string;
  description: string;
  tool_name: string;
  arguments: Record<string, unknown>;
  result: number | null;
  requested_by: string;
}

export interface HumanReview {
  required: boolean;
  status: "not_required" | "not_requested" | "pending" | "approved" | "rejected" | "modified";
  reviewer: string | null;
  note: string | null;
  decided_at: string | null;
  original_action: string | null;
}

export interface Report {
  investigation_id: string;
  question: string;
  status: string;
  status_label: string;
  risk_level: RiskLevel;
  confidence: number;
  evidence_quality: string;
  requires_human_review: boolean;
  human_review: HumanReview;
  summary: string;
  key_findings: Finding[];
  contradictions: Contradiction[];
  document_integrity: DocumentIntegrity[];
  actions: FollowUpAction[];
  missing_evidence: string[];
  calculations: Calculation[];
  risk_reasons: Finding[];
  recommended_action: string;
  sources: Citation[];
  limitations: string[];
  stats: {
    iterations: number;
    llm_calls: number;
    retrieval_rounds: number;
    retrieved_chunks: number;
    evidence_items: number;
    discarded_facts: number;
    tool_calls: number;
    duration_ms: number;
    llm_provider: string;
  };
  memory: { saved: string[]; recalled: { text: string; source: string; similarity: number }[] };
  answer_text: string;
}

export interface InvestigationSummary {
  id: string;
  question: string;
  status: string;
  approval_status: string;
  error: string | null;
  risk_level: RiskLevel | null;
  confidence_score: number | null;
  created_at: string;
  completed_at: string | null;
}

export interface InvestigationDetail extends InvestigationSummary {
  report: Report | null;
}

export interface ChatAnswer {
  conversation_id: string;
  investigation_id: string;
  answer: string;
  risk_level: RiskLevel;
  confidence: number;
  status: string;
  requires_human_review: boolean;
  approval_status: string;
  citations: Citation[];
}

export interface ApprovalDecision {
  decision: "approve" | "reject" | "modify";
  note?: string;
  modified_action?: string;
  approved_actions: string[];
}

export interface ProfileField {
  text: string | null;
  number: number | null;
  unit: string | null;
  dates: string[];
  role: "supplier" | "customer" | null;
  page: number;
  quote: string;
}

export interface LineItem {
  description: string;
  quantity: number | null;
  unit_price: number | null;
  amount: number | null;
  unit: string | null;
  page: number;
  quote: string;
}

export interface ForensicReport {
  analysed: boolean;
  score: number;
  level: "none" | "low" | "medium" | "high";
  indicators: (IntegrityIndicator & { category: string; weight: number })[];
  metadata: Record<string, string | number>;
  note: string | null;
}

export interface ForensicsOut {
  document_id: string;
  filename: string;
  report: ForensicReport | null;
  note: string | null;
}

export interface ComparisonCell {
  value: string;
  page: number;
  quote: string;
}

export interface Comparison {
  documents: { document_id: string; filename: string; document_type: string; profiled: boolean; built_by: string | null }[];
  rows: { field: string; label: string; cells: (ComparisonCell | null)[]; differs: boolean }[];
  line_items: { document_id: string; filename: string; items: LineItem[] }[];
  findings: (Omit<Contradiction, "id"> & { key: string })[];
  notes: string[];
}

export interface ActionConfig {
  enabled: boolean;
  server: string | null;
  target: string | null;
  problem: string | null;
}

export interface Preference {
  key: string;
  description: string;
  allowed_values: string[];
  default: string;
  value: string;
  is_set: boolean;
}

export interface MemoryNote {
  id: string;
  text: string;
  source: string;
  created_at: string;
}

export interface MemoryOverview {
  enabled: boolean;
  preferences: Preference[];
  notes: MemoryNote[];
}

// One event of a streamed chat turn (see app/api/sse.py).
export type StreamEvent =
  | { event: "started"; data: { conversation_id: string; investigation_id: string } }
  | { event: "progress"; data: { node: string; message: string } }
  | { event: "tool"; data: { tool_name: string; status: string } }
  | { event: "report"; data: Report }
  | { event: "answer"; data: ChatAnswer }
  | { event: "error"; data: { code: string; message: string } };
