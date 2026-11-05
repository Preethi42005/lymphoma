import { createFileRoute } from "@tanstack/react-router";
import { streamText, convertToModelMessages, type UIMessage } from "ai";
import { createLovableAiGatewayProvider } from "@/lib/ai-gateway.server";

const SYSTEM_PROMPT = `You are LymphAssist, an educational medical AI focused on lymphoma and its subtypes: Mantle Cell Lymphoma (MCL), Follicular Lymphoma (FL), Chronic Lymphocytic Leukemia / Small Lymphocytic Lymphoma (CLL/SLL), Diffuse Large B-Cell Lymphoma (DLBCL), and related non-Hodgkin & Hodgkin lymphomas.

Guidelines:
- Explain histopathological features, immunophenotypes (CD5, CD10, CD20, CD23, cyclin D1, BCL2, BCL6), staging (Ann Arbor / Lugano), treatments (R-CHOP, BR, BTK inhibitors, CAR-T), and prognosis in clear, structured language.
- Use markdown: short headings, bullet points, and bold key terms.
- Always append a brief disclaimer that responses are educational and NOT a substitute for professional medical diagnosis or advice.
- If asked something unrelated to lymphoma/hematology/oncology, politely redirect.`;

export const Route = createFileRoute("/api/chat")({
  server: {
    handlers: {
      POST: async ({ request }) => {
        const { messages } = (await request.json()) as { messages: UIMessage[] };
        if (!Array.isArray(messages)) {
          return new Response("Messages required", { status: 400 });
        }
        const key = process.env.LOVABLE_API_KEY;
        
        // Fallback to mock response when API key is missing
        if (!key) {
          const lastMessage = messages[messages.length - 1];
          const userQuery = lastMessage?.parts?.[0]?.text || "";
          
          // Simple mock response based on keywords
          let response = "";
          const query = userQuery.toLowerCase();
          
          if (query.includes("mcl") || query.includes("mantle")) {
            response = `**Mantle Cell Lymphoma (MCL)**

**Immunophenotype:**
- CD5+ (positive)
- CD23- (negative)  
- Cyclin D1+ (positive)
- CD20+
- Surface IgM/D+

**Genetic hallmark:** t(11;14) translocation leading to cyclin D1 overexpression.

**Treatment:**
- First-line: R-CHOP or bendamustine + rituximab (BR)
- For eligible patients: intensive regimens (R-HyperCVAD/MA) or autologous stem cell transplant
- Targeted therapies: BTK inhibitors (ibrutinib, acalabrutinib)

*Note: This is educational information only. Consult a hematologist for diagnosis and treatment decisions.*`;
          } else if (query.includes("cll") || query.includes("chronic")) {
            response = `**Chronic Lymphocytic Leukemia / Small Lymphocytic Lymphoma (CLL/SLL)**

**Immunophenotype:**
- CD5+ (positive)
- CD23+ (positive)
- Weak surface immunoglobulin
- CD20 (dim)
- CD79b (dim)

**Staging:** Rai or Binet staging systems based on lymphadenopathy, organomegaly, and cytopenias.

**Treatment:**
- Watchful waiting for asymptomatic early-stage disease
- First-line: BTK inhibitors (ibrutinib, acalabrutinib) ± anti-CD20 antibodies
- Chemoimmunotherapy: FCR, BR, or obinutuzumab + chlorambucil
- Novel agents: BCL2 inhibitor (venetoclax)

*Note: This is educational information only. Consult a hematologist for diagnosis and treatment decisions.*`;
          } else if (query.includes("fl") || query.includes("follicular")) {
            response = `**Follicular Lymphoma (FL)**

**Immunophenotype:**
- CD10+ (positive)
- BCL2+ (positive)
- BCL6+ (positive)
- CD20+
- CD5- (negative)

**Genetic hallmark:** t(14;18) translocation leading to BCL2 overexpression.

**Grading (WHO):**
- Grade 1-2: <15 centroblasts per HPF
- Grade 3A: ≥15 centroblasts per HPF (still has centrocytes)
- Grade 3B: sheets of centroblasts (treated like DLBCL)

**Treatment:**
- Grade 1-2: R-CHOP, bendamustine + rituximab, or rituximab monotherapy
- Grade 3: R-CHOP (similar to DLBCL approach)
- Relapsed/refractory: PI3K inhibitors, CAR-T, or bispecific antibodies

*Note: This is educational information only. Consult a hematologist for diagnosis and treatment decisions.*`;
          } else if (query.includes("dlbcl") || query.includes("diffuse")) {
            response = `**Diffuse Large B-Cell Lymphoma (DLBCL)**

**Immunophenotype:**
- CD20+ (positive)
- CD79a+
- PAX5+
- Variable: CD10, BCL6, BCL6, MUM1 (cell-of-origin classification)

**Subtypes:**
- GCB (Germinal Center B-cell) type: CD10+, BCL6+
- ABC (Activated B-cell) type: MUM1+, CD10-

**Treatment:**
- First-line: R-CHOP (rituximab, cyclophosphamide, doxorubicin, vincristine, prednisone)
- For high-risk or double-hit: DA-EPOCH-R
- Relapsed/refractory: CAR-T (axi-cel, tisa-cel), bispecific antibodies, or salvage chemotherapy + transplant

*Note: This is educational information only. Consult a hematologist for diagnosis and treatment decisions.*`;
          } else if (query.includes("stage") || query.includes("staging")) {
            response = `**Lymphoma Staging Systems**

**Ann Arbor Staging (Classic):**
- Stage I: Single lymph node region or extralymphatic site
- Stage II: Two regions on same side of diaphragm
- Stage III: Regions on both sides of diaphragm
- Stage IV: Disseminated involvement of extralymphatic organs

**Modifiers:**
- A: No systemic symptoms
- B: Fever, night sweats, weight loss >10%
- X: Bulky disease (>10 cm or >1/3 chest width)
- E: Extranodal extension

**Lugano Staging (Modern PET-based):**
- Uses same I-IV framework
- Incorporates PET-CT for response assessment
- Deauville score (1-5) for treatment response

*Note: This is educational information only. Consult a hematologist for staging and treatment decisions.*`;
          } else {
            response = `I can help with questions about lymphoma subtypes including:

- **MCL** (Mantle Cell Lymphoma)
- **FL** (Follicular Lymphoma)  
- **CLL** (Chronic Lymphocytic Leukemia)
- **DLBCL** (Diffuse Large B-Cell Lymphoma)

I can provide information on:
- Histopathological features
- Immunophenotypes (CD markers)
- Staging systems (Ann Arbor, Lugano)
- Treatment approaches

Please ask a specific question about any of these topics.

*Note: This is educational information only. Not a substitute for professional medical advice.*`;
          }

          // Return as a streaming response
          const stream = new ReadableStream({
            async start(controller) {
              const encoder = new TextEncoder();
              // Send the response in chunks
              const chunks = response.split(/(?=\n\n|\n\*\*)/g);
              for (const chunk of chunks) {
                await new Promise(r => setTimeout(r, 50));
                controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: "text-delta", delta: chunk })}\n\n`));
              }
              controller.enqueue(encoder.encode("data: [DONE]\n\n"));
              controller.close();
            }
          });

          return new Response(stream, {
            headers: {
              "Content-Type": "text/event-stream",
              "Cache-Control": "no-cache",
              "Connection": "keep-alive",
            },
          });
        }

        const gateway = createLovableAiGatewayProvider(key);
        const result = streamText({
          model: gateway("google/gemini-3.6-flash"),
          system: SYSTEM_PROMPT,
          messages: await convertToModelMessages(messages),
        });

        return result.toUIMessageStreamResponse({ originalMessages: messages });
      },
    },
  },
});
