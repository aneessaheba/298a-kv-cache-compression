"""Generate a concise PDF summary of the TriAttention-on-LongBench work."""
from pathlib import Path
from fpdf import FPDF

HERE = Path(__file__).resolve().parent
FIG = HERE / "results_out" / "figures"
OUT = HERE / "TriAttention_Summary.pdf"


def main():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def h(txt, size=15):
        pdf.set_font("Helvetica", "B", size)
        pdf.multi_cell(0, 8, txt)
        pdf.ln(1)

    def p(txt, size=10.5):
        pdf.set_font("Helvetica", "", size)
        pdf.multi_cell(0, 5.5, txt)
        pdf.ln(1.5)

    h("TriAttention KV-Cache Compression on LongBench", 16)
    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(0, 5.5, "Member: Samruddhi | Model: Llama-3.1-8B-Instruct (bf16) | Benchmark: LongBench v1")
    pdf.ln(3)

    h("Goal")
    p("Test the TriAttention KV-cache compression method on LongBench, document the results, and "
      "identify where the method falls short so the flaws can be targeted for improvement.")

    h("What TriAttention does")
    p("During generation a language model stores a key/value for every token (the KV cache), which "
      "grows with input length and consumes GPU memory. TriAttention shrinks it by scoring each cached "
      "key with a frequency-domain trigonometric series built from precomputed per-head query "
      "statistics, then keeping the highest-scoring tokens plus the recent window and initial sink "
      "tokens. It is run faithfully here using the authors' calibration procedure, not an approximation.")

    h("What we did")
    p("1. Data: built the LongBench dev set (800 examples, 50 per task x 16 tasks) with the Llama-3.1-8B "
      "tokenizer, matching the team's pipeline.")
    p("2. Method: implemented TriAttention as a kvpress ScorerPress and ran the authors' exact scoring.")
    p("3. Calibration: reproduced the authors' calibration (one forward pass over plain text, invert "
      "RoPE, per-head complex-frequency query stats) so results reflect the real method. An earlier "
      "online-estimate version was discarded because it invented and exaggerated flaws.")
    p("4. Chunked prefill: added chunk-by-chunk prefill so long documents fit in 32 GB GPU memory, "
      "enabling the full 800-record run (earlier only 260 short-prompt records fit).")
    p("5. Reporting: automated logs, CSVs, charts, animations, and the RESULTS/FLAWS documents.")

    h("Results (260-record subset, confirmed)")
    p("Halving the KV cache cost ~2 accuracy points; pushing to a 13x smaller cache dropped the macro "
      "score from 45.2 to 31.5. Full cache 45.24 | @0.5 43.18 | @0.75 37.23 | @0.9 32.54 | @0.95 31.50 "
      "(macro average out of 100).")

    fig = FIG / "score_vs_kv_cache.png"
    if fig.exists():
        pdf.image(str(fig), w=150)
        pdf.ln(2)

    h("The flaw we found")
    p("TriAttention hurts question-answering the most (multifieldqa_en, 2wikimqa, qasper drop sharply), "
      "while code, few-shot, and summarization hold up. QA answers depend on a few specific evidence "
      "tokens scattered in a long document; TriAttention scores keys by expected attention over query "
      "distance/geometry, not by the actual question, so those isolated evidence tokens are evicted. "
      "That is the concrete lever to improve: make the score sensitive to the question/evidence.")

    h("What is running now")
    p("The full 800-record run with chunked prefill: the full-cache baseline plus TriAttention at four "
      "compression ratios (0.5, 0.75, 0.9, 0.95), covering all prompt lengths including the longest "
      "(32K-token) documents that were skipped before. TriAttention processes these without running out "
      "of memory; the uncompressed baseline records out-of-memory on the very longest documents, which "
      "itself shows that full cache cannot handle inputs TriAttention can. On completion the run "
      "regenerates all CSVs, charts, and the RESULTS/FLAWS documents on the full 800 records.")

    pdf.output(str(OUT))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
