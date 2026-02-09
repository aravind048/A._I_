import typer
import json
from rich import print
from rich.console import Console
from rich.json import JSON
from corep_assistant.config import settings
from corep_assistant.logging_config import setup_logging
from corep_assistant.ingestion.loader import load_txt_documents
from corep_assistant.ingestion.chunker import chunk_documents
from corep_assistant.ingestion.writer import write_chunks_jsonl
from corep_assistant.retrieval.embedder import Embedder
from corep_assistant.retrieval.chunks_reader import read_chunks_jsonl
from corep_assistant.retrieval.vector_store import FaissVectorStore
from corep_assistant.retrieval.retriever import Retriever
from corep_assistant.pipeline.orchestrator import run_ca1_pipeline
from corep_assistant.pipeline.scenario_models import ScenarioCA1
from corep_assistant.retrieval.manifest import load_index_manifest
from corep_assistant.runs.saver import make_run_dir, save_run_bundle

app = typer.Typer(help="COREP Reporting Assistant (Prototype)")

console = Console()


@app.command()
def index(
    chunks_path: str = typer.Option("./data/processed/chunks.jsonl", "--chunks", "-c"),
    index_dir: str = typer.Option(None, "--index-dir", help="Defaults to settings.index_dir"),
    model_name: str = typer.Option("sentence-transformers/multi-qa-mpnet-base-dot-v1", "--model"),
    batch_size: int = typer.Option(32, "--batch-size"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing index files"),
):
    """
    Build FAISS index from chunks.jsonl.

    Input:
      - chunks_path: JSONL produced by `corep ingest`
      - index_dir: directory to store faiss.index + meta.jsonl + manifest
      - model_name: embedding model

    Output:
      - prints stats and index paths
    """
    setup_logging()
    index_dir = index_dir or settings.index_dir

    records = read_chunks_jsonl(chunks_path)
    if not records:
        print(f"[bold red]No chunks found in {chunks_path}[/bold red]")
        raise typer.Exit(code=1)

    store = FaissVectorStore(index_dir=index_dir)

    # ✅ NEW: prevent silent overwrite
    if (store.index_path.exists() or store.meta_path.exists() or store.manifest_path.exists()) and not force:
        print("[bold red]Index already exists.[/bold red]")
        print(f"Index dir: {index_dir}")
        print("Use --force to overwrite.")
        raise typer.Exit(code=2)

    texts = [r.text for r in records]

    print(f"[bold cyan]Embedding {len(texts)} chunks[/bold cyan]")
    embedder = Embedder(model_name=model_name)
    embeddings = embedder.embed_texts(texts, batch_size=batch_size)

    # ✅ NEW: free duplicate list (small but real improvement)
    del texts

    store.build_and_save(
        embeddings=embeddings,
        records=records,
        embed_model_name=model_name
    )

    print("[bold green]Index build complete[/bold green]")
    print(f"Index dir:   {index_dir}")
    print(f"Vectors:     {embeddings.shape[0]}")
    print(f"Dimension:   {embeddings.shape[1]}")
    print(f"Files:       {store.index_path.name}, {store.meta_path.name}, {store.manifest_path.name}")
    print(f"[bold cyan]Retrieval model:[/bold cyan] {model_name}")


@app.command()
def ingest(
    input_dir: str = typer.Option("./data/raw", "--input", "-i"),
    output_path: str = typer.Option(
        "./data/processed/chunks.jsonl", "--out", "-o"),
):
    """
    Ingest raw .txt docs -> chunk -> write JSONL.

    Input:
      - input_dir: directory with .txt files
      - output_path: JSONL output path

    Output:
      - prints counts and where file is written
    """
    setup_logging()

    docs = load_txt_documents(input_dir)
    if not docs:
        print(f"[bold red]No .txt documents found in {input_dir}[/bold red]")
        raise typer.Exit(code=1)

    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    written = write_chunks_jsonl(chunks, output_path)

    total_chars = sum(len(d.text) for d in docs)
    print("[bold green]Ingestion complete[/bold green]")
    print(f"Documents loaded: {len(docs)}")
    print(f"Total characters: {total_chars}")
    print(f"Chunks generated: {len(chunks)}")
    print(f"Chunks written:   {written}")
    print(f"Output file:      {output_path}")


@app.command()
def info():
    """
    Show current configuration (sanity check).
    Input: none
    Output: prints settings
    """
    setup_logging()
    print("[bold cyan]COREP Assistant Config[/bold cyan]")
    print(settings.model_dump())


@app.command()
def ask(
    question: str = typer.Option(..., "--question", "-q"),
    scenario_json: str = typer.Option(..., "--scenario",
                                      "-s", help="Scenario as JSON string"),

    template: str = typer.Option(
        "CA1", "--template", help="Template key, currently supports: CA1"),
    index_dir: str = typer.Option(
        None, "--index-dir", help="Path to FAISS index directory"),
    top_k: int = typer.Option(None, "--top-k", "-k",
                              help="Number of retrieved chunks"),
    embed_model: str = typer.Option(
        "sentence-transformers/multi-qa-mpnet-base-dot-v1", "--embed-model", help="Embedding model name"),
    preview_chars: int = typer.Option(
        220, "--preview", help="Snippet length for retrieved chunks"),
):
    """
    End-to-end query:
    question + scenario -> retrieval -> structured output -> template extract -> validations -> audit log.
    """
    setup_logging()

    # 1) Parse scenario JSON
    try:
        scenario_dict = json.loads(scenario_json)
        if not isinstance(scenario_dict, dict):
            raise ValueError(
                "Scenario JSON must be a JSON object at top-level.")
    except Exception as e:
        print("[bold red]Invalid --scenario JSON.[/bold red]")
        print("[yellow]You provided:[/yellow]", scenario_json)
        print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=2)

    # 2) Resolve defaults
    index_dir = index_dir or settings.index_dir
    top_k = top_k or settings.top_k
    template = template.upper().strip()

    if template != "CA1":
        print(
            f"[bold red]Unsupported template '{template}'.[/bold red] Only 'CA1' is implemented right now.")
        raise typer.Exit(code=2)

    # 3) Validate scenario shape early (clear errors)
    try:
        scenario = ScenarioCA1.model_validate(scenario_dict).model_dump()
    except Exception as e:
        print("[bold red]Scenario schema validation failed.[/bold red]")
        print("[yellow]Expected shape:[/yellow]")
        print({
            "reporting_date": "YYYY-MM-DD",
            "currency": "GBP",
            "entity": {"legal_entity_id": "BANK1", "consolidation_level": "SOLO"},
            "inputs": {
                "cet1_before_deductions": 120,
                "intangibles_deduction": 8,
                "dta_future_profit_deduction": 5,
                "at1": 10,
                "t2": 15
            }
        })
        print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=2)

    # 4) Embedding model compatibility check (prevents silent garbage retrieval)
    try:
        manifest = load_index_manifest(index_dir)
        indexed_model = manifest.get("embed_model_name")
        if indexed_model and indexed_model != embed_model:
            print("[bold red]Embedding model mismatch![/bold red]")
            print(f"Index built with:   {indexed_model}")
            print(f"You are querying with: {embed_model}")
            print("[yellow]Fix:[/yellow] Rebuild index using the same model as --embed-model, or query using the indexed model.")
            raise typer.Exit(code=2)
    except FileNotFoundError:
        print(
            "[bold red]Index manifest not found.[/bold red] Did you run `corep index`?")
        raise typer.Exit(code=2)

    # 5) Run pipeline
    result = run_ca1_pipeline(
        question=question,
        scenario=scenario,
        index_dir=index_dir,
        embed_model_name=embed_model,
        top_k=top_k
    )

    # 6) Pretty output
    print("[bold cyan]Retrieved passages[/bold cyan]")
    for r in result["retrieved"]:
        snippet = r["text"].replace("\n", " ")
        snippet = snippet[:preview_chars] + \
            ("..." if len(snippet) > preview_chars else "")
        print(
            f"- score={r['score']:.4f} source={r['source_id']} chunk={r['chunk_id']}")
        print(f"  [dim]{snippet}[/dim]")

    print("\n[bold green]Structured Output (schema)[/bold green]")
    console.print(JSON.from_data(result["structured"]))

    print("\n[bold magenta]COREP Template Extract[/bold magenta]")
    for row in result["template_extract"]:
        print(
            f"{row['field_id']:12} | {row['label']:<45} | {row['value']} {row['unit']}")

    print("\n[bold yellow]Validation[/bold yellow]")
    if not result["validation"]:
        print("No issues.")
    else:
        for v in result["validation"]:
            print(
                f"{v['severity']} {v['rule_id']}: {v['message']} affected={v['affected_fields']}")

    print("\n[bold white]Audit Log[/bold white]")
    for a in result["audit_log"]:
        print(f"{a['field_id']} => citations={a['citations']}")

    cov = result["structured"].get("debug", {}).get("citation_coverage", None)
    if cov is not None:
        print(f"\n[bold blue]Citation coverage:[/bold blue] {cov:.2f}")


@app.command()
def retrieve(
    question: str = typer.Option(..., "--question", "-q"),
    index_dir: str = typer.Option(
        None, "--index-dir", help="Defaults to settings.index_dir"),
    model_name: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2", "--model"),
    top_k: int = typer.Option(None, "--top-k", "-k",
                              help="Defaults to settings.top_k"),
    preview_chars: int = typer.Option(240, "--preview"),
):
    """
    Retrieve top-k relevant chunks for a query.

    Output:
      - prints ranked chunks with score + ids + snippet
    """
    setup_logging()
    index_dir = index_dir or settings.index_dir
    top_k = top_k or settings.top_k

    embedder = Embedder(model_name=model_name)
    r = Retriever(index_dir=index_dir, embedder=embedder)

    results = r.retrieve(question, top_k=top_k)
    print(f"[bold cyan]Retrieval model:[/bold cyan] {model_name}")

    if not results:
        print("[bold red]No results returned.[/bold red]")
        raise typer.Exit(code=2)

    print(f"[bold green]Top {len(results)} matches[/bold green]")
    for item in results:
        snippet = item.text.replace("\n", " ")
        snippet = snippet[:preview_chars] + \
            ("..." if len(snippet) > preview_chars else "")
        print("")
        print(
            f"[bold]score:[/bold] {item.score:.4f} | [bold]source:[/bold] {item.source_id} | [bold]chunk:[/bold] {item.chunk_id}")
        print(f"[dim]{snippet}[/dim]")

@app.command()
def run(
    question: str = typer.Option(..., "--question", "-q"),
    scenario_json: str = typer.Option(..., "--scenario", "-s", help="Scenario as JSON string"),

    template: str = typer.Option("CA1", "--template", help="Template key, currently supports: CA1"),
    index_dir: str = typer.Option(None, "--index-dir", help="Path to FAISS index directory"),
    top_k: int = typer.Option(None, "--top-k", "-k", help="Number of retrieved chunks"),
    embed_model: str = typer.Option("sentence-transformers/multi-qa-mpnet-base-dot-v1", "--embed-model", help="Embedding model name"),

    out_dir: str = typer.Option("./runs", "--out", help="Base output directory for saved runs"),
):
    """
    Run end-to-end pipeline AND save a run bundle to disk.
    """
    setup_logging()

    # Parse scenario JSON
    try:
        scenario_dict = json.loads(scenario_json)
        if not isinstance(scenario_dict, dict):
            raise ValueError("Scenario JSON must be a JSON object at top-level.")
    except Exception as e:
        print("[bold red]Invalid --scenario JSON.[/bold red]")
        print("[yellow]You provided:[/yellow]", scenario_json)
        print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=2)

    # Resolve defaults
    index_dir = index_dir or settings.index_dir
    top_k = top_k or settings.top_k
    template = template.upper().strip()

    if template != "CA1":
        print(f"[bold red]Unsupported template '{template}'.[/bold red] Only 'CA1' is implemented right now.")
        raise typer.Exit(code=2)

    # (Optional) Keep your existing manifest model mismatch check here if you already added it in ask()
    # Highly recommended.

    # Run pipeline
    result = run_ca1_pipeline(
        question=question,
        scenario=scenario_dict,
        index_dir=index_dir,
        embed_model_name=embed_model,
        top_k=top_k
    )

    # Save run bundle
    run_dir = make_run_dir(out_dir, template=template)
    saved = save_run_bundle(run_dir, result)

    print("[bold green]Run saved[/bold green]")
    print(f"Run directory: {run_dir}")
    print("Saved files:")
    for k, v in saved.items():
        print(f"- {k}: {v}")

    # Quick summary for humans
    cov = (result.get("structured", {}).get("debug", {}) or {}).get("citation_coverage", None)
    print("")
    print(f"[bold cyan]Summary[/bold cyan]")
    print(f"llm_mode: {result.get('llm_mode')}")
    print(f"retrieved_k: {len(result.get('retrieved', []))}")
    print(f"validation_count: {len(result.get('validation', []))}")
    if cov is not None:
        print(f"citation_coverage: {cov:.2f}")



if __name__ == "__main__":
    app()
