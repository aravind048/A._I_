from pathlib import Path


KNOWLEDGE_PATH = Path("knowledge")


def load_document(file_path):
    path = Path(file_path)
    return path.read_text(encoding="utf-8")


def get_content_type(title):
    title = title.lower()

    if "misconception" in title:
        return "misconception"

    if "worked example" in title:
        return "worked_example"

    if "assignment" in title:
        return "concept"

    if "mutable" in title:
        return "concept"

    if "condition" in title:
        return "concept"

    if "loop" in title:
        return "concept"

    if "variable" in title:
        return "concept"

    return "concept"


def chunk_markdown(text, source_file):
    sections = text.split("## ")

    chunks = []

    for index, section in enumerate(sections):

        section = section.strip()

        if not section:
            continue

        lines = section.split("\n")

        title = lines[0].strip()

        content = "\n".join(
            lines[1:]
        ).strip()

        if content:

            chunks.append({
                "chunk_id": (
                    f"{Path(source_file).stem.upper()}_"
                    f"{index:03d}"
                ),
                "concept": Path(
                    source_file
                ).stem,
                "source": str(source_file),
                "title": title,
                "content_type": get_content_type(
                    title
                ),
                "content": content
            })

    return chunks


def load_all_chunks():

    all_chunks = []

    for file_path in sorted(
        KNOWLEDGE_PATH.glob("*.md")
    ):

        document = load_document(file_path)

        chunks = chunk_markdown(
            document,
            file_path
        )

        all_chunks.extend(chunks)

    return all_chunks